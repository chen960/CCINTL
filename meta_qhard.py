from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
from datetime import timedelta

import sklearn
from sklearn.cluster import DBSCAN
import collections
import torch
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from reid import models
from reid.models.em import Memory
from reid.models.cm import ClusterMemory
from reid.trainers import MetaTrainer, Trainer_USL
from reid.qhard_trainer import MetaTrainer_qhard
from reid.evaluators import Evaluator, extract_features
from collections import defaultdict
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.faiss_rerank import compute_jaccard_distance
from reid.utils import split_cam, generate_pseudo_labels,generate_cluster_features
from reid import datasets
from reid.utils.tools import get_test_loader, get_metatrain_loader, get_plot_loader, get_vanilla_loader
from reid.utils.tsne import plotTSNE
from scipy import io
import math
import os
import wandb
os.environ["WANDB_API_KEY"] = 'a5fa7a2c73723de7b928bb488e8059eeb21a98cc'
os.environ["WANDB_MODE"] = "dryrun"


start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    model = model.cuda()
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.copy_weight(checkpoint['state_dict'])
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True
    if args.symmetric:
        logs_dir = osp.join(args.logs_dir+'_'+str(args.dataset)+'_epochs'+str(args.epochs)+'dsce')
    else:
        logs_dir = osp.join(args.logs_dir+'_'+str(args.dataset)+'_epochs'+str(args.epochs))
    # sys.stdout = Logger(osp.join(logs_dir, 'log.txt'))

    wandb.init(project='meta_asoftmax')
    logs_dir = wandb.run.dir
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
        return

    import shutil
    shutil.copytree('./reid/', osp.join(logs_dir, 'reid'))
    shutil.copy('./meta_qhard.py',logs_dir)
    shutil.copy('./nometa_qhard.py', logs_dir)

    # # for vis
    marCamSet = get_data('dukeCam', args.data_dir)
    marCamSet = get_data('marCam', args.data_dir)
    mar_loader = get_plot_loader(marCamSet, args.height, args.width,
                                 args.batch_size, args.workers, test_set=marCamSet.train)



    # optimizer for meta models
    params = [{"params": [value]} for value in model.module.params() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

# ------------重载checkpoint的epoch optimizer和lr scheduler
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        del checkpoint
    if args.resume:
        print("==>Training from {} to {}".format(start_epoch, args.epochs))
    # Initialize target-domain instance features

    cluster_loader = get_test_loader(dataset, args.height, args.width,
                                     args.batch_size, args.workers, testset=sorted(dataset.train))
    if not args.resume:        # 如果不是从头训练  则不需要初始化
    # if False:
        print("==> Initialize instance features in the feature memory")
        features, _ = extract_features(model, cluster_loader, print_freq=50)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        # Create feature memory
        memory = nn.DataParallel(
            Memory(2048, len(dataset.train),
                   temp=args.temp0, momentum=args.momentum)
        ).cuda()
        memory.module.features = F.normalize(features, dim=1).cuda()
    global_cams = torch.tensor([val[3] for val in cluster_loader.dataset])
    cam_count = global_cams.max() + 1
    del cluster_loader

    # if False:
    if not args.resume:
    # ECN-------------热身预训练----------------------
    # Trainer
        vaTrainer = Trainer_USL(model, memory)
        # instance pre-training
        pseudo_labeled_dataset = []
        pseudo_labels = torch.arange(len(dataset.train))  # instance loss for each one
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pseudo_labeled_dataset.append((fname, label.item(), cid))
# init memory like ECN-------------------

  # 再训练则不进行热身预训练
        for epoch in range(args.startE):
            torch.cuda.empty_cache()
            memory.module.labels = pseudo_labels.cuda()
            train_loader = get_vanilla_loader(dataset.images_dir, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances,
                                              iters, trainset=pseudo_labeled_dataset)
            print(f'-----Exemplar Pretraining, Epoch{epoch}...------')
            vaTrainer.train(epoch, train_loader, optimizer,
                            print_freq=args.print_freq, train_iters=args.iters)
        del memory
    # test pre-train
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)


# --------------------正式训练
    print('start training!')
    print('use optics')
    cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=-1)
    trainer = MetaTrainer_qhard(model)
    for epoch in range(start_epoch, args.epochs):
        # Calculate distance
        torch.cuda.empty_cache()
#------------------聚类获得伪标签 利用提取的特征直接聚类得到伪标签

        print('==> Create pseudo labels for unlabeled data')
        # 不指定testset就会默认使用query和gallery拼接的数据集
        cluster_loader = get_test_loader(dataset, args.height, args.width,
                                         args.batch_size, args.workers, testset=sorted(dataset.train))
        # extract_features返回features和labels两个字典
        features, _ = extract_features(model, cluster_loader, print_freq=50)
        # features按照文件名称顺序排序
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        # 特征距离矩阵   使用杰卡德距离
        rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)   # 聚类获得伪标签
        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0) # 获得集群个数
        chosen_cam = split_cam(cam_count, args.mRatio)
# ----------------给离群点重新分配伪标签
#         pseudo_labels = generate_pseudo_labels(pseudo_labels, features)
        nameMap = {val[0]: idx for (idx, val) in enumerate(sorted(dataset.train))}    # fanme和idx的映射
# ---------------获得聚类平均特征

        print('num_cluster:{}'.format(num_cluster))
        # 生成聚类中心的特征(平均)   有顺序的  0  -   num_cluster-1
        print('calculate cluster centroids')
        cluster_features = generate_cluster_features(pseudo_labels, features)
        del cluster_loader, features

# -----------------------生成meta两个数据库
        pseudo_labeled_dataset = defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label.item() != -1:
                if cid in chosen_cam:
                    # meta-train
                    pseudo_labeled_dataset[0].append((fname, label.item(), cid))
                else:
                    # meta-test
                    pseudo_labeled_dataset[1].append((fname, label.item(), cid))
        print('pseudo dataset have two parts:{} and {}'.format(len(pseudo_labeled_dataset[0]),len(pseudo_labeled_dataset[1])))
        # statistics of clusters and un-clustered instances
# --------------------------------------------------生成cluster memory
        if args.grow:
            import math
            # m = args.m*(1-math.cos((2*3.141592653589793*epoch)/(4*args.epochs)))
            m = args.m * epoch/args.epochs
        else:
            m = args.m
        memory = nn.DataParallel(
            ClusterMemory(2048, num_cluster, temp1=args.temp1, temp2=args.temp2,
                               momentum=args.momentum, use_hard=args.usehard,m=m,am=args.am,lamn=args.lamn) #
        ).cuda()

        memory.module.features = F.normalize(cluster_features, dim=1).cuda()

        trainer.memory = memory
# -------------------------------------------
        train_loader = get_metatrain_loader(dataset.images_dir, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset)
        trainer.train(epoch, train_loader, optimizer, nameMap,
                      print_freq=args.print_freq, train_iters=args.iters,
                      step_size=args.step_size, gamma=0.1, lr=args.lr,symmetric=args.symmetric,usecamqhard=args.usecamqhard)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1) or ((epoch+1)>= 40 and (epoch+1)%2==0):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best, fpath=osp.join(logs_dir, 'checkpoint.pth.tar'))
            # # for vis
            mar_feature, _ = extract_features(model, mar_loader, print_freq=args.print_freq)
            mar_feature = torch.stack([mar_feature[f] for f, _, _ in marCamSet.train], 0)
            marPid, marCam = [pid for _, pid, _ in marCamSet.train], \
                             [cam for _, _, cam in marCamSet.train]
            tsneCam = plotTSNE(mar_feature, marPid, marCam, f'{logs_dir}/{epoch}.jpg')
            io.savemat(f'{logs_dir}/{epoch}.mat', {'tsneCam': tsneCam, 'marPid': marPid, 'marCam': marCam})

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch+1, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MetaCam")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--startE', type=int, default=5)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyper-parameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyper-parameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resMeta',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the feature memory")
    parser.add_argument('--mRatio', type=float, default=0.5,
                        help="split ratio of meta-train & meta-test")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    # parser.add_argument('--temp', type=float, default=0.05,
    #                     help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    parser.add_argument('--symmetric', action='store_true', help="for sym ce")
    parser.add_argument('--usecamqhard', action='store_true')
    parser.add_argument('--usehard', action='store_true')
    parser.add_argument('--temp0', type=float, default=0.02, help="temperature for pretrain")
    parser.add_argument('--temp1', type=float, default=0.05, help="temperature for loss1")
    parser.add_argument('--temp2', type=float, default=1.0, help="temperature for loss2")
    parser.add_argument('--am', action='store_true')
    parser.add_argument('--grow', action='store_true')
    parser.add_argument('--m', type=float, default=0.1)
    parser.add_argument('--lamn', type=float, default=0.5)
    main()
