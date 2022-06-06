from __future__ import print_function, absolute_import
import time
import numpy as np

import torch
import torch.nn as nn

from .utils.meters import AverageMeter
from .models import *

# 使用MetaLearn  + clustercontrast中的qhard
class MetaTrainer_qhard(object):
    def __init__(self, encoder, memory=None):
        super(MetaTrainer_qhard, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, epoch, data_loaders, optimizer, name_map, print_freq=10,
              train_iters=200, step_size=20, gamma=0.1, lr=0.01,symmetric=False,usecamqhard=False):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        factor = gamma ** (epoch // step_size)
        meta_lr = lr * factor

        loader_count = len(data_loaders)  # number of cameras in cur dataset

        end = time.time()

        for i in range(train_iters):
            # load data
            meta_train_id = np.random.choice(loader_count)
            inputs = data_loaders[meta_train_id].next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, targets, _, cams, names = self._parse_data(inputs)
            # indexes = torch.tensor([name_map[name] for name in names]).cuda()
            # forward
            f_out = self.encoder(inputs)
            # compute loss with the hybrid memory
            loss_onestep = self.memory(f_out, targets,symmetric=symmetric,metatrain=True).mean()
# --------------------------与cluster cnontrast不同部分
            self.encoder.zero_grad()
            # first-order grad  一阶导保存，创建一个新模型使用优化后的参数--------------
            grad_info = torch.autograd.grad(
                loss_onestep, self.encoder.module.params(), create_graph=True
            )
            new_meta = create('resMeta', num_classes=0)
            new_meta.copy_model(self.encoder.module)  # generate a copy
            new_meta.update_params(
                lr_inner=meta_lr, source_params=grad_info, solver='sgd'
            )
            # According to Eq. 7, the temporary model (new_meta) is now related to the original model (self.encoder)
            # through 'grad_info'(grad_info is differentiable with 'create_graph=True'), and we can obtain high-order
            # gradients to optimize self.encoder
            del grad_info
            new_meta = nn.DataParallel(new_meta).to(self.device)
# ----------------------metatest上计算loss
            # another camera domain
            meta_test_id = 0 if meta_train_id == 1 else 1
            meta_inputs = data_loaders[meta_test_id].next()
            meta_inputs, meta_targets, _, meta_cams, metaNames = self._parse_data(meta_inputs)
            # meta_indexes = torch.tensor([name_map[name] for name in metaNames]).cuda()
            f_meta_out = new_meta(meta_inputs)
            loss_meta = self.memory(f_meta_out, meta_targets,symmetric=symmetric,metatrain=False).mean()
            loss_meta_onestep = self.memory(f_out, targets, symmetric=symmetric,metatrain=True).mean()
            loss_final = loss_meta_onestep + loss_meta

            optimizer.zero_grad()
            # to check: grad = torch.autograd.grad(loss_meta, self.encoder.module.params())
            # and grad should not be 'None'
            loss_final.backward()              # 优化网络
            optimizer.step()

            # update memory   更新分为θ‘二阶导和θ一阶导两部分
            with torch.no_grad():
                if isinstance(self.memory, nn.DataParallel):
                    self.memory.module.updateCM(self.encoder(meta_inputs), meta_targets, meta_cams,usecamqhard)    # 这里更新还是采用正式优化后的模型提取的特征优化
                    self.memory.module.updateCM(f_out, targets, cams,usecamqhard)   # 这里考虑更新两次会不会不太好
                else:
                    self.memory.updateCM(self.encoder(meta_inputs), meta_targets, meta_cams,usecamqhard)
                    self.memory.updateCM(f_out, targets, cams,usecamqhard)

            losses.update(loss_meta.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'LossMeta {:.3f}\t'
                      'LossOneStep {:.3f}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.avg, loss_meta_onestep.item()))

    def _parse_data(self, inputs):
        imgs, names, pids, cams, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cams.cuda(), names

    def _forward(self, inputs):
        return self.encoder(inputs)

class Trainer_USL_qhard(object):
    def __init__(self, encoder, memory=None):
        super(Trainer_USL_qhard, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer,
              print_freq=10, train_iters=200, symmetric=False,usecamqhard=False):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, targets, _, cams = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # compute loss with the hybrid memory
            loss = self.memory(f_out, targets, symmetric,metatrain=True).mean()  # + loss_noisy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update memory
            with torch.no_grad():
                try:
                    self.memory.module.updateCM(f_out, targets,cams,usecamqhard)
                except:
                    self.memory.updateCM(f_out, targets,usecamqhard)

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, names, pids, cams, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cams.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)