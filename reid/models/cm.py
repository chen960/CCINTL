import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        # for x, y in zip(inputs, targets):
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

# 这里的反向传播不是传播梯度，而是更新集群坐标
class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        # ctx代表聚类代表特征 inputs为输入batch特征
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())      # 计算qci   # 计算batch每个特征和聚类代表特征的相似度情况

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
# # -----------------------找qhrad
#         batch_centers = collections.defaultdict(list)
#         for instance_feature, index in zip(inputs, targets.tolist()):
#             batch_centers[index].append(instance_feature)
#
#         for index, features in batch_centers.items():
#             distances = []
#             for feature in features:
#                 distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
#                 distances.append(distance.cpu().numpy())
#
#             median = np.argmin(np.array(distances))
# # -------------------反向传播更新集群cm坐标
#             ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
#
#             ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module):
    # ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
    #                                momentum=args.momentum, use_hard=args.use_hard).cuda()
    # features为维度   samples为集群个数 temp为tao momentum为反向传播比率
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False,m=0,am=True,temp1=None,temp2=None,lamn=0.5):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.m = m
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.ce = CrossEntropyLabelSmooth(num_classes=num_samples)
        self.am = am
        self.temp1 = temp1
        self.temp2 = temp2
        self.lamn = lamn
        if self.am:
            print('use amsoftmax,m={},temp1={},temp2={}'.format(self.m,self.temp1,self.temp2))
        else:
            print('use softmax, temp1={},temp2={}'.format(self.temp1,self.temp2))
        print('use forward_metatrain and forward_metatest')
    def updateCM(self, inputs, targets,cams,use_camqhard=False):
        # momentum update  更新memorybank   indexes是在总数数据库中的索引号 不是标签
        if not self.use_hard:    # 平均特征更新CM
            for x, y in zip(inputs, targets):
                self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x  #
                self.features[y] /= self.features[y].norm()
        else:
            if use_camqhard:
                # ---------------------找摄像头下的qhard
                batch_centers = collections.defaultdict(list)
                batch_cams = collections.defaultdict(list)
                for instance_feature, index, instance_cam in zip(inputs, targets.tolist(),cams.tolist()):
                    batch_centers[index].append(instance_feature)  # 按照标签归类batch 样本
                    batch_cams[index].append(instance_cam)    # batch 每个样本的摄像头编号

                for index, features in batch_centers.items():
                    distances = []
                    for feature in features:
                        distance = feature.unsqueeze(0).mm(self.features[index].unsqueeze(0).t())[0][
                            0]  # 计算属于 m类的batch与m类集群特征的相似度
                        distances.append(distance.cpu().numpy())
                    cams = set(batch_cams[index])    # P类的这个batch所有摄像头
                    # print(cams)
                    # temp = torch.Tensor(len(cams),2048)

                    # 随机只采用一个P类 第i个摄像头下的特征进行更新
                    # selcam = int(np.random.choice(np.array(cams),size=1,replace=True,p=None))     # 随机选择一个cam
                    # sel = (np.array(batch_cams[index])==selcam)                       # 筛选随机选择cam下的索引
                    # absolute_index = np.arange(0, len(distances))                     # 绝对索引和相对索引对应
                    # for median in list((np.array(absolute_index)[sel])):               # 遍历该cam索引
                    #     self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * features[median]  # 利用该类该cam下的特征更新features
                    #     self.features[index] /= self.features[index].norm()

                    selcam = int(np.random.choice(np.array(cams), size=1, replace=True, p=None))  # 随机选择一个cam
                    sel = (np.array(batch_cams[index]) == selcam)  # 筛选随机选择cam下的索引
                    absolute_index = np.arange(0, len(distances))  # 绝对索引和相对索引对应
                    median = (np.array(absolute_index)[sel])[np.argmin(np.array(distances)[sel])]
                    self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * features[
                        median]  # 利用该类该cam下的特征更新features
                    self.features[index] /= self.features[index].norm()


                    # for i in cams:
                    #     sel = (np.array(batch_cams[index])==i)     # 找到P中所有摄像头id为i的图片  pytorch不能直接等于
                    #     absolute_index = np.arange(0,len(distances))  # 实现相对索引和绝对索引的对应
                    #     # median = np.argmin(np.array(distances)[sel])  # 找到P类摄像头i下相似度最低的为Pi qhard   这个是相对索引 需要找到绝对索引
                    #     median = (np.array(absolute_index)[sel])[np.argmin(np.array(distances)[sel])]     #
                    #     # temp[i] = features[median]
                    # # -------------------反向传播更新集群cm坐标
                    # # self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * temp.mean(dim=0)  # 利用qhard优化ClusterMemory
                    #     self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * features[median]  # 利用qhard优化ClusterMemory
                    #     self.features[index] /= self.features[index].norm()
            else:
            # # -----------------------找qhrad
                batch_centers = collections.defaultdict(list)
                for instance_feature, index in zip(inputs, targets.tolist()):
                    batch_centers[index].append(instance_feature)                          # 按照标签归类batch 样本

                for index, features in batch_centers.items():
                    distances = []
                    for feature in features:
                        distance = feature.unsqueeze(0).mm(self.features[index].unsqueeze(0).t())[0][0]  # 计算属于 m类的batch与m类集群特征的相似度
                        distances.append(distance.cpu().numpy())

                    median = np.argmin(np.array(distances))      # 相似度最低的为qhard
                    # -------------------反向传播更新集群cm坐标
                    self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * features[median]   # 利用qhard优化ClusterMemory
                    self.features[index] /= self.features[index].norm()
    def forward(self, inputs, targets, symmetric=False,metatrain=False):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard:
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
        if metatrain:
        # output为qhard、q与所有簇中心特征c的乘积
            if self.am is not True:
                # outputs /= self.temp1      # temp 是论文中的tao
                # loss = F.cross_entropy(outputs/self.temp1, targets)      # 实现损失函数
                loss = self.ce(outputs/self.temp1, targets)
            else:
                delt_outputs = torch.zeros_like(outputs).scatter_(1, targets.view(-1, 1), self.m)
                loss = self.ce((outputs - delt_outputs) / self.temp1, targets)

            # loss = self.amsoftmax(outputs, targets, self.m, s=self.temp1)
        else:
            # meta_test阶段只使用常规
            loss = self.ce(outputs / self.temp1, targets)

        # print(loss)
        # ------------noise tolerent loss
        if symmetric:
            onehot_targets = torch.zeros_like(outputs)
            onehot_targets.scatter_(1,targets.view(-1,1),1)     # 将标签转化为onehot B * 类别数
            logsoftmax = F.log_softmax(onehot_targets,dim=1)
            softfc = F.softmax(outputs/self.temp2,dim=1)
            loss_noise = - (softfc * logsoftmax).sum(1).mean()
        # print(loss_noise)
            return loss + self.lamn*loss_noise
            # return loss + 0.5*loss_noise
        return loss

    # def amsoftmax(self, outputs, targets, m, s=0.05):
    #     delt_outputs = torch.zeros_like(outputs).scatter_(1, targets.view(-1, 1), m)
    #     # outputs -= delt_outputs
    #     # outputs /= s
    #     # loss = F.cross_entropy(outputs, targets)
    #     loss = self.ce((outputs-delt_outputs)/s, targets)
    #     return loss

    # def asoftmax(self, outputs, targets,m=4):
    #     outputs = outputs.clamp(-1, 1)
    #     def myphi(x, m):
    #         import math
    #         x = x * m
    #         return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) + \
    #                x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)
    #     mask = outputs * 0.0
    #     mask.scatter_(1,targets.view(-1,1),1)
    #     mask = mask.bool()
    #
    #     theta = outputs[mask].acos()
    #     outputs[mask] = myphi(theta, m)  # myphi函数为phi
    #     outputs[mask] = outputs[mask].clamp(-1 * m, 1)
    #
    #     logpt = F.log_softmax(outputs,dim=1)
    #     logpt = logpt.gather(1,targets.view(-1,1))
    #     loss = logpt
    #     loss = loss.mean()
    #     return loss

import torch
import torch.nn as nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, targets, use_label_smoothing=True):
        """
        Args:
            outputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(outputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1).data, 1)
        if use_label_smoothing:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss