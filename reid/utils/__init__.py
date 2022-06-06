from __future__ import absolute_import
import collections
import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def cal_dist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def split_cam(camCount, mRatio):
    return torch.randperm(camCount)[:int(mRatio * camCount)]


# generate new dataset and calculate cluster centers
def generate_pseudo_labels(cluster_id, inputFeat):
    with_id, witho_id = inputFeat[cluster_id != -1], inputFeat[cluster_id == -1]
    disMat = cal_dist(with_id, witho_id)
    # relabel images
    neighbour = disMat.argmin(0).cpu().numpy()
    newID = cluster_id[cluster_id != -1][neighbour]
    cluster_id[cluster_id == -1] = newID
    return torch.from_numpy(cluster_id).long()
@torch.no_grad()
def generate_cluster_features(labels, features):
    # 输入为labels伪标签（列表 第i个元素的label）  features为特征
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i]].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers