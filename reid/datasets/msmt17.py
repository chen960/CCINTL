from __future__ import print_function, absolute_import
import os.path as osp
import tarfile
import os,sys
curPath=os.path.abspath(os.path.dirname(__file__))
rootPath=os.path.split(curPath)[0]
sys.path.append(rootPath)
import glob
import re
import urllib
import zipfile


def _pluck_msmt(list_file=None, dir_path=None, pattern=re.compile(r'([-\d]+)_c([-\d]+)_([-\d]+)')):
    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    ret = []
    pids = []
    for img_path in img_paths:
        pid, camid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  # junk images are just ignored
        assert 1 <= camid <= 15
        # camid -= 1  # index starts from 0
        if pid not in pids:
            pids.append(pid)
        # print(img_path)
        img_path = img_path.split('/',maxsplit=7)[7]         # 这里转化为相对路径
        # print(img_path)
        ret.append((img_path, pid, camid))        # img_path 不能为绝对路径  因为在utils.data.processor那里加了绝对路径

    return ret,pids


class Dataset_MSMT(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'MSMT17_V1')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'MSMT17_V1')
        self.train, train_pids = _pluck_msmt(dir_path=osp.join(exdir,'bounding_box_train' ))
        # self.val, val_pids = _pluck_msmt(dir_path=osp.join(exdir,'bounding_box_train'))
        # self.train = self.train + self.val
        self.query, query_pids = _pluck_msmt(dir_path=osp.join(exdir,'query'))
        self.gallery, gallery_pids = _pluck_msmt(dir_path=osp.join(exdir,'bounding_box_test'))
        self.num_train_pids = len(list(set(train_pids)))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))


class MSMT17(Dataset_MSMT):

    def __init__(self, root, split_id=0, download=True):
        super(MSMT17, self).__init__(root)
        self.load()

