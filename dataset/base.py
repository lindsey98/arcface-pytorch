
from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image
from shutil import copyfile
import time
from distutils.dir_util import copy_tree


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform = None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1 : im = im.convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.ys[index]

        return im, target

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]


class BaseDatasetMod(torch.utils.data.Dataset):
    def __init__(self, root, source, classes, transform = None):
        self.classes = classes
        self.root = root
        self.transform = transform
        self.ys, self.im_paths, self.I = [], [], []

        if not os.path.exists(root):
            print('copying file from source to root')
            print('from:', source)
            print('to:', root)
            c_time = time.time()

            copy_tree(source, root)

            elapsed = time.time() - c_time
            print('done copying file: %.2fs', elapsed)

    def nb_classes(self):
        #print(self.classes)
        #print(len(set(self.ys)), len(set(self.classes)))
        #print(type(self.ys))
        #print(len(set(self.ys) & set(self.classes)))
        assert set(self.ys) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        # convert gray to rgb
        if len(list(im.split())) == 1 : im = im.convert('RGB')
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index

    def get_label(self, index):
        return self.ys[index]

    def set_subset(self, I):
        self.ys = [self.ys[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
