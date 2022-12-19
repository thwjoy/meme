import errno
import os
import io
import json
import PIL
import pickle
import copy

from functools import reduce
import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, ImageFolder, MNIST, SVHN
import torchvision.transforms as transforms

from collections import Counter, OrderedDict
from collections import defaultdict

def join_datasets(data_1, labels_1, data_2, labels_2):
    data_arr_1 = []
    data_arr_2 = []
    label_arr = []
    for i in range(10):
        inds_1 = np.array(np.where(labels_1 == i)[0])
        inds_2 = np.array(np.where(labels_2 == i)[0])
        num = min(len(inds_1), len(inds_2))
        data_arr_1.append(data_1[inds_1[:num]] / 255.0)
        data_arr_2.append(data_2[inds_2[:num]] / 255.0)
        label_arr.append(torch.ones(num) * i)

    return torch.cat(data_arr_1, dim=0), torch.cat(data_arr_2, dim=0), torch.cat(label_arr, dim=0)

class MNIST_SVHN(torch.utils.data.Dataset):
    # static class variables for caching training data
    train_data_sup, train_labels_sup = None, None
    train_data_unsup, train_labels_unsup = None, None
    train_data, train_labels = None, None
    fixed_imgs = None

    def __init__(self, root, mode, download=True, sup_frac=1.0, **kwargs):
        super().__init__()
        #transform_mnist = transforms.Compose([transforms.Pad(2, 2)])
        mnist = MNIST(root, train=True if mode is not 'test' else False, download=download, transform=None)
        svhn = SVHN(root, split='train' if mode is not 'test' else 'test', download=download)
       
        self.mnist_data, self.mnist_labels = mnist.data.view(-1, 28**2).float(), mnist.targets

        self.svhn_data, self.svhn_labels = torch.tensor(svhn.data).float(), svhn.labels
        # we now need to shuffle the datasets so that they match

        # split the samples
        assert mode in ["sup", "unsup", "test"], "invalid train/test option values"

        if mode in ["sup", "unsup"]:

            if MNIST_SVHN.train_data is None:
                print("Splitting dataset")
                mnist, svhn, MNIST_SVHN.train_labels = join_datasets(self.mnist_data,
                                                                               self.mnist_labels,
                                                                               self.svhn_data,
                                                                               self.svhn_labels)

                shuf = np.linspace(0, MNIST_SVHN.train_labels.shape[0]-1, MNIST_SVHN.train_labels.shape[0])
                np.random.shuffle(shuf)
                MNIST_SVHN.train_data = (mnist[shuf], svhn[shuf])
                MNIST_SVHN.train_labels = MNIST_SVHN.train_labels[shuf]


                num_sup_samples = int(sup_frac * MNIST_SVHN.train_labels.shape[0])
                MNIST_SVHN.train_data_sup = (MNIST_SVHN.train_data[0][:num_sup_samples], MNIST_SVHN.train_data[1][:num_sup_samples])
                MNIST_SVHN.train_labels_sup = MNIST_SVHN.train_labels[:num_sup_samples]
                MNIST_SVHN.train_data_unsup = (MNIST_SVHN.train_data[0][num_sup_samples:], MNIST_SVHN.train_data[1][num_sup_samples:])
                MNIST_SVHN.train_labels_unsup = MNIST_SVHN.train_labels[num_sup_samples:]
                

                # shuffle the unsups
                if sup_frac != 1.0:
                    shuf = np.linspace(0, MNIST_SVHN.train_labels_unsup.shape[0]-1, MNIST_SVHN.train_labels_unsup.shape[0])
                    np.random.shuffle(shuf)
                    MNIST_SVHN.train_data_unsup = (MNIST_SVHN.train_data_unsup[0][shuf], MNIST_SVHN.train_data_unsup[1])
                    MNIST_SVHN.train_labels_unsup = MNIST_SVHN.train_labels_unsup[shuf]


            if mode == "sup":
                self.data, self.labels = MNIST_SVHN.train_data_sup, MNIST_SVHN.train_labels_sup
                print("Num sup samples %i" % self.labels.shape[0])
            else:
                self.data, self.labels = MNIST_SVHN.train_data_unsup, MNIST_SVHN.train_labels_unsup
                print("Num unsup samples %i" % self.labels.shape[0])
        else:
            self.mnist, self.svhn, self.labels = join_datasets(self.mnist_data,
                                                   self.mnist_labels,
                                                   self.svhn_data,
                                                   self.svhn_labels)
            shuf = np.linspace(0, len(self.labels)-1, len(self.labels)).astype(np.int)
            np.random.shuffle(shuf)
            self.data = (self.mnist[shuf], self.svhn[shuf])
            self.labels = self.labels[shuf]
            MNIST_SVHN.fixed_imgs = self.data[0][:64], self.data[1][:64]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.data[1][index], self.data[0][index], self.labels[index]

def setup_data_loaders(batch_size, sup_frac=1.0, root=None, **kwargs):

    if root is None:
        root = get_data_directory(__file__)
    if 'num_workers' not in kwargs:
        kwargs = {'num_workers': 4, 'pin_memory': True}
    cached_data = {}
    loaders = {}

    if sup_frac == 0.0:
        modes = ["test", "unsup"]
    elif sup_frac == 1.0:
        modes = ["test", "sup"]
    else:
        modes = ["test", "unsup", "sup"]

    for mode in modes:
        cached_data[mode] = MNIST_SVHN(root=root, mode=mode, download=True, sup_frac=sup_frac)
        loaders[mode] = DataLoader(cached_data[mode], batch_size=batch_size, shuffle=True, drop_last=False, **kwargs)
    return loaders 

