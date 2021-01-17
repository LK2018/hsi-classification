# -*- coding: utf-8 -*-

import os
import sys
from copy import deepcopy
import pdb

import pandas as pd
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

__all__ = ['read_data', 'get_proportional_masks', 'get_fixed_number_masks',
           'get_vector_samples', 'get_patch_samples', 'get_all_patches',
           'load_masks', 'VectorDataset', 'PatchDataset', 'FullImageDataset']


def read_data(data_dir, target_dir):
    """

    Parameters
    ----------
    data_dir: str, e.g. './indian_pines/Indian_pines_corrected.mat'
    target_dir: str, e.g. './indian_pines/Indian_pines_gt.mat'

    Returns
    -------
    data: ndarray, C*H*W
    target: ndarray, H*W

    """

    data = sio.loadmat(data_dir)
    data = list(filter(lambda x: isinstance(x, np.ndarray), data.values()))[0]
    data = data.transpose(2, 0, 1)
    data = normalize(data)

    target = sio.loadmat(target_dir)
    target = list(filter(lambda x: isinstance(x, np.ndarray),
                         target.values()))[0]

    return data, target


def normalize(data):
    """normalize the HSI data make the values between 0 and 1

    Parameters
    ----------
    data: ndarray, C*H*W

    Returns
    -------
    data: ndarray, C*H*W, the normalized data

    """

    data = data.astype(np.float)
    for i in range(len(data)):
        data[i, :, :] -= data[i, :, :].min()
        data[i, :, :] /= data[i, :, :].max()

    return data


def get_proportional_masks(target, train_prop, val_prop, save_dir=None):
    """get masks that be used to extracted training/val/test samples, training
    samples number is determined by the proportion

    Parameters
    ----------
    target: ndarray, H*W, the ground truth of HSI
    train_prop: float, the proportion of training samples, e.g. 0.2
    val_prop: float, the proportion of validation samples, e.g. 0.2
    save_dir: str, masks save path, a folder not a file path,
    e.g. './indian_pines'

    Returns
    -------
    train_mask: ndarray, H*W
    val_mask: ndarray, H*W
    test_mask: ndarray, H*W

    """

    assert train_prop + val_prop < 1
    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        # at least 3 samples for training
        train_num = max(int(round(len(idx) * train_prop)), 3)
        val_num = max(int(round(len(idx) * val_prop)), 3)

        np.random.shuffle(idx)
        train_idx = idx[:train_num]
        val_idx = idx[train_num:train_num + val_num]
        test_idx = idx[train_num + val_num:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'masks_{}_{}'.format(train_prop, val_prop)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'),
                    {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'),
                    {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'),
                    {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def get_each_class_num(target, total_num, min_num=3):

    class_num = target.max().astype(int)
    assert total_num >= class_num * min_num

    target_count = target[target != 0]
    num = float(len(target_count))
    num_i_ = pd.value_counts(target_count).sort_index().values
    num_i = np.floor(num_i_ / num * total_num).astype(np.int)
    num_i[num_i < min_num] = min_num

    if num_i.sum() > total_num:
        i = 0
        max_idx = (-num_i).argsort()
        while num_i.sum() != total_num:
            num_i[max_idx[i]] = max(min_num, num_i[max_idx[i]] - 1)
            i = i + 1 if (i + 1) < len(num_i) else 0

    if num_i.sum() < total_num:
        i = 0
        max_idx = num_i.argsort()
        while num_i.sum() != total_num:
            num_i[max_idx[i]] = min(num_i_[max_idx[i]], num_i[max_idx[i]] + 1)
            i = i + 1 if (i + 1) < len(num_i) else 0

    return num_i


def get_fixed_number_masks(target, train_num, val_num, save_dir=None):
    """get masks that be used to extracted training/val/test samples,
    training samples number is determined by a fixed number

    Parameters
    ----------
    target: ndarray, H*W, the ground truth of HSI
    train_num: int, the number of training samples, at least 50
    val_num: float, the proportion of validation samples, e.g. 0.2
    save_dir: str, masks save path, a folder not a file path,
    e.g. './indian_pines'

    Returns
    -------
    train_mask: ndarray, H*W
    val_mask: ndarray, H*W
    test_mask: ndarray, H*W

    """

    train = get_each_class_num(target, train_num)
    val = get_each_class_num(target, val_num)

    train_mask = np.zeros((target.shape[0], target.shape[1]))
    val_mask = train_mask.copy()
    test_mask = train_mask.copy()

    for i in range(1, target.max() + 1):
        idx = np.argwhere(target == i)
        train_i = train[i - 1]
        val_i = val[i - 1]

        np.random.shuffle(idx)
        train_idx = idx[:train_i]
        val_idx = idx[train_i:train_i + val_i]
        test_idx = idx[train_i + val_i:]

        train_mask[train_idx[:, 0], train_idx[:, 1]] = 1
        val_mask[val_idx[:, 0], val_idx[:, 1]] = 1
        test_mask[test_idx[:, 0], test_idx[:, 1]] = 1

    if save_dir:
        folder_name = 'masks_{}_{}'.format(train_num, val_num)
        save_dir = os.path.join(save_dir, folder_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        sio.savemat(os.path.join(save_dir, 'train_mask.mat'),
                    {'train_mask': train_mask})
        sio.savemat(os.path.join(save_dir, 'val_mask.mat'),
                    {'val_mask': val_mask})
        sio.savemat(os.path.join(save_dir, 'test_mask.mat'),
                    {'test_mask': test_mask})

    return train_mask, val_mask, test_mask


def load_masks(root_dir, target, train_prop, val_prop):
    """load training/val/test masks from saved masks file, this function will
    produce new masks if input masks path is not exist or load masks failed

    Parameters
    ----------
    root_dir: str, the root of masks be saved, e.g. './indian_pines'
    target: ndarray, H*W, ground truth
    train_prop: training proportion or training number
    val_prop: validation proportion

    Returns
    -------
    train_mask: ndarray, H*W
    val_mask: ndarray, H*W
    test_mask: ndarray, H*W

    """

    train_prop = int(train_prop) if train_prop >= 1 else train_prop
    val_prop = int(val_prop) if val_prop >= 1 else val_prop

    masks_dir = os.path.join(root_dir, 'masks_{}_{}'.format(train_prop,
                                                            val_prop))
    try:
        train_mask = sio.loadmat(os.path.join(masks_dir, 'train_mask.mat'))\
            ['train_mask']
        val_mask = sio.loadmat(os.path.join(masks_dir, 'val_mask.mat'))\
            ['val_mask']
        test_mask = sio.loadmat(os.path.join(masks_dir, 'test_mask.mat'))\
            ['test_mask']
    except IOError:
        import platform
        if platform.python_version()[0] == '2':
            input_func = eval('raw_input')
        else:
            input_func = eval('input')
        flag = input_func('Prepare dataset, masks file not found! If produce a '
                          'new group of masks? [y/n] >> ')
        while True:
            if flag == 'y':
                if train_prop < 1:
                    train_mask, val_mask, test_mask = \
                        get_proportional_masks(target, train_prop, val_prop,
                                               save_dir=root_dir)
                else:
                    train_mask, val_mask, test_mask = \
                        get_fixed_number_masks(target, train_prop, val_prop,
                                               save_dir=root_dir)
                break
            elif flag == 'n':
                print('Program has terminated.')
                sys.exit()
            else:
                flag = input_func('Unknown character! please enter again >> ')

    return train_mask, val_mask, test_mask


# ########################### vector sample, VectorDataset ##################
def get_vector_samples(data, target, mask):
    """get vector samples for the classifer like bp etc.

    Parameters
    ----------
    data: ndarray, C*H*W
    target: ndarray, H*W
    mask: ndarray, H*W, be used to get sample pairs from data and target

    Returns
    -------
    data: ndarray, N*C, N is samples number
    target: ndarray, N*1, 1D array

    """

    data = data*mask
    target = target*mask

    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T
    target = target.ravel()
    data = data[target != 0]
    target = target[target != 0] - 1

    return data, target


# ########################### patch sample, PatchDataset ####################
def get_patch_samples(data, target, mask, patch_size=13, shuffle=True):
    """get patch samples for the classifier like cnn etc.

    Parameters
    ----------
    data: ndarray, C*H*W
    target: ndarray, H*W
    mask: ndarray, H*W
    patch_size: int, default 13
    shuffle: bool, if shuffle the samples, default True

    Returns
    -------
    patch_data: ndarray, N*C*P*P, N is number of samples, P is patch size
    patch_target: ndarray, N*1

    """

    # padding data
    width = patch_size // 2
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    target = np.pad(target, ((width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')

    # get patches
    patch_target = target * mask
    patch_target = patch_target[patch_target != 0] - 1
    patch_data = np.zeros((patch_target.shape[0], data.shape[0], patch_size,
                           patch_size))
    index = np.argwhere(mask == 1)
    for i, loc in enumerate(index):
        patch = data[:, loc[0] - width:loc[0] + width + 1,
                loc[1] - width:loc[1] + width + 1]
        patch_data[i, :, :, :] = patch

    # shuffle
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(patch_data)
        np.random.set_state(state)
        np.random.shuffle(patch_target)

    return patch_data, patch_target


def get_all_patches(data, patch_size):
    """get patches of all data points in the HSI data

    Parameters
    ----------
    data: ndarray, C*H*W
    patch_size: int, e.g. 13

    Returns
    -------
    patch_data: ndarray, N*C*P*P, N=H*W, P is patch size

    """
    width = patch_size // 2
    mask = np.ones((data.shape[1], data.shape[2]))

    patch_data = np.zeros((data.shape[1] * data.shape[2], data.shape[0],
                           patch_size, patch_size))
    data = np.pad(data, ((0, 0), (width, width), (width, width)), 'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')
    index = np.argwhere(mask)
    for i, loc in enumerate(index):
        patch_data[i, :, :, :] = data[:, loc[0] - width:loc[0] + width + 1,
                                 loc[1] - width:loc[1] + width + 1]

    return patch_data


class VectorDataset(Dataset):
    """ dataset for bp net
    """

    def __init__(self, data_dir, target_dir, train_prop, val_prop,
                 state='train', save=True):

        self.raw_data, self.raw_target = read_data(data_dir, target_dir)
        self.train_prop, self.val_prop = train_prop, val_prop
        self.root_dir = os.path.dirname(data_dir)
        self.train_mask, self.val_mask, self.test_mask = \
            load_masks(self.root_dir, self.raw_target, train_prop, val_prop)

        self.state = state
        self.save = save

        self.train, self.val, self.test = self._get_vector_data()
        self.data, self.target = self._get_current_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def _get_current_data(self):

        if self.state == 'train':
            data, target = self.train
        elif self.state == 'val':
            data, target = self.val
        elif self.state == 'test':
            data, target = self.test
        else:
            print("VectorDataset class -> _get_current_data(), 'state' error!")
            sys.exit()

        return data, target

    def _get_vector_data(self):

        pt_dir = os.path.join(self.root_dir, 'vectors_{}_{}'.format(
            self.train_prop, self.val_prop))

        try:
            train = torch.load(os.path.join(pt_dir, 'train.pt'))
            val = torch.load(os.path.join(pt_dir, 'val.pt'))
            test = torch.load(os.path.join(pt_dir, 'test.pt'))

        except IOError:
            print("Not find '.pt' file or read file failed!")
            print("Begin to produce training data, val data and test data ...")
            train = get_vector_samples(self.raw_data, self.raw_target,
                                       self.train_mask)
            val = get_vector_samples(self.raw_data, self.raw_target,
                                       self.val_mask)
            test = get_vector_samples(self.raw_data, self.raw_target,
                                       self.test_mask)
            train = self._to_tensor(train)
            val = self._to_tensor(val)
            test = self._to_tensor(test)

            if self.save:
                if not os.path.exists(pt_dir):
                    os.makedirs(pt_dir)
                with open(os.path.join(pt_dir, 'train.pt'), 'wb') as f:
                    torch.save(train, f)
                with open(os.path.join(pt_dir, 'val.pt'), 'wb') as f:
                    torch.save(val, f)
                with open(os.path.join(pt_dir, 'test.pt'), 'wb') as f:
                    torch.save(test, f)

        return train, val, test

    @staticmethod
    def _to_tensor(sample):
        data, target = sample

        # data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).long()

        # Convert data and target to cuda format in the dataset rather than
        # training part can significantly improve training speed.
        data = torch.from_numpy(data).float().cuda()
        target = torch.from_numpy(target).long().cuda()
        # data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).long()
        return data, target

    def set_state(self, state):
        # state: str, one of 'train', 'val', 'test'
        self.state = state
        self.data, self.target = self._get_current_data()


class PatchDataset(Dataset):
    """dataset for the network based on cnn

    Parameters
    ----------
    data_dir: str, path of HSI data,
    e.g. './indian_pines/Indian_pines_corrected.mat'
    target_dir: str, path of HSI ground truth,
    e.g. './indian_pines/Indian_pines_gt.mat'
    train_prop: float or int, training proportion or training number,
    e.g. 0.2, 300
    val_prop: float, validation proportion, e.g. 0.2
    patch_size: int, the width of a patch that input network as a sample,
    e.g. 13
    state: str, the state of dataset, such as 'train', 'val', 'test',
    default is 'train'
    save: bool, if save training data, validation data and test data,
    format is '.pt', default is True

    Attributes
    ----------
    raw_data: ndarray, C*H*W, raw HSI data
    raw_target: ndarray, H*W, raw HSI ground truth
    root_dir: the root path of HSI data, is a folder
    train_mask/val_mask/test_mask: ndarray, H*W, training/validation/test
    sample mask
    train/val/test: tuple, (e1, e2), e1: data(N*C*P*P), e2: target(N,),
    N is the samples number, P is the patch size
    data/target: tensor, N*C*P*P/N*1, current data and target,
    one of train/val/test, has been converted to tensor

    Methods
    --------
    _get_current_data(): return current data and target according to
    attribute 'state'
    _get_patch_data(): return train, val and test sample pairs
    set_state(state): 'state'：one of 'train', 'val' and 'test'

    """

    def __init__(self, data_dir, target_dir, train_prop, val_prop,
                 patch_size, state='train', save=True):

        self.raw_data, self.raw_target = read_data(data_dir, target_dir)
        self.train_prop, self.val_prop = train_prop, val_prop
        self.root_dir = os.path.dirname(data_dir)
        self.train_mask, self.val_mask, self.test_mask = \
            load_masks(self.root_dir, self.raw_target, train_prop, val_prop)

        self.patch_size = patch_size
        self.state = state
        self.save = save

        self.train, self.val, self.test = self._get_patch_data()
        self.data, self.target = self._get_current_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def _get_current_data(self):

        if self.state == 'train':
            data, target = self.train
        elif self.state == 'val':
            data, target = self.val
        elif self.state == 'test':
            data, target = self.test
        else:
            print("PatchDataset class -> _get_current_data(), 'state' error!")
            sys.exit()

        return data, target

    def _get_patch_data(self):

        pt_dir = os.path.join(self.root_dir, 'patches_{}_{}_{}'.format(
            self.train_prop, self.val_prop, self.patch_size
        ))

        try:
            train = torch.load(os.path.join(pt_dir, 'train.pt'))
            val = torch.load(os.path.join(pt_dir, 'val.pt'))
            test = torch.load(os.path.join(pt_dir, 'test.pt'))

        except IOError:
            print("Not find '.pt' file or read file failed!")
            print("Begin to produce training data, val data and test data ...")
            train= get_patch_samples(
                self.raw_data, self.raw_target, self.train_mask, self.patch_size
            )
            val = get_patch_samples(
                self.raw_data, self.raw_target, self.val_mask, self.patch_size
            )
            test = get_patch_samples(
                self.raw_data, self.raw_target, self.test_mask, self.patch_size
            )
            train = self._to_tensor(train)
            val = self._to_tensor(val)
            test = self._to_tensor(test)

            if self.save:
                if not os.path.exists(pt_dir):
                    os.makedirs(pt_dir)
                with open(os.path.join(pt_dir, 'train.pt'), 'wb') as f:
                    torch.save(train, f)
                with open(os.path.join(pt_dir, 'val.pt'), 'wb') as f:
                    torch.save(val, f)
                with open(os.path.join(pt_dir, 'test.pt'), 'wb') as f:
                    torch.save(test, f)

        return train, val, test

    @staticmethod
    def _to_tensor(sample):
        data, target = sample

        # data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).long()

        # Convert data and target to cuda format in the dataset rather than
        # training part can significantly improve training speed.
        data = torch.from_numpy(data).float().cuda()
        target = torch.from_numpy(target).long().cuda()
        # data = torch.from_numpy(data).float()
        # target = torch.from_numpy(target).long()
        return data, target

    def set_state(self, state):
        # state: str, one of 'train', 'val', 'test'
        self.state = state
        self.data, self.target = self._get_current_data()


class FullImageDataset(Dataset):
    """dataset for networks like fcnn

    Parameters
    ----------
    data_dir: str, path of HSI data,
    e.g. './indian_pines/Indian_pines_corrected.mat'
    target_dir: str, path of HSI ground truth,
    e.g. './indian_pines/Indian_pines_gt.mat'
    train_prop: float or int, training proportion or training number,
    e.g. 0.2, 300
    val_prop: float, validation proportion, e.g. 0.2
    split_size: int, split the data in case the data is too big to train in GPU,
     for example, Indian pines data
    have size 200*145*145, if set 'split_size' to 100, then will split the image
     to at least 4 small overlapped
    images and the final data have the size 4*200*100*100
    overlap: int, the number of overlapped rows/columns of two adjacent small
     images, e.g. 10

    Attributes
    ----------
    raw_data: ndarray, C*H*W, raw HSI data
    raw_target: ndarray, H*W, raw HSI ground truth
    train_mask/val_mask/test_mask: ndarray, H*W, training/validation/test
     samples mask
    overlap_matrix: ndarry, have the same shape as raw_target, the value of
     location (i, j) denotes the overlapped numbers of data/target in location
     (i, j)
    data/target: tensor, N*C*H*W/H*W
    mask: tensor, N*H*W, be used to keep the samples to compute loss when
     training

    Methods
    -------
    split_data(to_tensor=False): split big image data into many small images,
     'to_tensor': bool,mif convert the splitted data to tensor, default is False
    stitch_data: stitch the small images, get the big complete image, 's_data':
     ndarray, N*C*S*S, S is the split size, return 'data' have the shape C*H*W

    """

    def __init__(self, data_dir, target_dir, train_prop, val_prop,
                 split_size, overlap):

        self.raw_data, self.raw_target = read_data(data_dir, target_dir)
        mask_dir = os.path.dirname(data_dir)
        self.train_mask, self.val_mask, self.test_mask = \
            load_masks(mask_dir, self.raw_target, train_prop, val_prop)

        self.split_size = split_size
        self.overlap = overlap
        self.overlap_matrix = np.ones(self.raw_target.shape, dtype=int)
        self.data, self.target, self.mask = self.split_data(to_tensor=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx], self.mask[idx], idx

    @staticmethod
    def _get_idx(hs, ws):
        idx = []
        for i in range(hs.start, hs.stop):
            for j in range(ws.start, ws.stop):
                idx.append((i, j))
        return idx

    def split_data(self, to_tensor=False):

        H, W = self.raw_target.shape
        S = self.split_size
        l = self.overlap

        data = []
        target = []
        split_train_mask = []
        train_mask_ = self.train_mask
        target_ = (self.raw_target - 1) * train_mask_

        si = 0
        si_ = -S
        while si != si_:
            sj = 0
            sj_ = -S
            while sj != sj_:
                hs = slice(si, si + S)
                ws = slice(sj, sj + S)
                hs_ = slice(si, si_ + S)
                ws_ = slice(sj, sj_ + S)

                data.append(self.raw_data[:, hs, ws])
                target.append(target_[hs, ws])
                split_train_mask.append(train_mask_[hs, ws])

                i1 = self._get_idx(hs, ws_)
                i2 = self._get_idx(hs_, ws)
                ii = i1 + i2
                ii = np.array(list(set(ii)))
                if len(ii) > 0:
                    self.overlap_matrix[ii[:, 0], ii[:, 1]] += 1

                sj_ = sj
                sj += S - l
                sj = min((W - S), sj)
            si_ = si
            si += S - l
            si = min((H - S), si)

        data = np.array(data)
        target = np.array(target)
        split_train_mask = np.expand_dims(np.array(split_train_mask), axis=1)

        if to_tensor:
            data = torch.from_numpy(data).float()
            target = torch.from_numpy(target).long()
            split_train_mask = torch.from_numpy(split_train_mask).float()

        return data, target, split_train_mask

    def stitch_data(self, s_data):

        # s_data: ndarray, N×C×S×S

        assert len(s_data.shape) == 4
        H, W = self.raw_target.shape
        S = self.split_size
        l = self.overlap
        assert S == s_data.shape[2] == s_data.shape[3]
        data = np.zeros((s_data.shape[1], H, W))

        k = 0
        si = 0
        si_ = -S
        while si != si_:
            sj = 0
            sj_ = -S
            while sj != sj_:
                hs = slice(si, si + S)
                ws = slice(sj, sj + S)

                data_i = np.zeros((s_data.shape[1], H, W))
                data_i[:, hs, ws] = s_data[k]
                data += data_i

                sj_ = sj
                sj += S - l
                sj = min((W - S), sj)
                k += 1
            si_ = si
            si += S - l
            si = min((H - S), si)

        data = data.astype(np.float64)
        data /= self.overlap_matrix

        return data


if __name__ == '__main__':

    data_dir = '../data/indian_pines/Indian_pines_corrected.mat'
    target_dir = '../data/indian_pines/Indian_pines_gt.mat'
    train_prop = 100
    val_prop = 50
#     patch_size = 7
#
#     # dataset = PatchDataset(data_dir, target_dir, train_prop, val_prop,
#     patch_size, state='train')
#     # pdb.set_trace()
    _, target = read_data(data_dir, target_dir)
#     load_masks('./', target, train_prop=0.3, val_prop=0.2)

    tr, va, te = get_fixed_number_masks(target, train_prop, val_prop,
                                    save_dir=None)


