# -*- coding: utf-8 -*-

import os
import sys
import time
import pdb

import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .display import plot_accuracy_curves, display_map_and_save
from .utils import check_path, str_dict_for_show, recur_str_dict_for_show, \
    flatten_a_nested_dict, combine_dicts

__all__ = ['compute_accuracy', 'compute_accuracy_from_mask',
           'compute_accuracy_each_class', 'get_confusion_matrix',
           'compute_kappa_coef', 'compute_accuracy_std', 'records_to_csv',
           'ResultOutput']


# ############################# computer accuracy by vector data #############
def compute_accuracy(pred, target):
    """compute accuracy using vector predicted data

    Parameters
    ----------
    pred: ndarray, shape: (N,)
    target: ndarray, shape: (N,)

    Returns
    -------
    accuracy: float

    """

    accuracy = float((pred == target).sum()) / float(len(target))
    return accuracy


# ############################# compute accuracy by 2D data and mask #########
def compute_accuracy_from_mask(pred, target, mask):
    """compute accuracy using 2D predicted data and a 2D mask

    Parameters
    ----------
    pred: ndarray, H*W
    target: ndarray, H*W
    mask: one of train, validation and test masks, ndarray, H*W

    Returns
    -------
    accuracy: float

    """

    pred = pred.copy()
    pred[target == 0] = 0
    target = target.copy()
    pred = pred * mask
    target = target * mask

    pred = pred[target != 0]
    target = target[target != 0]
    accuracy = float((pred == target).sum()) / float(len(pred))

    return accuracy


def compute_accuracy_each_class(pred, target, mask):
    """compute accuracy of each class

    Parameters
    ----------
    pred: ndarray, H*W
    target: ndarray, H*W
    mask: ndarray, H*W

    Returns
    -------
    accuracy: float

    """

    pred = pred.copy()
    pred[target == 0] = 0
    target = target.copy()
    pred = pred * mask
    target = target * mask

    accuracy = np.zeros(target.max().astype(np.int), dtype='float64')
    for i in range(target.max().astype(np.int)):
        if len(target[target == i + 1]) == 0:
            accuracy[i] = 0
        else:
            accuracy[i] = float((pred[target == i + 1] ==
                                 target[target == i + 1]).sum()) / \
                          float(len(target[target == i + 1]))

    return accuracy


def get_confusion_matrix(pred, target):
    """compute confusion matrix

    Parameters
    ----------
    pred: 1D or 2D ndarray
    target: 1D or 2D ndarray

    Returns
    -------
    conf_m: 2D ndarray, confusion matrix

    """

    if len(pred.shape) == 1:
        conf_m = confusion_matrix(target, pred,
                                  labels=range(1, int(target.max()) + 1))
    elif len(pred.shape) == 2:
        conf_m = confusion_matrix(target.ravel(), pred.ravel(),
                                  labels=range(1, int(target.max()) + 1))
    else:
        print("Get confusion matrix error! Dimension of 'pred' and 'target' "
              "must be 1 or 2.")
        sys.exit()

    return conf_m


def compute_kappa_coef(pred, target):
    """compute kappa coefficient

    Parameters
    ----------
    pred: 1D or 2D ndarray
    target: 1D or 2D ndarray

    Returns
    -------
    kappa: float, kappa coefficient (κ)

    """

    conf_m = get_confusion_matrix(pred, target)
    total = np.sum(conf_m)
    pa = np.trace(conf_m) / float(total)
    pe = np.sum(np.sum(conf_m, axis=0) * np.sum(conf_m, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)

    return kappa


def records_to_csv(record_id, hyper_params, train_records, accuracy, save_dir):
    """save one record that contains training/data/... parameters and accuracies
    when finishing a training

    Parameters
    ----------
    record_id: str, mark the current training
    hyper_params: dict, the params need to be recorded,
    e.g. {'method': 'resnet','train_prop': 0.05, 'lr': 0.0001, ...}
    train_records: dict, records during each training,
    e.g. {train_time: 100s, predict_time: 1s, ...}
    accuracy: dict, accuracies dict,
    e.g. {'test_oa': 0.91, 'test_aa': 0.92, 'test_kappa': 0.90, ...}
    save_dir: str, the output csv file path, e.g. './accuracy.csv'

    Returns
    -------
    df: pd.DataFrame

    """

    hyper_params = flatten_a_nested_dict(hyper_params)
    train_records = flatten_a_nested_dict(train_records)

    records = {'id': record_id}
    records = combine_dicts(records, hyper_params, train_records, accuracy)

    try:
        df = pd.read_csv(save_dir, engine='python')
        bkp_dir = '%s_bkp.csv' % os.path.splitext(save_dir)[0]
        df.to_csv(bkp_dir, index=False)  # backup
        idx = len(df)
        for key, value in records.items():
            df.loc[idx, key] = str(value)
    except IOError:
        df = pd.DataFrame()
        for key, value in records.items():
            df.loc[0, key] = str(value)
    df.to_csv(save_dir, index=False)

    return df


def compute_accuracy_std(acc_df, save_dir, acc_fields, fixed_fields=None):
    """compute standard deviation of accuracies of many times training

    Parameters
    ----------
    acc_df: pd.DataFrame, accuracy data, like:
    ===========================================================================
    |      id     |    method   |    test_oa     |    test_aa   |   test_kappa
    +-------------+-------------+----------------+--------------+--------------
    |      1      |      m1     |       0.92     |      0.90    |      0.91
    +-------------+-------------+----------------+--------------+--------------
    |      2      |      m1     |       0.91     |      0.93    |      0.92
    +-------------+-------------+----------------+--------------+--------------
    |    ...
    +-------------------
    acc_df must contain the field 'method', the accuracy value should be less
    than 1
    save_dir: str, the output will be saved as a '.csv' file,
    e.g. './accuracy_std.csv'
    acc_fields: list, specify the accuracy fields to calculate the standard
    deviation, e.g. ['test_oa', 'test_aa', 'test_kappa', ...]
    fixed_fields: dict, fix the values of some fields and filter the acc_df
    data, e.g. {'train_prop': 0.2, ...}, default is None

    Returns
    -------
    std_df: pd.DataFrame, output df data

    """

    if fixed_fields is not None:
        for key, value in fixed_fields.items():
            acc_df = acc_df.loc[acc_df[key] == value]

    acc_fields_ = ['method'] + acc_fields
    acc_df = acc_df.copy()
    acc_df = acc_df[acc_fields_]

    methods = np.unique(acc_df['method'].values)
    std_df = pd.DataFrame(index=acc_fields, columns=methods)

    num = 0
    for i, m in enumerate(methods):
        df = acc_df[acc_df['method'] == m]
        df = df.drop(columns='method')
        df *= 100
        if i == 0:
            num = len(df)
        else:
            try:
                # make sure that each method have the same number of rows
                assert num == len(df)
            except AssertionError:
                print("Error! Compute accuracy standard deviation, number of "
                      "accuracy data rows "
                      "of method '{}' is not equal to the methods {}".
                      format(m, methods[:i]))
                sys.exit()
        acc_mean = df.values.mean(axis=0)
        acc_std = df.values.std(axis=0)
        std_df[m] = ['%.2f ± %.2f' % (m, s) for m, s in zip(acc_mean, acc_std)]

    std_df.to_csv(save_dir, index=True)

    return std_df


class ResultOutput:
    """a class can output training results:
    1) output one result record (accuracies, parameters, ...) after each
       training
    2) display result maps and save

    Parameters
    ----------
    pred: predicted map, ndarray: H*W
    target: ground truth, ndarray: H*W
    train_mask/val_mask/test_mask: ndarray: H*W
    map_save: result map save path, a folder, str, e.g. './maps'
    acc_save: accuracy save path, a '.csv' file path, str, e.g. './accuracy.csv'
    hyper_params: dict, hyper parameters that need to be output,
            e.g. {'method': 'resnet', 'train_prop': 0.1, ...}
    train_records: dict, necessary records during the training,
                   e.g. {'train_time': **, 'predict_time': **, best_record:
                   {epoch: **, batch: **, loss: **, ...}, ...}

    Attributes
    ----------

    Methods
    -------
    output(): output results (maps, accuracies and so on)

    """

    def __init__(self, pred, data, target, train_mask, val_mask, test_mask,
                 map_save, acc_save, hyper_params=None, train_records=None):

        self.pred = pred
        self.data = data
        self.target = target
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.map_save = check_path(map_save)
        check_path(os.path.dirname(acc_save))
        self.acc_save = acc_save
        self.hyper_params = hyper_params
        self.train_records = train_records

        if self.hyper_params is None:
            self.hyper_params = {}
        if self.train_records is None:
            self.train_records = {}

    def _compute_accuracies(self):

        train_oa = compute_accuracy_from_mask(self.pred, self.target,
                                              self.train_mask)
        val_oa = compute_accuracy_from_mask(self.pred, self.target,
                                            self.val_mask)

        pred = self.pred[self.test_mask == 1]
        target = self.target[self.test_mask == 1]

        test_oa = compute_accuracy_from_mask(self.pred, self.target,
                                             self.test_mask)
        test_ca = compute_accuracy_each_class(self.pred, self.target,
                                              self.test_mask)
        test_aa = np.mean(test_ca)
        test_kp = compute_kappa_coef(pred, target)

        acc_dict = {'train_oa': train_oa, 'val_oa': val_oa, 'test_oa': test_oa,
                    'test_aa': test_aa, 'test_kappa': test_kp}
        for i in range(len(test_ca)):
            acc_dict['test_a%s' % (i + 1)] = test_ca[i]

        conf_m = get_confusion_matrix(pred, target)

        return acc_dict, conf_m

    def _save_maps(self):

        pred_ = self.pred.copy()
        pred_[self.target == 0] = 0

        rgb_img = self.data.transpose(1, 2, 0)
        rgb_bands = self.hyper_params.get('data', {})
        rgb_bands = rgb_bands.get('rgb_bands', [0, 1, 2])
        rgb_img = rgb_img[:, :, rgb_bands]

        # .png
        target_png = os.path.join(self.map_save, 'label.png')
        data_png = os.path.join(self.map_save, 'rgb_image.png')
        pred_png = os.path.join(self.map_save, 'pred_map.png')
        pred_png_ = os.path.join(self.map_save, 'pred_map_rm.png')

        display_map_and_save(self.target, target_png)
        display_map_and_save(rgb_img, data_png)
        display_map_and_save(self.pred, pred_png)
        display_map_and_save(pred_, pred_png_)

        # .mat
        sio.savemat(os.path.join(self.map_save, 'train_mask.mat'),
                    {'train_mask': self.train_mask})
        sio.savemat(os.path.join(self.map_save, 'val_mask.mat'),
                    {'val_mask': self.val_mask})
        sio.savemat(os.path.join(self.map_save, 'test_mask.mat'),
                    {'test_mask': self.test_mask})
        sio.savemat(os.path.join(self.map_save, 'label.mat'),
                    {'label': self.target})
        sio.savemat(os.path.join(self.map_save, 'pred_map.mat'),
                    {'pred_map': self.pred})
        sio.savemat(os.path.join(self.map_save, 'pred_map_rm.mat'),
                    {'pred_map_rm': pred_})

    def output(self):

        try:
            df = pd.read_csv(self.acc_save, engine='python')
            record_id = str(len(df) + 1)
        except IOError:
            record_id = '1'

        # compute accuracies
        acc_dict, conf_m = self._compute_accuracies()

        # write hyper parameters, training records and accuracies to a csv file
        records_to_csv(record_id, self.hyper_params, self.train_records,
                       acc_dict, self.acc_save)

        # save classification maps, ground truth and masks
        self._save_maps()

        # write hyper parameters, training records and accuracies to a txt file
        sign = 100
        blank1 = sign // 2 - 16 // 2
        blank2 = sign // 2 - 10 // 2
        s1 = '\n%s\n' % \
             time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        s2 = '\n%s\n%s%s\n%s\n' % ('#' * sign, ' ' * blank1, 'Hyper Parameters',
                                   '#' * sign)
        s3 = '\n%s\n%s%s\n%s\n' % ('#' * sign, ' ' * blank1, 'Training Records',
                                   '#' * sign)
        s4 = '\n%s\n%s%s\n%s\n' % ('#' * sign, ' ' * blank2, 'Accuracies',
                                   '#' * sign)
        # s2 = '\n%s%s\n' % (' ' * blank1, 'Hyper Parameters')
        # s3 = '\n%s%s\n' % (' ' * blank1, 'Training Records')
        # s4 = '\n%s%s\n' % (' ' * blank2, 'Accuracies')

        with open(check_path(os.path.join(self.map_save, 'report.txt')), 'w') \
                as f:
            f.write(s1)
            f.write(s2)
            f.write(recur_str_dict_for_show(self.hyper_params))
            f.write(s3)
            f.write(recur_str_dict_for_show(self.train_records))
            f.write(s4)
            f.write(str_dict_for_show(acc_dict))
            f.write('\ntest_confusion_matrix: \n')
            np.savetxt(f, conf_m, fmt='%5d', newline='\n')

