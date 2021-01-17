"""
3-D deep learning approach for remote sensing image classification
Ben Hamida Amina, Benoit Alexandre, Lambert Patrick, Ben Amar Chokri
IEEE Transactions on Geoscience and Remote Sensing 2018
"""

import os
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import scipy.io as sio

from config import *


class ThreeDCnnHa6(nn.Module):

    def __init__(self, input_channels, n_classes, patch_size=5):

        super(ThreeDCnnHa6, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), (1, 1, 1), (1, 0, 0))
        self.pool1 = nn.Conv3d(20, 20, (3, 3, 3), (2, 2, 2), (1, 1, 1))
        self.conv2 = nn.Conv3d(20, 35, (3, 1, 1), (1, 1, 1), (0, 0, 0))
        self.pool2 = nn.Conv3d(35, 35, (2, 1, 1), (2, 2, 2), (0, 0, 0))
        self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        self.pool3 = nn.Conv3d(35, 35, (3, 1, 1), (2, 2, 2), (1, 0, 0))

        self.features_num = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_num, n_classes)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ThreeDCnnHa8(nn.Module):

    def __init__(self, input_channels, n_classes, patch_size=5):

        super(ThreeDCnnHa8, self).__init__()

        self.input_channels = input_channels
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), (1, 1, 1), (1, 0, 0))
        self.pool1 = nn.Conv3d(20, 2, (3, 1, 1), (2, 1, 1), (1, 0, 0))
        self.conv2 = nn.Conv3d(2, 35, (3, 3, 3), (1, 1, 1), (1, 1, 1))
        self.pool2 = nn.Conv3d(35, 2, (2, 1, 1), (2, 1, 1), (1, 0, 0))
        self.conv3 = nn.Conv3d(2, 35, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        self.pool3 = nn.Conv3d(35, 2, (1, 1, 1), (2, 1, 1), (1, 0, 0))
        self.conv4 = nn.Conv3d(2, 35, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        self.pool4 = nn.Conv3d(35, 4, (1, 1, 1), (2, 2, 2), (0, 0, 0))

        self.features_num = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_num, n_classes)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.pool3(self.conv3(x))
            x = self.pool4(self.conv4(x))
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.pool2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.pool3(x))
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def run(params):
    """perform training
       save: accuracy table (.csv),
             model (.pkl),
             loss curve (.png)
             classification map (.png, .mat),
             soft label (.mat),
             training report (.txt),
             training log (.log)
             train/val/test masks (.mat)

    :param params: CfgNode type (package: yacs)
    :return:
    """

    # ##################### get parameters and define logger ################

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    # get parameters
    data_name = params.data.data_name
    data_dir = params.data.data_dir
    target_dir = params.data.target_dir
    train_prop = params.data.train_prop
    val_prop = params.data.val_prop
    patch_size = params.data.patch_size
    model_params = params.model
    train_params = params.train
    method_name = params.method_name
    result_dir = params.result_dir
    folder_level = params.folder_level

    train_prop = train_prop if train_prop < 1 else int(train_prop)
    val_prop = val_prop if val_prop < 1 else int(val_prop)

    result_root = result_dir
    local_v = locals()
    for s in folder_level:
        result_dir = check_path(os.path.join(result_dir, str(local_v[s])))

    # define output dirs
    acc_dir = os.path.join(result_root, 'accuracy.csv')
    log_dir = os.path.join(result_dir, 'train.log')
    model_dir = os.path.join(result_dir, 'weights.pkl')
    soft_dir = os.path.join(result_dir, 'soft_label.mat')
    loss_dir = os.path.join(result_dir, 'loss_curve.png')

    # define logger
    logger = define_logger(log_dir)

    # print parameters
    num1 = 25
    num2 = 100
    logger.info('%s begin a new training: %s %s' % ('#' * num1, method_name,
                                                    '#' * num1))
    params_str = recur_str_dict_for_show(params, total_space=num2)
    logger.info('show parameters ... \n%s' % params_str)

    # ########################### dataset, model ############################

    logger.info('produce dataset ...')
    dataset = PatchDataset(data_dir, target_dir, train_prop, val_prop,
                           patch_size)
    raw_data = dataset.raw_data
    raw_target = dataset.raw_target
    dataset.train = dataset.train[0].unsqueeze(dim=1), dataset.train[1]
    dataset.val = dataset.val[0].unsqueeze(dim=1), dataset.val[1]
    dataset.set_state('train')

    logger.info('initialize model ...')
    model = ThreeDCnnHa6(**model_params)
    # model = ThreeDCnnHa8(**model_params)

    #  ############################# training ##############################

    logger.info('begin to train ...')
    s = time.time()
    output = train_bp_cnn(model, dataset, train_params)

    e = time.time()
    logger.info('write best record to log ... \n%s' %
                str_dict_for_show(output['best'], total_space=num2))

    train_time = e - s
    train_time_mean = train_time / params.train.epoch
    logger.info('training time: %.4fs' % train_time)
    logger.info('training time mean: %.4fs' % train_time_mean)

    logger.info('save model ...')
    torch.save(output['model'].state_dict(), model_dir)

    logger.info('plot loss and accuracy curves ...')
    graph_data = {
        'loss': output['loss'],
        'train accuracy': output['train'],
        'val accuracy': output['val'],
    }
    plot_loss_and_accuracy(graph_data=graph_data, save_dir=loss_dir,
                           title='Loss and Accuracy')

    # ############################## prediction ############################

    model_ = output['model']
    all_patch = get_all_patches(raw_data, patch_size)
    all_patch = torch.from_numpy(all_patch).float().unsqueeze(dim=1)
    all_target = raw_target.ravel()
    all_target = torch.from_numpy(all_target).long()
    dataset.data, dataset.target = all_patch, all_target

    logger.info('begin to predict ...')
    s = time.time()
    pred, proba, _ = test_bp_cnn(model_, dataset)
    e = time.time()
    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    logger.info('save soft label ...')
    pred = pred.reshape(raw_target.shape) + 1
    proba = proba.reshape(raw_target.shape + (-1,))
    sio.savemat(soft_dir, {'soft_label': proba})

    # output predicted map(png/mat), accuracy table and other records
    logger.info('save classification maps etc. ...')
    train_records = {
        'best': output['best'],
        'train_time': '%.4f' % train_time,
        'train_time_mean': '%.4f' % train_time_mean,
        'pred_time': '%.4f' % pred_time
    }
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    ro = ResultOutput(pred, raw_data, raw_target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()

