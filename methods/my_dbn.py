# -*- coding: utf-8 -*-

"""
Classification of hyperspectral image based on deep belief networks
Li, Tong and Zhang, Junping and Zhang, Ye
2014 IEEE international conference on image processing (ICIP)

Spectral–Spatial Classification of Hyperspectral Data Based on Deep Belief
Network
Chen, Yushi and Zhao, Xing and Jia, Xiuping
IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
2015

"""

import os
import time

import numpy as np

from config import *


def run(params):

    # ##################### get parameters and define logger ################

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    # get parameters
    data_name = params.data.data_name
    data_dir = params.data.data_dir
    target_dir = params.data.target_dir
    train_prop = params.data.train_prop
    val_prop = params.data.val_prop

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
    # soft_dir = os.path.join(result_dir, 'soft_label.mat')
    # loss_dir = os.path.join(result_dir, 'loss_curve.png')

    # define logger
    logger = define_logger(log_dir)

    # print parameters
    num1 = 25
    num2 = 100
    logger.info('%s begin a new training: %s %s' % ('#' * num1, method_name,
                                                    '#' * num1))
    params_str = recur_str_dict_for_show(params, total_space=num2)
    logger.info('show parameters ... \n%s' % params_str)

    # ########################### get data, train ############################

    logger.info('get data ...')
    mask_dir = os.path.dirname(data_dir)
    data, target = read_data(data_dir, target_dir)
    train_mask, val_mask, test_mask = load_masks(mask_dir, target, train_prop,
                                                 val_prop)
    x_train, y_train = get_vector_samples(data, target, train_mask)

    logger.info('get model ...')
    from dbn.tensorflow import SupervisedDBNClassification
    classifier = SupervisedDBNClassification(**train_params)

    logger.info('begin to train ...')
    s = time.time()
    classifier.fit(x_train, y_train)
    e = time.time()
    train_time = e - s
    logger.info('training time: %.4fs' % train_time)

    logger.info('save model ...')
    classifier.save(model_dir)

    # ########################### predict, output ###########################

    all_data = data.reshape(-1, data.shape[1] * data.shape[2]).T

    classifier = SupervisedDBNClassification.load(model_dir)

    logger.info('begin to predict ...')
    s = time.time()
    pred = classifier.predict(all_data)
    pred = np.array(pred)
    pred = pred.reshape(target.shape) + 1
    e = time.time()
    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    # output predicted map(png/mat), accuracy table and other records
    logger.info('save classification maps etc. ...')
    train_records = {
        'train_time': '%.4f' % train_time,
        'pred_time': '%.4f' % pred_time
    }

    ro = ResultOutput(pred, data, target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()

