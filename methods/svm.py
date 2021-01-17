from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import scipy.io as sio
from sklearn.svm import SVC
from sklearn import model_selection

from config import *


class SVM:
    """
    svm_params: dict, e.g. {'class_weight': 'balanced', ...}
    svm_grid_params: list, e.g.
    [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3], 'C': [1, 10, 100, 1000]},
     {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
     {'kernel': ['poly'], 'degree': [3], 'gamma': [1e-1, 1e-2, 1e-3]}]
    """

    def __init__(self, svm_params, svm_grid_params):

        self.svm_params = svm_params
        self.svm_grid_params = svm_grid_params
        self.estimator = SVC(**self.svm_params)
        self.best_params = []

    def train_svm(self, train_data, train_target, **kwargs):

        estimator = SVC(**self.svm_params)
        estimator = model_selection.GridSearchCV(estimator,
                                                 self.svm_grid_params, **kwargs)
        estimator.fit(train_data, train_target)
        print("SVM best parameters : {0}".format(estimator.best_params_))

        self.estimator = estimator
        self.best_params = estimator.best_params_

        return self.estimator, self.best_params

    def get_unary(self, data):

        # data: C*H*W

        assert 'probability' in self.svm_params.keys() and \
               self.svm_params['probability'] is True
        data = data.transpose(1, 2, 0)
        unary = self.estimator.predict_proba(data.reshape(-1, data.shape[2]))
        unary = unary.reshape(data.shape[:2] + (-1, ))

        return unary

    def predict(self, data):

        # data: C*H*W

        data = data.transpose(1, 2, 0)
        prediction = self.estimator.predict(data.reshape(-1, data.shape[2]))
        prediction = prediction.reshape(data.shape[:2]) + 1

        return prediction


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

    method_name = params.method_name
    result_dir = params.result_dir
    folder_level = params.folder_level

    svm_params = params.svm_params
    svm_grid_params = params.svm_grid_params

    train_prop = train_prop if train_prop < 1 else int(train_prop)
    val_prop = val_prop if val_prop < 1 else int(val_prop)

    result_root = result_dir
    local_v = locals()
    for s in folder_level:
        result_dir = check_path(os.path.join(result_dir, str(local_v[s])))

    # define output dirs
    acc_dir = os.path.join(result_root, 'accuracy.csv')
    log_dir = os.path.join(result_dir, 'train.log')
    # model_dir = os.path.join(result_dir, 'weights.pkl')
    soft_dir = os.path.join(result_dir, 'soft_label.mat')
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

    # ######################### get data, train model, predict ###############

    logger.info('get data ...')
    data, target = read_data(data_dir, target_dir)
    mask_dir = os.path.dirname(data_dir)

    train_mask, val_mask, test_mask = load_masks(mask_dir, target, train_prop,
                                                 val_prop)
    train_data, train_target = get_vector_samples(data, target, train_mask)

    logger.info('train model ...')
    svm = SVM(svm_params, svm_grid_params)
    s = time.time()
    clf, _ = svm.train_svm(train_data, train_target, verbose=1, n_jobs=16)
    e = time.time()
    train_time = e - s
    logger.info('training time: %.4fs' % train_time)

    logger.info('begin to predict ...')
    s = time.time()
    pred = svm.predict(data)
    e = time.time()
    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    # ########################### output ####################################

    logger.info('save soft label ...')
    unary = svm.get_unary(data)
    sio.savemat(soft_dir, {'soft_label': unary})

    logger.info('save classification maps etc. ...')
    train_records = {
        'train_time': '%.4f' % train_time,
        'pred_time': '%.4f' % pred_time
    }

    ro = ResultOutput(pred, data, target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()






