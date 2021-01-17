
import os
import time

import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization

from config import *


def rfc_cv(n_estimators, max_features, max_depth, data, target):
    estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                max_features=max_features,
                max_depth=max_depth,
                n_jobs=16
    )
    cval = cross_val_score(estimator, data, target, scoring='accuracy', cv=3)
    return cval.mean()


def optimize_rfc(train_data, train_target, opt_params, iter):
    estimators = opt_params['n_estimators']
    features = opt_params['max_features']
    depth = opt_params['max_depth']

    def rfc_crossval(n_estimators, max_features, max_depth):
        return rfc_cv(
                n_estimators=int(n_estimators),
                max_features=int(max_features),
                max_depth=int(max_depth),
                data=train_data,
                target=train_target,
        )

    optimizer = BayesianOptimization(
            f=rfc_crossval,
            pbounds={
                "n_estimators": estimators,
                "max_features": features,
                "max_depth": depth,
            },
            verbose=2
    )
    optimizer.maximize(n_iter=iter, alpha=1e-4)
    print("Best RandomForest params: {}".format(optimizer.max['params']))

    estimator = RandomForestClassifier(
        n_estimators=int(optimizer.max['params']['n_estimators']),
        max_depth=int(optimizer.max['params']['max_depth']),
        max_features=int(optimizer.max['params']['max_features']),
        n_jobs=16
    )
    estimator.fit(train_data, train_target)
    # joblib.dump(estimator, 'rfc.pkl')
    return estimator


class RandomForest:
    """
    rf_params: dict, e.g. {'n_estimators': (100, 200), 'max_features':
    (1, 200), 'max_depth': (1, 50)}
    crf_params: dict, e.g. {''filter_size': 5, ...}
    """

    def __init__(self, rf_params):

        self.rf_params = rf_params
        self.estimator = None

    def train_rf(self, train_data, train_target, iter):

        self.estimator = optimize_rfc(train_data, train_target,
                                      self.rf_params, iter=iter)
        return self.estimator

    def get_unary(self, data):

        # data: C*H*W

        data = data.transpose(1, 2, 0)
        unary = self.estimator.predict_proba(data.reshape(-1, data.shape[2]))
        unary = unary.reshape(data.shape[:2] + (-1,))

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

    rf_params = params.rf_params
    iter = params.iter

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

    rfc = RandomForest(rf_params)
    s = time.time()
    rfc.train_rf(train_data, train_target, iter=iter)
    e = time.time()
    train_time = e - s
    logger.info('training time: %.4fs' % train_time)

    logger.info('begin to predict ...')
    s = time.time()
    pred = rfc.predict(data)
    e = time.time()
    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    # ########################### output ####################################

    logger.info('save soft label ...')
    unary = rfc.get_unary(data)
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

