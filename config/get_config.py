# -*- coding: utf-8 -*-

import sys
import yaml

from yacs.config import CfgNode as CN

from . import MyLogger, recur_combine_dicts, flatten_a_nested_dict

__all__ = ['get_params1', 'get_params2', 'cfgnode_to_dict', 'define_logger',
           'method_name_dict']

method_name_dict = {
    '3D-CNN-Li': '3d_cnn_li',
    '3D-CNN-Hamida': '3d_cnn_hamida',
    'DeepFCN': 'deepfcn',
    'Semi-2D-CNN': 'semi_2d_cnn',
    'SSRN': 'ssrn',
    'HybridSN': 'hybridsn',
    'SAE': 'sae',
    'DBN': 'my_dbn',
    'SVM': 'svm',
    'RandomForest': 'rf'
}

common_params = {
    'data': {
        'indian_pines': [
            'data.data_name', 'indian_pines',
            'data.data_dir', '/home/lthpc/hsi_classification/data/indian_pines/'
                             'Indian_pines_corrected.mat',
            'data.target_dir', '/home/lthpc/hsi_classification/data/'
                               'indian_pines/Indian_pines_gt.mat',
            'data.rgb_bands', '[29, 10, 11]'
        ],
        'pavia_university': [
            'data.data_name', 'pavia_university',
            'data.data_dir', '/home/lthpc/hsi_classification/data/paviau/'
                             'PaviaU.mat',
            'data.target_dir', '/home/lthpc/hsi_classification/data/paviau/'
                               'PaviaU_gt.mat',
            'data.data_dir', '/home/lthpc/hsi_classification_test/data/paviau/'
                             'PaviaU.mat',
            'data.target_dir', '/home/lthpc/hsi_classification_test/data/paviau/'
                               'PaviaU_gt.mat',
            'data.rgb_bands', '[56, 33, 13]'
        ],
        'salinas': [
            'data.data_name', 'salinas',
            'data.data_dir', '/home/lthpc/hsi_classification/data/salinas/'
                             'Salinas_corrected.mat',
            'data.target_dir', '/home/lthpc/hsi_classification/data/salinas/'
                               'Salinas_gt.mat',
            'data.rgb_bands', '[56, 33, 13]'
        ],
        'ksc': [
            'data.data_name', 'ksc',
            'data.data_dir', '/home/lthpc/hsi_classification/data/ksc/KSC.mat',
            'data.target_dir', '/home/lthpc/hsi_classification/data/ksc/'
                               'KSC_gt.mat',
            'data.rgb_bands', '[31, 21, 11]'
        ]
    },
    '3d_cnn_li': {
        'indian_pines': [
            'model.input_channels', 200,
            'model.n_classes', 16,
            'model.patch_size', 5,
        ],
        'pavia_university': [
            'model.input_channels', 103,
            'model.n_classes', 9,
            'model.patch_size', 5,
        ],
        'salinas': [
            'model.input_channels', 204,
            'model.n_classes', 16,
            'model.patch_size', 5,
        ],
        'ksc': [
            'model.input_channels', 176,
            'model.n_classes', 13,
            'model.patch_size', 5,
        ]
    },
    '3d_cnn_hamida': {
        'indian_pines': [
            'model.input_channels', 200,
            'model.n_classes', 16,
            'model.patch_size', 5,
            'train.scheduler_params.milestones', '[500, 1400]'
        ],
        'pavia_university': [
            'model.input_channels', 103,
            'model.n_classes', 9,
            'model.patch_size', 5,
            'train.scheduler_params.milestones', '[800, 1400]'
        ],
        'salinas': [
            'model.input_channels', 204,
            'model.n_classes', 16,
            'model.patch_size', 5,
            'train.scheduler_params.milestones', '[1200, 1400]'
        ],
        'ksc': [
            'model.input_channels', 176,
            'model.n_classes', 13,
            'model.patch_size', 5,
            'train.scheduler_params.milestones', '[800, 1400]'
        ]
    },
    'deepfcn': {
        'indian_pines': [
            'model.in_channels', 200,
            'model.n_classes', 17,
            'train.criterion_params.weight',
            '[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
        ],
        'pavia_university': [
            'model.in_channels', 103,
            'model.n_classes', 10,
            'train.criterion_params.weight',
            '[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
        ],
        'salinas': [
            'model.in_channels', 204,
            'model.n_classes', 17,
            'train.criterion_params.weight',
            '[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
        ],
        'ksc': [
            'model.in_channels', 176,
            'model.n_classes', 14,
            'train.criterion_params.weight',
            '[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]'
        ]
    },
    'semi_2d_cnn': {
        'indian_pines': [
            'model.input_channels', 200,
            'model.n_classes', 16,
            'model.patch_size', 9,
        ],
        'pavia_university': [
            'model.input_channels', 103,
            'model.n_classes', 9,
            'model.patch_size', 9,
        ],
        'salinas': [
            'model.input_channels', 204,
            'model.n_classes', 16,
            'model.patch_size', 9,
        ],
        'ksc': [
            'model.input_channels', 176,
            'model.n_classes', 13,
            'model.patch_size', 9,
        ]
    },
    'hybridsn': {
        'indian_pines': [
            'model.output_units', 16,
            'data.n_pc', 30
        ],
        'pavia_university': [
            'model.output_units', 9,
            'data.n_pc', 15
        ],
        'salinas': [
            'model.output_units', 16,
            'data.n_pc', 15
        ],
        'ksc': [
            'model.output_units', 13,
            'data.n_pc', 15
        ]
    },
    'sae': {
        'indian_pines': [
            'model.input_dim', 200,
            'model.hidden_units', 400,
            'model.n_classes', 16
        ],
        'pavia_university': [
            'model.input_dim', 103,
            'model.hidden_units', 100,
            'model.n_classes', 9
        ],
        'salinas': [
            'model.input_dim', 204,
            'model.hidden_units', 100,
            'model.n_classes', 16
        ],
        'ksc': [
            'model.input_dim', 176,
            'model.hidden_units', 100,
            'model.n_classes', 13
        ]
    },
    'rf': {
        'indian_pines':
            ['rf_params.max_features', '[1, 200]'],
        'pavia_university':
            ['rf_params.max_features', '[1, 103]'],
        'salinas':
            ['rf_params.max_features', '[1, 204]'],
        'ksc':
            ['rf_params.max_features', '[1, 176]']
    }
}

batch_size = {
    'indian_pines': {0.01: 200, 0.02: 300, 0.03: 400, 0.05: 600, 0.1: 600,
                     0.15: 800, 0.2: 1200, 50: 50, 100: 100, 150: 150, 200: 200,
                     250: 250, 300: 300},
    'pavia_university': {0.01: 500, 0.02: 500, 0.03: 700, 0.05: 800, 0.1: 1000,
                         0.15: 1500, 0.2: 2000, 50: 50, 100: 100, 150: 150,
                         200: 200, 250: 250, 300: 300},
    'salinas': {0.01: 600, 0.02: 600, 0.03: 600, 0.05: 800, 0.1: 1000,
                0.15: 1500, 0.2: 2000, 50: 50, 100: 100, 150: 150, 200: 200,
                250: 250, 300: 300}
}


def get_params1(method_name, override_params):
    """ load 'config.yaml', get parameters of one method and override
    parameters

    :param method_name: str, e.g. '3d_cnn_li'
    :param override_params: dict or list
      e.g. dict: {train: {epoch: 2000, batch_size: 100}, model: {...}, ...}
           list: ['train.epoch', 2000, 'train.batch_size', 100,
                  'data.train_prop', 0.05, ...]
    :return: CN object
    """

    file = open('./config/config.yaml', 'rb')
    params = yaml.load(stream=file, Loader=yaml.FullLoader)
    params = params[method_name_dict[method_name]]
    if isinstance(override_params, dict):
        params = recur_combine_dicts(params, override_params)
        params = CN(init_dict=params, new_allowed=True)
    elif isinstance(override_params, list):
        params = CN(init_dict=params, new_allowed=True)
        params.merge_from_list(override_params)
    else:
        print("error! the type of 'override_params' is not dict or list, "
              "utils.py: get_params(method_name, override_params)")
        sys.exit()

    params.freeze()

    return params


def get_params2(method_name, data_name, override_params):
    """

    :param method_name:
    :param data_name:
    :param override_params: list
    :return:
    """

    file = open('./config/config.yaml', 'rb')
    params = yaml.load(stream=file, Loader=yaml.FullLoader)

    method_key = method_name_dict[method_name]
    params = params[method_key]
    params_tmp = common_params['data'][data_name] + \
                 common_params[method_key][data_name] \
        if method_key in common_params else common_params['data'][data_name]

    if isinstance(override_params, dict):
        params = recur_combine_dicts(params, override_params)
        params = CN(init_dict=params, new_allowed=True)
    elif isinstance(override_params, list):
        params = CN(init_dict=params, new_allowed=True)
        params.merge_from_list(override_params)
    else:
        print("error! the type of 'override_params' is not dict or list, "
              "utils.py: get_params(method_name, override_params)")
        sys.exit()

    params.merge_from_list(params_tmp)

    override_params = flatten_a_nested_dict(override_params) if \
        isinstance(override_params, dict) else override_params
    if 'train.batch_size' not in override_params:
        if method_key == 'ssrn':
            pass
        elif method_key  == 'hybrisn':
            pass
        elif method_key == 'sae':
            pass
        elif method_key == 'my_dbn':
            pass
        elif method_key == 'svm':
            pass
        elif method_key == 'rf':
            pass
        else:
            params.train.batch_size = \
                batch_size[data_name][params.data.train_prop]

    params.freeze()

    return params


def cfgnode_to_dict(cfg):

    cfg = dict(cfg)
    for key, value in cfg.items():
        if isinstance(value, CN):
            cfg[key] = cfgnode_to_dict(value)

    return cfg


def define_logger(log_dir, **kwargs):
    fh_format = kwargs.get('fh_format', '%(asctime)s (%(module)s) %(message)s')
    ch_format = kwargs.get('ch_format', '%(asctime)s (%(module)s) %(message)s')

    logger = MyLogger(log_dir)
    logger.fh_format = fh_format
    logger.ch_format = ch_format
    logger = logger.get_logger()

    return logger

