# -*- coding: utf-8 -*-

import os
import logging

__all__ = ['check_path', 'str_dict', 'str_dict_for_show',
           'recur_str_dict_for_show', 'flatten_a_nested_dict',
           'combine_dicts', 'recur_combine_dicts', 'MyLogger']


def check_path(path):
    """check a file path or folder, if the dirname is not find then creat it
    and return the input, for a file path, if it is exist then return a path
    string that obtained by adding a number suffix to the input, else return
    the input

    Parameters
    ----------
    path: str, a file path or a folder

    Returns
    -------
    path: str, a file path or a folder

    """

    path_tuple = os.path.splitext(path)

    if path_tuple[1] == '':
        path_dirname = path_tuple[0]
    elif path_tuple[1].split('.')[1].isdigit():
        path_dirname = path
    else:
        path_dirname = os.path.dirname(path)

    if not os.path.exists(path_dirname):
        print("Check path: '%s', folder '%s' not found! will create it ..." %
              (path, path_dirname))
        os.makedirs(path_dirname)

    if os.path.isfile(path):
        if os.path.exists(path):
            i = 1
            while True:
                path = '%s(%s)%s' % (path_tuple[0], i, path_tuple[1])
                if not os.path.exists(path):
                    break
                i += 1

    return path


def str_dict(dict):
    """convert a dict to a string

    Parameters
    ----------
    dict: dictionary

    Returns
    -------
    out_str: string

    """

    out_str = ''

    for key, value in dict.items():
        out_str += '%s: %s ' % (key, value)

    return out_str


def str_dict_for_show(dict, total_space=100):
    """convert a dict to a string so as to print in terminal

    Parameters
    ----------
    dict: dict
    total_space: int, display width

    Returns
    -------
    out_str: string

    """

    total_space = int(round(total_space / 2))

    keys = list(dict.keys())
    max_key = str(keys[0])  # the key that length is maximum
    for key in keys:
        if len(str(key)) > len(max_key):
            max_key = str(key)

    key_space = len(max_key) + 2
    value_space = total_space - key_space
    key_phd = '%%-%ds' % key_space  # placeholder
    value_phd = '%%-%ds' % value_space

    out_str = '%s\n' % ('-' * total_space * 2)
    flag = 0
    for key, value in dict.items():
        key = str(key)
        if key_space + len(str(value)) > total_space:
            out_str += key_phd % (str(key) + ': ') + '%s\n' % str(value) \
                if flag == 0 \
                else '\n' + key_phd % (str(key) + ': ') + '%s\n' % str(value)
            flag = 0
        elif flag == 0:
            out_str += key_phd % (str(key) + ': ') + value_phd % str(value)
            flag = 1
        elif flag == 1:
            out_str += key_phd % (str(key) + ': ') + value_phd % str(value) + \
                '\n'
            flag = 0

    out_str = out_str[:-1]
    out_str = '%s\n%s\n' % (out_str, '-' * total_space * 2)

    return out_str


def recur_str_dict_for_show(input_dict, total_space=100):
    """recursively convert a nested dict to a string for showing in terminal or
       write to file.

    Parameters
    ----------
    input_dict: a nested dict
    total_space

    Returns
    -------

    """

    def _recur_str_dict_for_show(input_dict, total_space, indent=0):

        half_space = int(round(total_space / 2))

        keys = list(input_dict.keys())
        max_key = str(keys[0])  # the key that length is maximum
        for key in keys:
            if len(str(key)) > len(max_key):
                max_key = str(key)

        key_space = len(max_key) + 2
        value_space = half_space - key_space
        key_phd = '%%-%ds' % key_space  # placeholder
        value_phd = '%%-%ds' % value_space

        out_str = '%s%s\n' % (' ' * indent, '-' * total_space)
        flag = 0
        for key, value in input_dict.items():
            if isinstance(value, dict):
                out_str += '%s%s\n' % (
                ' ' * indent, str(key) + ':') if flag == 0 \
                    else '\n%s%s\n' % (' ' * indent, str(key) + ':')
                indent_ = indent + len(key) + 2
                total_space_ = total_space - (len(key) + 2)
                out_str += _recur_str_dict_for_show(
                    value, total_space_, indent_)
                out_str += '%s%s\n' % (' ' * indent_, '-' * total_space_)
                flag = 0
            else:
                key = str(key)
                if key_space + len(str(value)) > half_space:
                    out_str += ' ' * indent + key_phd % (
                                str(key) + ': ') + '%s\n' \
                               % str(value) \
                        if flag == 0 \
                        else '\n' + ' ' * indent + key_phd % (str(key) + ': ') \
                             + '%s\n' % str(value)
                    flag = 0
                elif flag == 0:
                    out_str += ' ' * indent + key_phd % (str(key) + ': ') + \
                               value_phd % str(value)
                    flag = 1
                elif flag == 1:
                    out_str += key_phd % (str(key) + ': ') + value_phd % str(
                        value) \
                               + '\n'
                    flag = 0

        out_str = out_str + '\n' if flag == 1 else out_str

        return out_str

    out_str = _recur_str_dict_for_show(input_dict, total_space)
    out_str += '%s\n' % ('-' * total_space)

    return out_str


def flatten_a_nested_dict(input_dict, parent_key=''):

    out_dict = {}
    for key, value in input_dict.items():
        new_key = '%s.%s' % (parent_key, key) if parent_key else key
        if isinstance(value, dict):
            out_dict.update(flatten_a_nested_dict(value, new_key))
        else:
            out_dict.update({new_key: value})

    return out_dict


def combine_dicts(*dicts):
    """combine dicts

    Parameters
    ----------
    dicts: many dicts, [dict1, dict2, ...]

    Returns
    -------
    dict_: merged dict

    """

    dict_ = dicts[0]
    for dict in dicts[1:]:
        dict_ = {k: v for d in [dict_, dict] for k, v in d.items()}

    return dict_


def recur_combine_dicts(*dicts):
    """recursively combine dicts

    Parameters
    ----------
    dicts: dicts list, where each dict may be a nested dict, [dict1, dict2, ...]

    Returns
    -------
    dict_: merged dict

    """

    def combine_two_dicts(dict1, dict2):
        for key, value in dict2.items():
            if key not in dict1.keys():
                dict1[key] = value
            else:
                if isinstance(value, dict):
                    combine_two_dicts(dict1[key], value)
                else:
                    dict1[key] = value
        return dict1

    dict_ = dicts[0]
    for i in range(1, len(dicts)):
        dict_ = combine_two_dicts(dict_, dicts[i])

    return dict_


class MyLogger:
    """wrap a logger

    Parameters
    ----------
    file_path: str, log file save path

    """

    file_path = os.path.join(os.getcwd(), 'logfile.log')
    count = 0

    def __init__(self, file_path=None):

        MyLogger.count += 1
        self.logger = logging.getLogger('logger_%d' % MyLogger.count)
        if file_path is not None:
            MyLogger.file_path = file_path

        self.fh_format = '%(asctime)s (%(module)s.%(funcName)s) %(message)s'
        self.ch_format = '(%(module)s.%(funcName)s) %(message)s'

    def get_logger(self):

        fh = logging.FileHandler(filename=MyLogger.file_path, mode='a')
        ch = logging.StreamHandler()

        fh.setFormatter(logging.Formatter(fmt=self.fh_format,
                                          datefmt='%Y-%m-%d %H:%M:%S'))
        ch.setFormatter(logging.Formatter(fmt=self.ch_format))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.INFO)

        fh.close()
        ch.close()

        return self.logger


if __name__ == '__main__':

    # input_dict = {
    #     "method": "ssdip",
    #     "gpu": "0",
    #     "data_dir": "../hsi_data/indian_pines/Indian_pines_corrected.mat",
    #     "target_dir": "../hsi_data/indian_pines/Indian_pines_gt.mat",
    #     "train_prop": 100,
    #     "val_prop": 0.2,
    #     "split_size": 145,
    #     "overlap": 0,
    #     "lr": 0.0002,
    #     "epoch": 10,
    #     "batch_size": 4,
    #     "noise_var": 0.05,
    #     "input_channels": 200,
    #     "data_dir1": "../hsi_data/indian_pines/Indian_pines_corrected.mat",
    #     "target_dir2": "../hsi_data/indian_pines/Indian_pines_gt.mat",
    #     "output_channels": 16,
    #     "spectral_channels": 128,
    #     "data_dir3": "../hsi_data/indian_pines/Indian_pines_corrected.mat",
    #     "spatial_channels": 128,
    #     "spectral_blocks": 3,
    #     "spatial_blocks": 3,
    #     "neighb_num": 8,
    #     "prob_thre": 0.98,
    #     "iter": 5,
    #     "filter_size": 5,
    #     "crf_data": "../hsi_data/indian_pines/pc3.mat",
    #     "data_dir4": "../hsi_data/indian_pines/Indian_pines_corrected.mat"
    # }

    input_dict = {
        'method_name': '3D-CNN-Li',
        'gpu': '0',
        'result_dir': './results',
        'folder_level': ['method_name', 'data_name', 'train_prop'],
        'model': {
            'input_channels': 200,
            'n_classes': 16,
            'patch_size': 5
        },
        'data': {
            'data_name': 'indian_pines',
            'data_dir': '../hsi_data/indian_pines/Indian_pines_corrected.mat',
            'target_dir': '../hsi_data/indian_pines/Indian_pines_gt.mat',
            'train_prop': 0.05,
            'val_prop': 0.05,
            'patch_size': 5
        },
        'train': {
            'epoch': 3000,
            'batch_size': 200,
            'optimizer': 'SGD',
            'optimizer_params': {
                'lr': 0.0001,
                'momentum': 0.9,
                'weight_decay': 0.0005
            },
            'scheduler': 'StepLR',
            'scheduler_params': {
                'step_size': 500,
                'gamma': 0.5
            }
        }
    }

    # input_dict = {
    #     'train': {
    #         'epoch': 3000,
    #         'batch_size': 200,
    #         'optimizer': 'SGD',
    #         'optimizer_params': {
    #             'lr': 0.0001,
    #             'momentum': 0.9,
    #             'weight_decay': 0.0005
    #         },
    #         'scheduler': 'StepLR',
    #         'scheduler_params': {
    #             'step_size': 500,
    #             'gamma': 0.5
    #         }
    #     }
    # }

    # print(str_dict(dict))
    # print(str_dict_for_show(dict, 90))
    out_str = recur_str_dict_for_show(input_dict, total_space=70)
    print(out_str)

    # dict1 = {'a': 1, 'b': 2, 'c': 3}
    # dict2 = {'d': 4, 'e': 5, 'f': 6}
    # dict3 = {'g': 7, 'h': 8, 'i': 9}
    # dict4 = {'j': 10, 'k': 11, 'l': 12}
    # print(combine_dicts(dict1, dict2, dict3, dict4))
    #
    # dict1 = {'a': {'a': 1, 'b': {'a': 1, 'b': 2}}, 'b': 2, 'c': {'a': 1, 'b': 2}}
    # dict2 = {'a': {'a': 1, 'b': {'a': 1, 'b': 3}}, 'e': 5, 'f': 6}
    # dict3 = {'g': 7, 'h': 8, 'i': 9}
    # dict4 = {'a': {'a': 1, 'b': {'a': 1, 'b': 3}, 'c': 1}, 'k': 11, 'c': 12}
    #
    # print(recur_combine_dicts(dict1, dict2, dict3, dict4))

    # print(flatten_a_nested_dict(dict1))

