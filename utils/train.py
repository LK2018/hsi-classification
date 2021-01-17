# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pdb
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchsummary import summary

from .accuracy import compute_accuracy, compute_accuracy_from_mask

__all__ = ['train_bp_cnn', 'train_fcnn', 'test_bp_cnn', 'test_fcnn']


_criterion = {
    'CrossEntropy': nn.CrossEntropyLoss
}
_optimizer = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
}
_scheduler = {
    'StepLR': lr_scheduler.StepLR,
    'MultiStepLR': lr_scheduler.MultiStepLR
}


# def train_bp_cnn(model, dataset, params):
#     """train the model like bp and cnn
#
#     Parameters
#     ----------
#     model: subclass of torch.nn.Module, bp, cnn network model
#     dataset: subclass of torch.utils.data.Dataset: VectorDataset, PatchDataset
#     params: dict, training params
#
#     Returns
#     -------
#     output_dict: dict, {'model': **, 'loss': **, ...}
#
#     """
#
#     # ###################### parameters ####################################
#     lr = params['lr']
#     epoch = params['epoch']
#     batch_size = params['batch_size']
#     test_prop = params.get('test_prop', 1.)
#     test_inter = params.get('test_inter', 1)
#     noise_var = params.get('noise_ver', 0)
#
#     # ###################### define dataloader etc. ########################
#     criterion = params.get('criterion', nn.CrossEntropyLoss())
#     optimizer = params.get('optimizer', optim.Adam(model.parameters(), lr=lr))
#     scheduler = params.get('scheduler')
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     val_acc = 0.
#     train_acc = 0.
#     test_acc = 0.
#     best_val = 0.
#     best_acc_dict = {}
#     best_model = None
#     loss_list = []
#     train_acc_list = []
#     val_acc_list = []
#     test_acc_list = []
#
#     # ######################## begin to train ##############################
#     model.train()
#     model = model.cuda()
#     summary(model, input_size=dataset.data.shape[1:])
#
#     for e in range(epoch):
#
#         loss_i = 0.
#         train_i = 0.
#         val_i = 0.
#         test_i = 0.
#         count = 0
#         i = 0
#
#         for i, samples in enumerate(dataloader):
#
#             data, target = samples
#             data_ = data.detach().clone()
#             data = data + data_.normal_() * noise_var
#             data = data.cuda()
#             target = target.cuda()
#
#             output = model(data)
#             loss = criterion(output, target)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             if i % test_inter == 0:
#                 _, _, train_acc = test_bp_cnn(model, dataset, state='train')
#                 _, _, val_acc = test_bp_cnn(model, dataset, state='val')
#                 _, _, test_acc = test_bp_cnn(model, dataset, state='test',
#                                              prop=test_prop)
#                 torch.cuda.empty_cache()
#                 train_i += train_acc
#                 val_i += val_acc
#                 test_i += test_acc
#                 count += 1
#
#             loss_i += loss.item()
#
#             if val_acc > best_val:
#                 best_train = train_acc
#                 best_val = val_acc
#                 best_test = test_acc
#                 best_loss = loss.item()
#                 best_acc_dict = {'epoch': e + 1, 'loss': best_loss,
#                                  'train': best_train, 'val': best_val,
#                                  'test': best_test}
#                 best_model = copy.deepcopy(model)
#
#             # print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: '
#             #       '{3:.6f} | Val: {4:.6f} | Test: {5:.6f}'.
#             #       format(e + 1, i + 1, loss.item(), train_acc, val_acc,
#             #              test_acc), '\r', end='')
#
#             print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: '
#                   '{3:.6f} | Val: {4:.6f} | Test: {5:.6f}'.
#                   format(e + 1, i + 1, loss.item(), train_acc, val_acc,
#                          test_acc))
#
#         # if set scheduler, update it
#         if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(-(val_i / count))
#         elif scheduler is not None:
#             scheduler.step()
#
#         count = 1 if count == 0 else count
#         loss_list.append(loss_i / (i + 1))
#         train_acc_list.append(train_i / count)
#         val_acc_list.append(val_i / count)
#         test_acc_list.append(test_i / count)
#
#     print('Best Results: Epoch: {0:5}, Loss: {1:.8}, Train: {2:.6}, Val: '
#           '{3:.6}, Test: {4:.6}'.
#           format(best_acc_dict['epoch'], best_acc_dict['loss'],
#                  best_acc_dict['train'], best_acc_dict['val'],
#                  best_acc_dict['test']))
#
#     # ############################# output results #########################
#     output_dict = {
#         'model': best_model,
#         'loss': np.array(loss_list),
#         'train': np.array(train_acc_list),
#         'val': np.array(val_acc_list),
#         'test': np.array(test_acc_list),
#         'best': best_acc_dict
#     }
#
#     return output_dict


def train_bp_cnn(model, dataset, params):
    """train the model like bp and cnn

    Parameters
    ----------
    model: subclass of torch.nn.Module, bp, cnn network model
    dataset: subclass of torch.utils.data.Dataset: VectorDataset, PatchDataset
    params: dict, training params

    Returns
    -------
    output_dict: dict, {'model': **, 'loss': **, ...}

    """

    # ###################### parameters ######################################
    epoch = params['epoch']
    batch_size = params['batch_size']
    if 'criterion' in params.keys():
        criterion = _criterion[params['criterion']]()
    else:
        criterion = nn.CrossEntropyLoss()
    if 'optimizer' in params.keys():
        optimizer = _optimizer[params['optimizer']]\
            (model.parameters(), **params['optimizer_params'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if 'scheduler' in params.keys():
        scheduler = _scheduler[params['scheduler']]\
            (optimizer, **params['scheduler_params'])
    else:
        scheduler = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val = 0.
    best_record = {}
    best_model = None
    loss_list = []
    train_list = []
    val_list = []

    # ######################## begin to train ################################
    model.train()
    model = model.cuda()
    summary(model, input_size=dataset.data.shape[1:])

    for e in range(epoch):

        loss_i = 0.
        train_i = 0.
        val_i = 0.
        i = 0

        for i, (data, target) in enumerate(dataloader):

            data = data.cuda()
            target = target.cuda()
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, _, train_acc = test_bp_cnn(model, dataset, state='train')
            _, _, val_acc = test_bp_cnn(model, dataset, state='val')
            torch.cuda.empty_cache()

            train_i += train_acc
            val_i += val_acc
            loss_i += loss.item()

            if val_acc > best_val:
                best_train = train_acc
                best_val = val_acc
                best_loss = loss.item()
                best_record = {'epoch': e + 1, 'batch': i + 1, 'loss':
                    best_loss, 'train': best_train, 'val': best_val}
                best_model = copy.deepcopy(model)

            print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
                  'val: %.6f ' % (e + 1, i + 1, loss.item(), train_acc,
                                  val_acc), '\r', end='')

        # if set scheduler, update it
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-val_i)
        elif scheduler is not None:
            scheduler.step()

        loss_list.append(loss_i / (i + 1))
        train_list.append(train_i / (i + 1))
        val_list.append(val_i / (i + 1))

    print('%s best record %s' % ('-' * 40, '-' * 40))
    print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
          'val: %.6f ' % (best_record['epoch'], best_record['batch'],
                          best_record['loss'], best_record['train'],
                          best_record['val']))

    # ############################# output results ###########################
    output_dict = {
        'model': best_model,
        'loss': np.array(loss_list),
        'train': np.array(train_list),
        'val': np.array(val_list),
        'best': best_record
    }

    return output_dict


def train_fcnn(model, dataset, params):
    """train the model like fcnn

    Parameters
    ----------
    model:subclass of torch.nn.Module, network like dip
    dataset: subclass of torch.utils.data.Dataset: FullImageDataset
    params: dict, training params

    Returns
    -------
    output_dict: dict, {'model': **, map: **, ...}

    """

    # ####################### training parameters #############################
    lr = params['lr']
    epoch = params['epoch']
    batch_size = params['batch_size']
    noise_var = 0
    if 'noise_var' in params.keys():
        noise_var = params['noise_var']

    # ################ define dataloader, objective function, optimizer ... ###
    train_mask, val_mask, test_mask = dataset.train_mask, dataset.val_mask, \
                                      dataset.test_mask
    raw_target = dataset.raw_target
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model = model.cuda()

    prob_s = None
    best_val = 0
    best_acc_dict = {}
    best_model = None
    loss_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    map_output = None

    # ################################### begin to train ######################
    for e in range(epoch):
        prob = torch.zeros((len(dataset), raw_target.max(),) +
                           dataset.data.shape[2:])

        loss_i = 0.
        train_i = 0.
        val_i = 0.
        test_i = 0.
        i = 0

        for i, samples in enumerate(dataloader):

            data, target, train_mask_, idx = samples
            data_ = data.detach().clone()
            data = data + data_.normal_() * noise_var
            data = data.cuda()
            target = target.cuda()
            train_mask_ = train_mask_.cuda()

            output = model(data)
            loss = criterion(output * train_mask_, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prob[idx] = output.cpu().data
            prob_s = F.softmax(prob, dim=1).numpy()
            prob_s = dataset.stitch_data(prob_s)
            map = np.argmax(prob_s, axis=0) + 1

            train_acc = compute_accuracy_from_mask(map, raw_target, train_mask)
            val_acc = compute_accuracy_from_mask(map, raw_target, val_mask)
            test_acc = compute_accuracy_from_mask(map, raw_target, test_mask)
            train_i += train_acc
            val_i += val_acc
            test_i += test_acc

            if val_acc > best_val:
                best_train = train_acc
                best_val = val_acc
                best_test = test_acc
                best_acc_dict = {'epoch': e + 1, 'train': best_train,
                                 'val': best_val, 'test': best_test}
                best_model = copy.deepcopy(model)
                map_output = map.copy()

            print('Epoch: {0:5}, Batch: {1:3} | Loss: {2:13.8f} | Train: '
                  '{3:.6f}| Val: {4:.6f} | Test: {5:.6f}'.
                  format(e + 1, i + 1, loss.item(), train_acc, val_acc,
                         test_acc), '\r', end='')
            loss_i += loss.item()

        loss_list.append(loss_i / (i + 1))
        train_acc_list.append(train_i / (i + 1))
        val_acc_list.append(val_i / (i + 1))
        test_acc_list.append(test_i / (i + 1))

    print('\r')
    print('Best Results: Epoch: {0:5}, Train: {1:.6}, Val: {2:.6}, Test: '
          '{3:.6}'. format(best_acc_dict['epoch'], best_acc_dict['train'],
                           best_acc_dict['val'], best_acc_dict['test']))

    # ############################# output results ############################
    output_dict = {
        'model': best_model,
        'map': map_output,
        'prob': prob_s,
        'loss': np.array(loss_list),
        'train': np.array(train_acc_list),
        'val': np.array(val_acc_list),
        'test': np.array(test_acc_list),
        'best': best_acc_dict
    }

    return output_dict


def test_bp_cnn(model, dataset, state=None, prop=1.):
    """test the model like bp and cnn

    Parameters
    ----------
    model: subclass of torch.nn.Module, bp, cnn network model
    dataset: subclass of torch.utils.data.Dataset: VectorDataset, PatchDataset
    state: str, one of 'train', 'val' and 'test'
    prop: float, e.g. 0.2, random sample 20% data from test data for test to
    avoid out of memory

    Returns
    -------
    pred: ndarray, N*1
    proba: ndarray, probability array, N*C, C is number of classes
    accuracy: float, test accuracy

    """

    state_ = dataset.state
    data, target = dataset.data, dataset.target

    if state is not None:
        assert state in ['train', 'val', 'test']
        dataset.set_state(state)
        data, target = dataset.data, dataset.target

    if prop < 1:
        num = int(round(len(data) * prop))
        state = np.random.get_state()
        data = data[np.random.choice(len(data), num), :, :, :]
        np.random.set_state(state)
        target = target[np.random.choice(len(target), num)]
        dataset.data, dataset.target = data, target

    model.eval()
    try:
        with torch.no_grad():
            output = model(dataset.data.cuda()).cpu().data
            torch.cuda.empty_cache()
    except:
        output = None
        dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
        for idx, batch_data in enumerate(dataloader):
            with torch.no_grad():
                batch_output = model(batch_data[0].cuda()).cpu().data
            if idx == 0:
                output = batch_output
            else:
                output = torch.cat((output, batch_output), dim=0)
            torch.cuda.empty_cache()
    pred = torch.max(output, dim=1)[1].numpy()
    target = dataset.target.cpu().numpy()
    accuracy = compute_accuracy(pred, target)
    dataset.set_state(state_)

    proba = F.softmax(output, dim=1).numpy()

    return pred, proba, accuracy


def test_fcnn(model, dataset):
    pass


