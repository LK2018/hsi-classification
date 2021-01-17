"""
Deep Learning-Based Classification of Hyperspectral Data
Chen, Yushi and Lin, Zhouhan and Zhao, Xing and Wang, Gang and Gu, Yanfeng
IEEE Journal of Selected topics in applied earth observations and remote sensing
2014
"""

import os
from copy import deepcopy
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import scipy.io as sio

from config import *


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_units):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_units, bias=True)
        self.decoder = nn.Linear(hidden_units, input_dim, bias=True)
        self.act = nn.ReLU()
        # self.act = nn.Sigmoid()

    def forward(self, x, rep=False):

        hidden = self.encoder(x)
        hidden = self.act(hidden)
        if rep == False:
            out = self.decoder(hidden)
            #out = self.act(out)
            return out
        else:
            return hidden


class SAE(nn.Module):

    def __init__(self, encoder_list, hidden_units, n_classes):

        super().__init__()

        self.encoderList = encoder_list
        self.en1 = encoder_list[0]
        self.en2 = encoder_list[1]
        self.en3 = encoder_list[2]

        self.fc = nn.Linear(hidden_units, n_classes, bias=True)

    def forward(self, x):

        out = x
        out = self.en1(out, rep=True)
        out = self.en2(out, rep=True)
        out = self.en3(out, rep=True)
        out = self.fc(out)
        # out = F.log_softmax(out)

        return out


########################################################################


def train_ae(encoder_list, train_layer, dataset, batch_size, epoch, lr):

    print('%s train %sth encoder %s' % ('-' * 40, train_layer + 1, '-' * 40))

    for i in range(len(encoder_list)):
        encoder_list[i].cuda()

    # optimizer = optim.SGD(encoder_list[train_layer].parameters(), lr=lr)
    optimizer = optim.Adam(encoder_list[train_layer].parameters(), lr=lr)
    criterion = nn.L1Loss()
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    min_val_loss = 1e5
    best_model = None
    best_records = None

    for e in range(epoch):

        if train_layer != 0:
            for i in range(train_layer):
                for param in encoder_list[i].parameters():
                    param.requires_grad = False

        for batch_idx, (x, _) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            out = x
            if train_layer != 0:
                for i in range(train_layer):
                    out = encoder_list[i](out, rep=True)

            pred = encoder_list[train_layer](out, rep=False)

            loss = criterion(pred, out)
            loss.backward()
            optimizer.step()

            dataset.set_state('val')
            val_loss = val_ae(encoder_list, train_layer, dataset)
            dataset.set_state('train')

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = deepcopy(encoder_list[train_layer])
                best_records = {'epoch': e + 1, 'batch': batch_idx + 1,
                                'train_loss': loss.item(),
                                'val_loss': val_loss.item()}

            print('epoch: %5s  batch: %3s  train_loss: %10.6f  val_loss: '
                  '%10.6f' % (e + 1, batch_idx + 1, loss.item(),
                              val_loss.item()), '\r', end='')

    print('\r')
    print('best records: ')
    print('epoch: %5s  batch: %3s  train_loss: %10.6f  val_loss: %10.6f' %
          (best_records['epoch'], best_records['batch'],
           best_records['train_loss'], best_records['val_loss']))

    return best_model, best_records


def val_ae(encoder_list, train_layer, dataset):

    for i in range(len(encoder_list)):
        encoder_list[i].cuda()

    criterion = nn.L1Loss()
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.cuda()
            out = x
            if train_layer != 0:
                for i in range(train_layer):
                    out = encoder_list[i](out, rep=True)

            pred = encoder_list[train_layer](out, rep=False)

            loss = criterion(pred, out)

    return loss


def train_sae(model, dataset, batch_size, epoch, lr):

    print('%s fine-tune the SAE %s ' % ('-' * 40, '-' * 40))

    model.cuda()
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ceriation = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    best_val = 0
    best_model = None
    best_records = None

    for e in range(epoch):

        for batch_idx, (x, target) in enumerate(dataloader):
            optimizer.zero_grad()
            x, target = x.cuda(), target.cuda()

            out = model(x)

            loss = ceriation(out, target)
            loss.backward()
            optimizer.step()

            _, _, train_acc = val_sae(model, dataset)
            dataset.set_state('val')
            _, _, val_acc = val_sae(model, dataset)
            dataset.set_state('train')

            if val_acc > best_val:
                best_val = val_acc
                best_model = deepcopy(model)
                best_records = {'epoch': e + 1, 'batch': batch_idx + 1,
                                'loss': loss.item(), 'train_acc': train_acc,
                                'val_acc': val_acc}

            print('epoch: %5s  batch: %3s  loss: %10.6f  train_acc: %.6f  '
                  'val_acc: %.6f' % (e + 1, batch_idx + 1, loss.item(),
                                     train_acc, val_acc), '\r', end='')

    print('\r')
    print('best records')
    print('epoch: %5s  batch: %3s  loss: %10.6f  train_acc: %.6f  '
          'val_acc: %.6f' % (best_records['epoch'], best_records['batch'],
                             best_records['loss'], best_records['train_acc'],
                             best_records['val_acc']))

    return best_model, best_records


def val_sae(model, dataset):

    model.eval()
    try:
        with torch.no_grad():
            output = model(dataset.data.cuda()).cpu().data
            torch.cuda.empty_cache()
    except:
        output = None
        dataloader = DataLoader(dataset, batch_size=2000, shuffle=False)
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
    accuracy = float((pred == target).sum()) / float(len(target))

    proba = F.softmax(output, dim=1).numpy()

    return pred, proba, accuracy


def run(params):

    # ##################### get parameters ###################################

    # device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)

    # get parameters
    data_name = params.data.data_name
    data_dir = params.data.data_dir
    target_dir = params.data.target_dir
    train_prop = params.data.train_prop
    val_prop = params.data.val_prop

    encoder_num = params.model.encoder_num
    input_dim = params.model.input_dim
    hidden_units = params.model.hidden_units
    n_classes = params.model.n_classes

    ae_epoch = params.train.ae_epoch
    sae_epoch = params.train.sae_epoch
    ae_lr = params.train.ae_lr
    sae_lr = params.train.sae_lr
    ae_batch = params.train.ae_batch
    sae_batch = params.train.sae_batch

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

    # ########################### dataset, model #############################

    logger.info('prepare dataset ...')
    dataset = VectorDataset(data_dir, target_dir, train_prop, val_prop,
                            state='train')

    logger.info('begin to train ...')
    s = time.time()
    encoder_list = []
    encoder_trained = []
    for i in range(encoder_num):
        if i == 0:
            encoder_list += [AutoEncoder(input_dim, hidden_units)]
        else:
            encoder_list += [AutoEncoder(hidden_units, hidden_units)]
        encoder_i, _ = train_ae(encoder_list, i, dataset, ae_batch, ae_epoch,
                             ae_lr)
        encoder_trained += [encoder_i]

    model = SAE(encoder_trained, hidden_units, n_classes)
    model, _ = train_sae(model, dataset, sae_batch, sae_epoch, sae_lr)
    e = time.time()
    train_time = e - s
    logger.info('training time: %.4fs' % train_time)

    logger.info('save model ...')
    torch.save(model.state_dict(), model_dir)

    # ########################## predict #####################################

    raw_data, raw_target = dataset.raw_data, dataset.raw_target
    all_data = raw_data.reshape(-1, raw_data.shape[1] * raw_data.shape[2]).T
    all_target = raw_target.ravel()
    all_data = torch.from_numpy(all_data).float().cuda()
    all_target = torch.from_numpy(all_target).long().cuda()
    dataset.data, dataset.target = all_data, all_target

    logger.info('begin to predict ...')
    s = time.time()
    pred, proba, _ = val_sae(model, dataset)
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
        'train_time': '%.4f' % train_time,
        'pred_time': '%.4f' % pred_time
    }
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    ro = ResultOutput(pred, raw_data, raw_target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()

