"""
Going Deeper with Contextual CNN for Hyperspectral Image Classification
Lee Hyungtae，Kwon Heesung
IEEE Transactions on Image Processing 2017
"""

import os
import copy
import time
import pdb

import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from config import *


class DeepFCN(nn.Module):
    """
    Going Deeper with Contextual CNN for Hyperspectral Image Classification
    """

    def __init__(self, in_channels, n_classes):

        super(DeepFCN, self).__init__()

        self.conv1_5 = nn.Conv2d(in_channels, 128, (5, 5), 1, (2, 2))
        self.conv1_3 = nn.Conv2d(in_channels, 128, (3, 3), 1, (1, 1))
        self.conv1_1 = nn.Conv2d(in_channels, 128, (1, 1), 1)
        self.relu1 = nn.ReLU(True)
        self.lrn1 = nn.LocalResponseNorm(384)

        self.conv2 = nn.Conv2d(384, 128, (1, 1), 1)
        self.relu2 = nn.ReLU(True)
        self.lrn2 = nn.LocalResponseNorm(128)

        self.conv3 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu5 = nn.ReLU(True)
        self.conv6 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu7 = nn.ReLU(True)
        self.drop7 = nn.Dropout(0.5)
        self.conv8 = nn.Conv2d(128, 128, (1, 1), 1)
        self.relu8 = nn.ReLU(True)
        self.drop8 = nn.Dropout(0.5)
        self.conv9 = nn.Conv2d(128, n_classes, (1, 1), 1)

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):

        # depth concat
        x_5 = self.conv1_5(x)
        x_3 = self.conv1_3(x)
        x_1 = self.conv1_1(x)
        x = torch.cat((x_5, x_3, x_1), dim=1)
        x = self.lrn1(self.relu1(x))

        # residual block1
        x = self.lrn2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(x + self.conv4(x))

        # residual block2
        x = self.relu5(self.conv5(x))
        x = self.relu6(x + self.conv6(x))

        # rest conv layers
        x = self.drop7(self.relu7(self.conv7(x)))
        x = self.drop8(self.relu8(self.conv8(x)))
        x = self.conv9(x)

        return x


class CommonDataset(Dataset):
    """dataset for training/validating 'Deep-FCN'
    Args:
        data: np.ndarray, (N, C, P, P)
        taget: np.bdarray, (N, P, P)
        cuda: bool, if concert to cuda, default is False
        augment: bool, if augment data, default is False

    """

    def __init__(self, data, target, cuda=False, augment=False):

        if augment is True:
            data, target = self._mirror_augment(data, target)

        # data = np.expand_dims(data, axis=1)
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()

        if cuda is True:
            self.data = self.data.cuda()
            self.target = self.target.cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    @staticmethod
    def _mirror_augment(data, target):

        hm_data = data.copy()  # horizontal
        hm_target = target.copy()
        vm_data = data.copy()  # vertical
        vm_target = target.copy()
        dm_data = data.copy()  # diagonal
        dm_target = target.copy()

        height, width = data.shape[-2:]
        hm_center = int(np.ceil(width / 2))
        vm_center = int(np.ceil(height / 2))

        for i in range(hm_center):
            tmp = hm_data[:, :, :, i].copy()
            hm_data[:, :, :, i] = hm_data[:, :, :, (width - 1) - i]
            hm_data[:, :, :, (width - 1) - i] = tmp
            tmp = hm_target[:, :, i].copy()
            hm_target[:, :, i] = hm_target[:, :, (width - 1) - i]
            hm_target[:, :, (width - 1) - i] = tmp

        for i in range(vm_center):
            tmp = vm_data[:, :, i, :].copy()
            vm_data[:, :, i, :] = vm_data[:, :, (width - 1) - i, :]
            vm_data[:, :, (width - 1) - i, :] = tmp
            tmp = vm_target[:, i, :].copy()
            vm_target[:, i, :] = vm_target[:, (width - 1) - i, :]
            vm_target[:, (width - 1) - i, :] = tmp

        for i in range(height):
            for j in range(i + 1):
                tmp = dm_data[:, :, i, j].copy()
                dm_data[:, :, i, j] = dm_data[:, :, j, i]
                dm_data[:, :, j, i] = tmp
                tmp = dm_target[:, i, j].copy()
                dm_target[:, i, j] = dm_target[:, j, i]
                dm_target[:, j, i] = tmp

        data = np.concatenate((data, hm_data, vm_data, dm_data), axis=0)
        target = np.concatenate((target, hm_target, vm_target, dm_target),
                                axis=0)

        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(target)

        return data, target


class PredictDataset(Dataset):
    """dataset for prediction
    Args:
        data: np.ndarray, (N, C, H, W)
        cuda: bool, if convert to cuda, default is False

    """

    def __init__(self, data, cuda=False):
        self.data = torch.from_numpy(data).float()

        if cuda is True:
            self.data = self.data.cuda()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_samples(raw_data, raw_target, mask, patch_size):
    """get sample pairs, raw_data(C, H, W), raw_target(H, W)

    """

    raw_data = raw_data.copy()
    raw_target = raw_target.copy()

    width = patch_size // 2
    raw_data = np.pad(raw_data, ((0, 0), (width, width), (width, width)),
                      'constant')
    raw_target = np.pad(raw_target, ((width, width), (width, width)),
                        'constant')
    mask = np.pad(mask, ((width, width), (width, width)), 'constant')

    idx = np.argwhere(mask == 1)
    data = np.zeros((len(idx), raw_data.shape[0], patch_size, patch_size))
    target = np.zeros((len(idx), patch_size, patch_size))

    for i, idx_i in enumerate(idx):
        h_s = slice(idx_i[0] - width, idx_i[0] + width + 1)
        w_s = slice(idx_i[1] - width, idx_i[1] + width + 1)
        data[i] = raw_data[:, h_s, w_s]
        target[i] = raw_target[h_s, w_s]

    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

    return data, target


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


def train(model, train_dataset, val_dataset, params):
    """train 'Deep-FCN'
    Args:
        model: DeepFCN object, 'Deep-FCN' network
        train_dataset: CommonDataset object, training dataset
        val_dataset: CommonDataset object, validation dataset
        params: dict, training parameters

    """

    # training parameters
    epoch = params['epoch']
    batch_size = params['batch_size']
    if 'criterion' in params.keys():
        if 'weight' in params['criterion_params']:
            params['criterion_params']['weight'] = \
                torch.Tensor(params['criterion_params']['weight']).float().\
                    cuda()
        criterion = _criterion[params['criterion']] \
            (**params['criterion_params'])
    else:
        criterion = nn.CrossEntropyLoss()
    if 'optimizer' in params.keys():
        optimizer = _optimizer[params['optimizer']] \
            (model.parameters(), **params['optimizer_params'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if 'scheduler' in params.keys():
        scheduler = _scheduler[params['scheduler']] \
            (optimizer, **params['scheduler_params'])
    else:
        scheduler = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)
    model.train()
    model = model.cuda()
    summary(model, train_dataset.data.shape[1:])

    best_epoch = 0
    best_loss = 0.
    best_train = 0.
    best_val = 0.
    best_model = None
    loss_list = []
    train_acc_list = []
    val_acc_list = []

    for e in range(epoch):

        loss_ = 0.
        train_acc_ = 0.
        val_acc_ = 0.
        i = 0

        for i, (data, target) in enumerate(train_loader):
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, train_acc = test(model, train_dataset)
            _, val_acc = test(model, val_dataset, batch_size=3000)
            # for PU and SA, set a smaller validation batch_size

            if val_acc > best_val:
                best_val = val_acc
                best_train = train_acc
                best_model = copy.deepcopy(model)
                best_loss = loss.item()
                best_epoch = e + 1

            loss_ += loss.item()
            train_acc_ += train_acc
            val_acc_ += val_acc

            # print('epoch: %-4d | batch: %-4d | loss: %-8.6f | train acc: '
            #       '%-8.6f | val acc: %-8.6f' %
            #       (e + 1, i + 1, loss.item(), train_acc, val_acc))
            print('epoch: %-4d | batch: %-4d | loss: %-8.6f | train acc: '
                  '%-8.6f | val acc: %-8.6f' %
                  (e + 1, i + 1, loss.item(), train_acc, val_acc), '\r', end='')

        loss_list.append(loss_ / (i + 1))
        train_acc_list.append(train_acc_ / (i + 1))
        val_acc_list.append(val_acc_ / (i + 1))

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-(val_acc_ / (i + 1)))
        elif scheduler is not None:
            scheduler.step()

    best_records = {'epoch': best_epoch, 'loss': best_loss,
                    'train_acc': best_train, 'val_acc': best_val}

    output_dict = {
        'model': best_model,
        'best': best_records,
        'loss': loss_list,
        'train': train_acc_list,
        'val': val_acc_list
    }

    return output_dict


def test(model, dataset, batch_size=2000):
    """perform test and compute test accuracy
    Args:
        model: DeepFCN object
        dataset: CommonDataset object

    """

    i, j = (int(e / 2) for e in dataset.data.shape[-2:])

    model.eval()
    model = model.cuda()

    try:
        pred = model(dataset.data.cuda()).cpu().data.numpy()
        torch.cuda.empty_cache()
    except RuntimeError:
        pred = None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for idx, (data, _) in enumerate(dataloader):
            pred_ = model(data.cuda()).cpu().data.numpy()
            torch.cuda.empty_cache()
            if idx == 0:
                pred = pred_
            else:
                pred = np.concatenate((pred, pred_), axis=0)
    # pdb.set_trace()
    pred = np.argmax(pred, axis=1)
    pred = pred[:, i, j]
    target = dataset.target[:, i, j].cpu().data.numpy()
    accuracy = float((pred == target).sum()) / len(target)

    return pred, accuracy


def predict(model, data):
    """predict the full HSI data
   Args:
       model: DeepFCN object
       data: np.ndarray, (C, H, W)

    """

    data = np.expand_dims(data, axis=0)
    dataset = PredictDataset(data, cuda=True)

    model.eval()
    model = model.cuda()

    pred = model(dataset.data).cpu().data
    proba = F.softmax(pred, dim=1).numpy()
    pred = pred.numpy()
    torch.cuda.empty_cache()

    pred = np.argmax(pred, axis=1)
    pred = pred.squeeze(axis=0)
    proba = proba.squeeze(axis=0)

    return pred, proba


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
    mask_dir = os.path.dirname(data_dir)
    data, target = read_data(data_dir, target_dir)
    train_mask, val_mask, test_mask = load_masks(mask_dir, target, train_prop,
                                                 val_prop)
    train_data, train_target = get_samples(data, target, train_mask, patch_size)
    val_data, val_target = get_samples(data, target, val_mask, patch_size)

    train_dataset = CommonDataset(train_data, train_target, cuda=False,
                                  augment=False)
    val_dataset = CommonDataset(val_data, val_target, cuda=False, augment=False)

    logger.info('initialize model ...')
    model = DeepFCN(**model_params)

    #  ############################# training ##############################

    logger.info('begin to train ...')
    s = time.time()
    output = train(model, train_dataset, val_dataset, train_params)

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
    logger.info('begin to predict ...')
    s = time.time()
    pred, proba = predict(model_, data)
    e = time.time()
    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    logger.info('save soft label ...')
    sio.savemat(soft_dir, {'soft_label': proba})

    # output predicted map(png/mat), accuracy table and other records
    logger.info('save classification maps etc. ...')
    train_records = {
        'best': output['best'],
        'train_time': '%.4f' % train_time,
        'train_time_mean': '%.4f' % train_time_mean,
        'pred_time': '%.4f' % pred_time
    }
    ro = ResultOutput(pred, data, target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()

