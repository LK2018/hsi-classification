"""
A semi-supervised Convolutional Neural Network for Hyperspectral Image
Classification
Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
Remote Sensing Letters 2017
"""

import os
import copy
import time

import numpy as np
import scipy.io as sio
import torch
from torch import nn, optim
from torch.nn import init
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from config import *


class Semi2DCNN(nn.Module):

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=9):
        super(Semi2DCNN, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for
        # the convolutional "and pooling layer" -> actually 3x3 2D convolutions
        # with B1 outputs "the value of B1 is set to be 80"
        self.conv1 = nn.Conv2d(input_channels, 80, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv1_bn = nn.BatchNorm2d(80)

        self.features_sizes = self._get_sizes()

        self.fc_enc = nn.Linear(self.features_sizes[2], n_classes)

        # Decoder
        self.fc1_dec = nn.Linear(self.features_sizes[2], self.features_sizes[2])
        self.fc1_dec_bn = nn.BatchNorm1d(self.features_sizes[2])
        self.fc2_dec = nn.Linear(self.features_sizes[2], self.features_sizes[1])
        self.fc2_dec_bn = nn.BatchNorm1d(self.features_sizes[1])
        self.fc3_dec = nn.Linear(self.features_sizes[1], self.features_sizes[0])
        self.fc3_dec_bn = nn.BatchNorm1d(self.features_sizes[0])
        self.fc4_dec = nn.Linear(self.features_sizes[0], input_channels)

        self.apply(self.weight_init)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels,
                         self.patch_size, self.patch_size))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        _, c, w, h = x.size()
        size0 = c * w * h

        x = self.pool1(x)
        _, c, w, h = x.size()
        size1 = c * w * h

        x = self.conv1_bn(x)
        _, c, w, h = x.size()
        size2 = c * w * h

        return size0, size1, size2

    def forward(self, x):
        # x = x.squeeze()
        x_conv1 = self.conv1_bn(self.conv1(x))
        x = x_conv1
        x_pool1 = self.pool1(x)
        x = x_pool1
        x_enc = F.relu(x).view(-1, self.features_sizes[2])
        x = x_enc

        x_classif = self.fc_enc(x)

        # x = F.relu(self.fc1_dec_bn(self.fc1_dec(x) + x_enc))
        x = F.relu(self.fc1_dec(x))
        x = F.relu(self.fc2_dec_bn(
            self.fc2_dec(x) + x_pool1.view(-1, self.features_sizes[1])))
        x = F.relu(self.fc3_dec_bn(
            self.fc3_dec(x) + x_conv1.view(-1, self.features_sizes[0])))
        x = self.fc4_dec(x)

        return x_classif, x


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        # self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        # self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.flip_augmentation = hyperparams['flip_augmentation']
        # self.radiation_augmentation = hyperparams['radiation_augmentation']
        # self.mixture_augmentation = hyperparams['mixture_augmentation']
        # self.center_pixel = hyperparams['center_pixel']
        self.flip_augmentation = False
        self.radiation_augmentation = False
        self.mixture_augmentation = False
        # self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = True
        # supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        # if supervision == 'full':
        #     mask = np.ones_like(gt)
        #     for l in self.ignored_labels:
        #         mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        # elif supervision == 'semi':
        mask = np.ones_like(gt)

        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)

        return data, label


def train(net, data_loader, val_loader, lr=0.001, patch_size=9, epoch=2000):

    optimizer = optim.SGD(net.parameters(), lr=lr)
    criterion = (nn.CrossEntropyLoss(), lambda rec, data:
    F.mse_loss(rec, data[:, :, patch_size // 2, patch_size // 2]))

    net.train()

    loss_list = []
    train_list = []
    val_list = []
    best_val = 0.
    best_model = None
    best_record = {}

    for e in range(epoch):

        loss_i = 0.
        train_i = 0.
        val_i = 0.
        i = 0

        for i, (data, target) in enumerate(data_loader):

            optimizer.zero_grad()
            output, rec = net(data)
            loss = criterion[0](output, target) + \
                   net.aux_loss_weight * criterion[1](rec, data)
            loss.backward()
            optimizer.step()

            train_acc = val(net, data_loader)
            val_acc = val(net, val_loader)

            print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
                  'val: %.6f ' % (e + 1, i + 1, loss.item(), train_acc,
                                  val_acc), '\r', end='')

            if val_acc > best_val:
                best_train = train_acc
                best_val = val_acc
                best_loss = loss.item()
                best_record = {'epoch': e + 1, 'batch': i + 1, 'loss':
                    best_loss, 'train': best_train, 'val': best_val}
                best_model = copy.deepcopy(net)

                train_i += train_acc
                val_i += val_acc
                loss_i += loss.item()

        loss_list.append(loss_i / (i + 1))
        train_list.append(train_i / (i + 1))
        val_list.append(val_i / (i + 1))

        print('%s best record %s' % ('-' * 40, '-' * 40))
        print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
              'val: %.6f ' % (best_record['epoch'], best_record['batch'],
                              best_record['loss'], best_record['train'],
                              best_record['val']))

        output_dict = {
            'model': best_model,
            'loss': np.array(loss_list),
            'train': np.array(train_list),
            'val': np.array(val_list),
            'best': best_record
        }

        return output_dict


def val(net, data_loader):

    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            output, rec = net(data)
            _, output = torch.max(output, dim=1)
            #target = target - 1
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


# _criterion = {
#     'CrossEntropy': nn.CrossEntropyLoss
# }
# _optimizer = {
#     'Adam': optim.Adam,
#     'SGD': optim.SGD
# }
# _scheduler = {
#     'StepLR': lr_scheduler.StepLR,
#     'MultiStepLR': lr_scheduler.MultiStepLR
# }
#
#
# def train(model, dataset, params):
#
#     epoch = params['epoch']
#     batch_size = params['batch_size']
#     if 'criterion' in params.keys():
#         if 'weight' in params['criterion_params']:
#             params['criterion_params']['weight'] = \
#                 torch.Tensor(params['criterion_params']['weight']).float(). \
#                     cuda()
#         criterion = _criterion[params['criterion']] \
#             (**params['criterion_params'])
#     else:
#         criterion = nn.CrossEntropyLoss()
#     if 'optimizer' in params.keys():
#         optimizer = _optimizer[params['optimizer']]\
#             (model.parameters(), **params['optimizer_params'])
#     else:
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#     if 'scheduler' in params.keys():
#         scheduler = _scheduler[params['scheduler']]\
#             (optimizer, **params['scheduler_params'])
#     else:
#         scheduler = None
#
#     patch_size = params['patch_size']
#     criterion = (criterion, lambda rec, data:
#     F.mse_loss(rec, data[:, :, patch_size // 2, patch_size // 2]))
#
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     best_val = 0.
#     best_record = {}
#     best_model = None
#     loss_list = []
#     train_list = []
#     val_list = []
#
#     # ######################## begin to train ################################
#     model.train()
#     model = model.cuda()
#     summary(model, input_size=dataset.data.shape[1:])
#
#     for e in range(epoch):
#
#         loss_i = 0.
#         train_i = 0.
#         val_i = 0.
#         i = 0
#
#         for i, (data, target) in enumerate(dataloader):
#
#             data = data.cuda()
#             target = target.cuda()
#             output, rec = model(data)
#             loss = criterion[0](output, target) + \
#                    model.aux_loss_weight * criterion[1](rec, data)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             _, _, train_acc = test(model, dataset, state='train')
#             _, _, val_acc = test(model, dataset, state='val')
#             torch.cuda.empty_cache()
#
#             train_i += train_acc
#             val_i += val_acc
#             loss_i += loss.item()
#
#             if val_acc > best_val:
#                 best_train = train_acc
#                 best_val = val_acc
#                 best_loss = loss.item()
#                 best_record = {'epoch': e + 1, 'batch': i + 1, 'loss':
#                     best_loss, 'train': best_train, 'val': best_val}
#                 best_model = copy.deepcopy(model)
#
#             print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
#                   'val: %.6f ' % (e + 1, i + 1, loss.item(), train_acc,
#                                   val_acc), '\r', end='')
#
#         # if set scheduler, update it
#         if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(-val_i)
#         elif scheduler is not None:
#             scheduler.step()
#
#         loss_list.append(loss_i / (i + 1))
#         train_list.append(train_i / (i + 1))
#         val_list.append(val_i / (i + 1))
#
#     print('%s best record %s' % ('-' * 40, '-' * 40))
#     print('epoch: %5s, batch: %3s | loss: %13.6f | train: %.6f | '
#           'val: %.6f ' % (best_record['epoch'], best_record['batch'],
#                           best_record['loss'], best_record['train'],
#                           best_record['val']))
#
#     # ############################# output results ###########################
#     output_dict = {
#         'model': best_model,
#         'loss': np.array(loss_list),
#         'train': np.array(train_list),
#         'val': np.array(val_list),
#         'best': best_record
#     }
#
#     return output_dict
#
#
# def test(model, dataset, state=None):
#
#     state_ = dataset.state
#
#     if state is not None:
#         assert state in ['train', 'val', 'test']
#         dataset.set_state(state)
#
#     model.eval()
#     try:
#         with torch.no_grad():
#             output, rec = model(dataset.data.cuda()).cpu().data
#             torch.cuda.empty_cache()
#     except:
#         output = None
#         dataloader = DataLoader(dataset, batch_size=2000, shuffle=False)
#         for idx, batch_data in enumerate(dataloader):
#             with torch.no_grad():
#                 batch_output, rec = model(batch_data[0].cuda())
#                 batch_output = batch_output.cpu().data
#             if idx == 0:
#                 output = batch_output
#             else:
#                 output = torch.cat((output, batch_output), dim=0)
#             torch.cuda.empty_cache()
#     pred = torch.max(output, dim=1)[1].numpy()
#     target = dataset.target.cpu().numpy()
#     accuracy = float((pred == target).sum()) / float(len(target))
#     dataset.set_state(state_)
#
#     proba = F.softmax(output, dim=1).numpy()
#
#     return pred, proba, accuracy


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
    dataset.set_state('train')

    logger.info('initialize model ...')
    model = Semi2DCNN(**model_params)

    #  ############################# training ##############################

    logger.info('begin to train ...')
    s = time.time()
    output = train(model, dataset, train_params)

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
    all_patch = torch.from_numpy(all_patch).float()
    all_target = raw_target.ravel()
    all_target = torch.from_numpy(all_target).long()
    dataset.data, dataset.target = all_patch, all_target

    logger.info('begin to predict ...')
    s = time.time()
    pred, proba, _ = test(model_, dataset)
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

