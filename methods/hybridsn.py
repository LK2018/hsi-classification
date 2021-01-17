"""
HybridSN: Exploring 3-D--2-D CNN feature hierarchy for hyperspectral image
classification
Roy, Swalpa Kumar and Krishna, Gopal and Dubey, Shiv Ram and Chaudhuri, Bidyut B
IEEE Geoscience and Remote Sensing Letters
2019
"""

import os
import time

from keras.layers import Conv2D, Conv3D, Flatten, Dense, Reshape, \
    BatchNormalization
from keras.layers import Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report, cohen_kappa_score

import numpy as np
import scipy.io as sio

from config import *


def loadData(data_dir, target_dir):

    data = sio.loadmat(data_dir)
    data = list(filter(lambda x: isinstance(x, np.ndarray), data.values()))[0]

    target = sio.loadmat(target_dir)
    target = list(filter(lambda x: isinstance(x, np.ndarray), target.values()))[
        0]

    return data, target


def splitTrainTestSet(X, y, testRatio, randomState=345):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


def applyPCA(X, numComponents=75):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX, pca


def padWithZeros(X, margin=2):

    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


def createImageCubes(X, y, windowSize, train_mask, val_mask, test_mask):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1,
                    c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1

    train_idx = np.where(train_mask.ravel() == 1)[0]
    val_idx = np.where(val_mask.ravel() == 1)[0]
    test_idx = np.where(test_mask.ravel() == 1)[0]

    X_train = patchesData[train_idx, :, :, :]
    X_val = patchesData[val_idx, :, :, :]
    X_test = patchesData[test_idx, :, :, :]
    y_train = patchesLabels[train_idx] - 1
    y_val = patchesLabels[val_idx] - 1
    y_test = patchesLabels[test_idx] - 1

    return X_train, X_val, X_test, y_train, y_val, y_test


def createImageCubes_all(X, y, windowSize, removeZeroLabels=False):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize,
                            X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1,
                    c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def get_model(input_shape, output_units):

    ## input layer
    # input_layer = Input((S, S, L, 1))
    input_layer = Input(input_shape)

    ## convolutional layers
    conv_layer1 = Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')(
        input_layer)
    conv_layer2 = Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')(
        conv_layer1)
    conv_layer3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(
        conv_layer2)
    print(conv_layer3._keras_shape)
    conv3d_shape = conv_layer3._keras_shape
    conv_layer3 = Reshape(
        (conv3d_shape[1], conv3d_shape[2], conv3d_shape[3] * conv3d_shape[4]))(
        conv_layer3)
    conv_layer4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(
        conv_layer3)

    flatten_layer = Flatten()(conv_layer4)

    ## fully connected layers
    dense_layer1 = Dense(units=256, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units, activation='softmax')(dense_layer2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


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
    windowSize = params.data.patch_size
    K = params.data.n_pc

    batch_size = params.train.batch_size
    epoch = params.train.epoch
    lr = params.train.lr
    decay = params.train.decay

    output_units = params.model.output_units

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
    filepath = os.path.join(result_dir, 'weights.hdf5')
    soft_dir = os.path.join(result_dir, 'soft_label.mat')

    # define logger
    logger = define_logger(log_dir)

    # print parameters
    num1 = 25
    num2 = 100
    logger.info('%s begin a new training: %s %s' % ('#' * num1, method_name,
                                                    '#' * num1))
    params_str = recur_str_dict_for_show(params, total_space=num2)
    logger.info('show parameters ... \n%s' % params_str)

    # ############################### dataset ################################

    logger.info('prepare train dataset and val dataset ...')
    X, y = loadData(data_dir, target_dir)
    X, pca = applyPCA(X, numComponents=K)

    mask_dir = os.path.join(os.path.dirname(data_dir),
                            'masks_%s_%s' % (train_prop, val_prop))
    train_mask = sio.loadmat(os.path.join(mask_dir, 'train_mask.mat'))[
        'train_mask']
    val_mask = sio.loadmat(os.path.join(mask_dir, 'val_mask.mat'))['val_mask']
    test_mask = sio.loadmat(os.path.join(mask_dir, 'test_mask'))['test_mask']

    Xtrain, Xval, Xtest, ytrain, yval, ytest = \
        createImageCubes(X, y, windowSize, train_mask, val_mask, test_mask)

    Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    Xval = Xval.reshape(-1, windowSize, windowSize, K, 1)

    ytrain = np_utils.to_categorical(ytrain)
    yval = np_utils.to_categorical(yval)
    logger.info('train data: %s, val data: %s' % (Xtrain.shape, Xval.shape))

    # ############################ model and train ############################

    logger.info('initialize model ...')
    input_shape = (windowSize, windowSize, K, 1)
    model = get_model(input_shape, output_units)
    model.summary()

    # compiling the model
    # adam = Adam(lr=0.001, decay=1e-06)
    adam = Adam(lr=lr, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    # checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    logger.info('begin to train ...')
    s = time.time()
    history = model.fit(x=Xtrain, y=ytrain, batch_size=batch_size, epochs=epoch,
                        validation_data=(Xval, yval), callbacks=callbacks_list)
    e = time.time()

    train_time = e - s
    train_time_mean = train_time /epoch
    logger.info('training time: %.4fs' % train_time)
    logger.info('training time mean: %.4fs' % train_time_mean)

    # ############################ predict ####################################

    logger.info('begin to predict ...')

    # load best weights
    model.load_weights(filepath)
    model.compile(loss='categorical_crossentropy', optimizer=adam,
                  metrics=['accuracy'])

    Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    ytest = np_utils.to_categorical(ytest)

    Xall, yall = createImageCubes_all(X, y, windowSize)
    Xall = Xall.reshape(-1, windowSize, windowSize, K, 1)
    yall = np_utils.to_categorical(yall)

    Y_pred_test = model.predict(Xtest)
    y_pred_test = np.argmax(Y_pred_test, axis=1)
    classification = classification_report(np.argmax(ytest, axis=1),
                                           y_pred_test)
    print(classification)

    s = time.time()
    Y_pred_all = model.predict(Xall)
    e = time.time()
    soft_label = Y_pred_all.reshape(y.shape + (-1,))
    y_pred_all = np.argmax(Y_pred_all, axis=1)
    y_pred_all = y_pred_all.reshape(y.shape) + 1

    pred_time = (e - s)
    logger.info('predicted time: %.4fs' % pred_time)

    logger.info('save soft label ...')
    sio.savemat(soft_dir, {'soft_label': soft_label})

    # output predicted map(png/mat), accuracy table and other records
    logger.info('save classification maps etc. ...')
    train_records = {
        'train_time': '%.4f' % train_time,
        'train_time_mean': '%.4f' % train_time_mean,
        'pred_time': '%.4f' % pred_time
    }

    raw_data, raw_target = read_data(data_dir, target_dir)

    ro = ResultOutput(y_pred_all, raw_data, raw_target, train_mask, val_mask,
                      test_mask, result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()

