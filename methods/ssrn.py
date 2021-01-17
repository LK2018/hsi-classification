# -*- coding utf-8 -*-

"""
Spectral--spatial residual network for hyperspectral image classification:
A 3-D deep learning framework
Zhong, Zilong and Li, Jonathan and Luo, Zhiming and Chapman, Michael
IEEE Transactions on Geoscience and Remote Sensing
2017
"""

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    Conv3D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers.core import Reshape
from keras import regularizers
from keras.layers.merge import add

def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def _bn_relu_spc(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu_spc(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))
    def f(input):
        # conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, W_regularizer=W_regularizer)(input)
        conv = Conv3D(kernel_initializer=init,strides=subsample,kernel_regularizer= W_regularizer, filters=nb_filter, kernel_size=(kernel_dim1,kernel_dim2,kernel_dim3))(input)
        # conv = Conv3D(kernel_initializer="he_normal", strides=(1,1,2), kernel_regularizer=regularizers.l2(1.e-4), filters=32,
        #               kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))
        return _bn_relu_spc(conv)

    return f


def _bn_relu_conv_spc(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        return Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f


def _shortcut_spc(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = 1
    stride_dim2 = 1
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid",
                                 W_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            # conv1 = Convolution3D(nb_filter=nb_filter,
            #                       kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
            #                      subsample=init_subsample,
            #                      init="he_normal", border_mode="same",
            #                      W_regularizer=l2(0.0001))(input)
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(1, 1, 7), padding='same')(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=init_subsample)(input)

        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
        return _shortcut_spc(input, residual)

    return f


def bottleneck_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution3D(nb_filter=nb_filter,
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv_spc(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut_spc(input, residual)

    return f


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        # conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, W_regularizer=W_regularizer)(input)
        conv = Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3))(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1,1,1))
    init = conv_params.setdefault("init", "he_normal")
    border_mode = conv_params.setdefault("border_mode", "same")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        # return Convolution3D(nb_filter=nb_filter, kernel_dim1=kernel_dim1, kernel_dim2=kernel_dim2,kernel_dim3=kernel_dim3, subsample=subsample,
        #                      init=init, border_mode=border_mode, W_regularizer=W_regularizer)(activation)
        return  Conv3D(kernel_initializer=init, strides=subsample, kernel_regularizer=W_regularizer,
                          filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3), padding=border_mode)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_dim1 = (input._keras_shape[CONV_DIM1]+1) // residual._keras_shape[CONV_DIM1]
    stride_dim2 = (input._keras_shape[CONV_DIM2]+1) // residual._keras_shape[CONV_DIM2]
    stride_dim3 = (input._keras_shape[CONV_DIM3]+1) // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    print("input shape:", input._keras_shape)
    # 1 X 1 conv if shape is different. Else identity.
    # if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
    #     shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
    #                              kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
    #                              subsample=(stride_dim1, stride_dim2, stride_dim3),
    #                              init="he_normal", border_mode="valid",
    #                              W_regularizer=l2(0.0001))(input)
    shortcut = Conv3D(kernel_initializer="he_normal", strides=(stride_dim1, stride_dim2, stride_dim3), kernel_regularizer=regularizers.l2(0.0001),
                          filters=residual._keras_shape[CHANNEL_AXIS], kernel_size=(1, 1, 1), padding='valid')(input)

    return add([shortcut, residual])


def _residual_block(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_function(
                    nb_filter=nb_filter,
                    init_subsample=init_subsample,
                    is_first_block_of_first_layer=(is_first_layer and i == 0)
                )(input)
        return input

    return f


def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            # conv1 = Convolution3D(nb_filter=nb_filter,
            #                       kernel_dim1=3, kernel_dim2=3, kernel_dim3=1,
            #                      subsample=init_subsample,
            #                      init="he_normal", border_mode="same",
            #                      W_regularizer=l2(0.0001))(input)
            conv1 = Conv3D(kernel_initializer="he_normal", strides=init_subsample, kernel_regularizer=regularizers.l2(0.0001),
                          filters=nb_filter, kernel_size=(3, 3, 1), padding='same')(input)
        else:
            conv1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1, subsample=init_subsample)(input)

        residual = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of nb_filter * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Convolution3D(nb_filter=nb_filter,
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=init_subsample,
                                 init="he_normal", border_mode="same",
                                 W_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    # if K.image_dim_ordering() == 'tf':
    if K.image_data_format() == 'channels_last':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn_spc, block_fn, repetitions1, repetitions2):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, kernel_dim1, kernel_dim2, kernel_dim3)")

        # Permute dimension order if necessary
        # if K.image_dim_ordering() == 'tf':
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2],input_shape[3], input_shape[0])

        # Load function from str if needed.
        block_fn_spc = _get_block(block_fn_spc)
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        print("input shape:", input._keras_shape[3])

        # ########################## LiKui 2020-06-17 ####################################
        conv1_spc = _conv_bn_relu_spc(nb_filter=24, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(input)

        block_spc = conv1_spc
        nb_filter = 24
        for i, r in enumerate(repetitions1):
            block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block_spc)
            nb_filter *= 2

        # Last activation
        block_spc = _bn_relu_spc(block_spc)

        block_norm_spc = BatchNormalization(axis=CHANNEL_AXIS)(block_spc)
        block_output_spc = Activation("relu")(block_norm_spc)

        conv_spc_results = _conv_bn_relu_spc(nb_filter=128,kernel_dim1=1,kernel_dim2=1,kernel_dim3=block_output_spc._keras_shape[CONV_DIM3])(block_output_spc)

        print("conv_spc_result shape:", conv_spc_results._keras_shape)

        conv2_spc = Reshape((conv_spc_results._keras_shape[CONV_DIM1],conv_spc_results._keras_shape[CONV_DIM2],conv_spc_results._keras_shape[CHANNEL_AXIS],1))(conv_spc_results)

        conv1 = _conv_bn_relu(nb_filter=24, kernel_dim1=3, kernel_dim2=3, kernel_dim3=128,
                              subsample=(1, 1, 1))(conv2_spc)
        #conv1 = _conv_bn_relu(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=input._keras_shape[3], subsample=(1, 1, 1))(input)
        #pool1 = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), border_mode="same")(conv1)
        #conv1 = Convolution3D(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=176,subsample=(1,1,1))(input)
        print("conv1 shape:", conv1._keras_shape)

        block = conv1
        nb_filter = 24
        for i, r in enumerate(repetitions2):
            block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
            nb_filter *= 2

        # conv1_spc = _conv_bn_relu_spc(nb_filter=16, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7, subsample=(1, 1, 2))(
        #     input)
        #
        # block_spc = conv1_spc
        # nb_filter = 16
        # for i, r in enumerate(repetitions1):
        #     block_spc = _residual_block_spc(block_fn_spc, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(
        #         block_spc)
        #     nb_filter *= 2
        #
        # # Last activation
        # block_spc = _bn_relu_spc(block_spc)
        #
        # block_norm_spc = BatchNormalization(axis=CHANNEL_AXIS)(block_spc)
        # block_output_spc = Activation("relu")(block_norm_spc)
        #
        # conv_spc_results = _conv_bn_relu_spc(nb_filter=128, kernel_dim1=1, kernel_dim2=1,
        #                                      kernel_dim3=block_output_spc._keras_shape[CONV_DIM3])(block_output_spc)
        #
        # print("conv_spc_result shape:", conv_spc_results._keras_shape)
        #
        # conv2_spc = Reshape((conv_spc_results._keras_shape[CONV_DIM1], conv_spc_results._keras_shape[CONV_DIM2],
        #                      conv_spc_results._keras_shape[CHANNEL_AXIS], 1))(conv_spc_results)
        #
        # conv1 = _conv_bn_relu(nb_filter=16, kernel_dim1=3, kernel_dim2=3, kernel_dim3=128,
        #                       subsample=(1, 1, 1))(conv2_spc)
        # # conv1 = _conv_bn_relu(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=input._keras_shape[3], subsample=(1, 1, 1))(input)
        # # pool1 = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), border_mode="same")(conv1)
        # # conv1 = Convolution3D(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=176,subsample=(1,1,1))(input)
        # print("conv1 shape:", conv1._keras_shape)
        #
        # block = conv1
        # nb_filter = 16
        # for i, r in enumerate(repetitions2):
        #     block = _residual_block(block_fn, nb_filter=nb_filter, repetitions=r, is_first_layer=(i == 0))(block)
        #     nb_filter *= 2

        # Last activation
        block = _bn_relu(block)

        block_norm = BatchNormalization(axis=CHANNEL_AXIS)(block)
        block_output = Activation("relu")(block_norm)

        # Classifier block
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[CONV_DIM1],
                                            block._keras_shape[CONV_DIM2],
                                            block._keras_shape[CONV_DIM3],),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        drop1 = Dropout(0.5)(flatten1)
        dense = Dense(units=num_outputs, activation="softmax", kernel_initializer="he_normal")(drop1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [1],[1])      #[2, 2, 2, 2]

    @staticmethod
    def build_resnet_12(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block_spc, basic_block, [2], [2])

def main():
    model = ResnetBuilder.build_resnet_8((1, 7, 7, 200), 16)            # IN DATASET model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
    #model = ResnetBuilder.build_resnet_6((1,7,7,176), 13)               # KSC DATASET
    #model = ResnetBuilder.build_resnet_6((1, 7, 7, 103), 9)             # UP DATASET
    #model = ResnetBuilder.build_resnet_34((1, 27, 27, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


###############################################################################

import os

import numpy as np
# import matplotlib.pyplot as plt
import scipy.io as sio
# from keras.models import Sequential, Model
# from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, \
#     ZeroPadding3D
# from keras.layers import Activation, Dropout, Flatten, Dense, \
#     BatchNormalization, Input
from keras.utils.np_utils import to_categorical
# from sklearn.decomposition import PCA
# from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
from keras.optimizers import RMSprop
import keras.callbacks as kcallbacks
# from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from operator import truediv

# from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, \
#     averageAccuracy, ssrn_SS_IN

# sys.path.append('/home')
#
# from hsi_utils.display import display_map_and_save

from config import *


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:,
                     range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


def sampling(proptionVal,
             groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if
                   x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    #    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def res4_model_ss(img_rows, img_cols, img_channels, nb_classes, lr):

    model_res4 = ResnetBuilder.build_resnet_8(
        (1, img_rows, img_cols, img_channels), nb_classes)

    RMS = RMSprop(lr=lr)
    # Let's train the model using RMSprop
    model_res4.compile(loss='categorical_crossentropy', optimizer=RMS,
                       metrics=['accuracy'])

    return model_res4


def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def outputStats(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, history, loss_and_metrics, CATEGORY, path1, path2):

    # f = open(path1, 'a')
    f = open(path1, 'w')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :' + str(np.sum(TRAINING_TIME_AE)) + '\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:' + str(np.sum(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    # print(print_matrix)
    # print(element_mean)
    # print(element_std)
    for i in range(CATEGORY):
        # print(i)
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    # np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t", newline='\n')
    with open(path2, 'w') as f:
        for s in print_matrix:
            f.write(s + '\n')

    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print(history.history.keys())


# ################ LiKui 2020-5-13 ####################################################
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_dir', type=str)
#     parser.add_argument('--target_dir', type=str)
#     parser.add_argument('--mask_dir', type=str)
#     parser.add_argument('--data_name', type=str)
#     parser.add_argument('--train_prop', type=str)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--epoch', type=int, default=200)
#     parser.add_argument('--patch_size', type=int, default=7)
#     parser.add_argument('--gpu', type=str, default='0')
#     args = parser.parse_args()
#     return args


def get_indices(mask):
    indices = np.where(mask.ravel() == 1)[0]
    np.random.shuffle(indices)
    return indices.tolist()


# ############################# LiKui 2020-6-20 #####################################
# def check_path(path):
#     path_tuple = os.path.splitext(path)
#
#     if os.path.exists(path):
#         i = 1
#         while True:
#             path = '%s(%s)%s' % (path_tuple[0], i, path_tuple[1])
#             if not os.path.exists(path):
#                 break
#             i += 1
#
#     return path


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
    patch_size = params.data.patch_size
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
    model_dir = os.path.join(result_dir, 'weights.hdf5')
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

    # args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # # mat_data = sio.loadmat('/home/zilong/SSRN/datasets/IN/Indian_pines_corrected.mat')
    # mat_data = sio.loadmat('./datasets/IN/Indian_pines_corrected.mat')
    # data_IN = mat_data['indian_pines_corrected']
    # # mat_gt = sio.loadmat('/home/zilong/SSRN/datasets/IN/Indian_pines_gt.mat')
    # mat_gt = sio.loadmat('./datasets/IN/Indian_pines_gt.mat')
    # gt_IN = mat_gt['indian_pines_gt']

    data_IN = sio.loadmat(data_dir)
    data_IN = list(filter(lambda x: isinstance(x, np.ndarray), data_IN.values()))[0]
    gt_IN = sio.loadmat(target_dir)
    gt_IN = list(filter(lambda x: isinstance(x, np.ndarray), gt_IN.values()))[0]

    # print(data_IN.shape)

    # mask_dir = args.mask_dir
    mask_dir = os.path.join(os.path.dirname(data_dir),
                            'masks_%s_%s' % (train_prop, val_prop))
    train_mask = sio.loadmat(os.path.join(mask_dir, 'train_mask.mat'))['train_mask']
    val_mask = sio.loadmat(os.path.join(mask_dir, 'val_mask.mat'))['val_mask']
    test_mask = sio.loadmat(os.path.join(mask_dir, 'test_mask'))['test_mask']

    train_indices = get_indices(train_mask)
    val_indices = get_indices(val_mask)
    test_indices = get_indices(test_mask)

    # ##########################################################################################
    # #new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
    # new_gt_IN = gt_IN
    #
    # batch_size = 16
    # nb_classes = 16
    # nb_epoch = 200  #400
    # img_rows, img_cols = 7, 7         #27, 27
    # patience = 200
    #
    # INPUT_DIMENSION_CONV = 200
    # INPUT_DIMENSION = 200
    #
    # # 20%:10%:70% data for training, validation and testing
    #
    # TOTAL_SIZE = 10249
    # VAL_SIZE = 1025
    #
    # TRAIN_SIZE = 2055
    # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    # VALIDATION_SPLIT = 0.8                      # 20% for trainnig and 80% for validation and testing
    # # TRAIN_NUM = 10
    # # TRAIN_SIZE = TRAIN_NUM * nb_classes
    # # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    # # VAL_SIZE = TRAIN_SIZE
    #
    # img_channels = 200
    # PATCH_LENGTH = 3                #Patch_size (13*2+1)*(13*2+1)
    # ############################################################################################

    # new_gt_IN = set_zeros(gt_IN, [1,4,7,9,13,15,16])
    new_gt_IN = gt_IN

    # data_name = args.data_name
    # train_prop = args.train_prop
    # batch_size = args.batch_size
    batch_size = train_params.batch_size
    nb_classes = int(gt_IN.max())
    # nb_epoch = args.epoch
    nb_epoch = train_params.epoch
    lr = train_params.lr
    img_rows, img_cols = patch_size, patch_size  # 27, 27
    patience = 200

    INPUT_DIMENSION_CONV = data_IN.shape[-1]
    INPUT_DIMENSION = data_IN.shape[-1]

    TOTAL_SIZE = len(gt_IN[gt_IN != 0])
    VAL_SIZE = len(val_indices)

    TRAIN_SIZE = len(train_indices)
    TEST_SIZE = len(test_indices)
    ALL_SIZE = data_IN.shape[0] * data_IN.shape[1]
    # VALIDATION_SPLIT = 0.8                      # 20% for trainnig and 80% for validation and testing
    # TRAIN_NUM = 10
    # TRAIN_SIZE = TRAIN_NUM * nb_classes
    # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    # VAL_SIZE = TRAIN_SIZE

    img_channels = data_IN.shape[-1]
    PATCH_LENGTH = patch_size // 2  # Patch_size (13*2+1)*(13*2+1)

    data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

    data = preprocessing.scale(data)

    # scaler = preprocessing.MaxAbsScaler()
    # data = scaler.fit_transform(data)

    data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
    whole_data = data_
    padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

    ITER = 1
    CATEGORY = nb_classes

    train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1,
                           INPUT_DIMENSION_CONV))
    test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1,
                          INPUT_DIMENSION_CONV))
    val_data = np.zeros((VAL_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1,
                         INPUT_DIMENSION_CONV))
    all_data = np.zeros((ALL_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1,
                         INPUT_DIMENSION_CONV))

    KAPPA_RES_SS4 = []
    OA_RES_SS4 = []
    AA_RES_SS4 = []
    TRAINING_TIME_RES_SS4 = []
    TESTING_TIME_RES_SS4 = []
    ELEMENT_ACC_RES_SS4 = np.zeros((ITER, CATEGORY))

    # seeds = [1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229]

    # seeds = [1334]

    # models_dir = './models/{}'.format(data_name)
    # records_dir = './records/{}'.format(data_name)
    # models_dir = './models'
    # records_dir = './records/{}'.format(data_name)  # ###########################
    # map_save = './Cmaps/{}'.format(data_name)  # ###########################

    # ################## tmp ###################################################
    # models_dir = './tmp/models'
    # records_dir = './tmp/records'
    # map_save = './tmp/maps'

    # if not os.path.exists(models_dir):
    #     os.makedirs(models_dir)
    # if not os.path.exists(records_dir):
    #     os.makedirs(records_dir)
    # if not os.path.exists(map_save):
    #     os.makedirs(map_save)

    for index_iter in range(ITER):
        #     print("# %d Iteration" % (index_iter + 1))

        # save the best validated model

        # best_weights_RES_path_ss4 = '/home/zilong/SSRN/models/Indian_best_RES_3D_SS4_10_' + str(
        #     index_iter + 1) + '.hdf5'

        # best_weights_RES_path_ss4 = os.path.join(models_dir,
        #                                          '{}_best_RES_3D_SS4_{}.hdf5'.format(
        #                                              data_name, train_prop,
        #                                          ))

        # if os.path.exists(best_weights_RES_path_ss4):  # #############################
        #     os.system('rm %s' % best_weights_RES_path_ss4)

        # best_weights_RES_path_ss4 = check_path(best_weights_RES_path_ss4)
        best_weights_RES_path_ss4 = model_dir

        # np.random.seed(seeds[index_iter])
        #    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
        #     train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

        # TRAIN_SIZE = len(train_indices)
        # print (TRAIN_SIZE)
        #
        # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
        # print (TEST_SIZE)

        y_train = gt[train_indices] - 1
        y_train = to_categorical(np.asarray(y_train))

        y_test = gt[test_indices] - 1
        y_test = to_categorical(np.asarray(y_test))

        y_val = gt[val_indices] - 1
        y_val = to_categorical(np.asarray(y_val))

        # print ("Validation data:")
        # collections.Counter(y_test_raw[-VAL_SIZE:])
        # print ("Testing data:")
        # collections.Counter(y_test_raw[:-VAL_SIZE])

        train_assign = indexToAssignment(train_indices, whole_data.shape[0],
                                         whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(train_assign)):
            train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0],
                                                   train_assign[i][1], PATCH_LENGTH)

        test_assign = indexToAssignment(test_indices, whole_data.shape[0],
                                        whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(test_assign)):
            test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0],
                                                  test_assign[i][1], PATCH_LENGTH)

        val_assign = indexToAssignment(val_indices, whole_data.shape[0],
                                       whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(val_assign)):
            val_data[i] = selectNeighboringPatch(padded_data, val_assign[i][0],
                                                 val_assign[i][1], PATCH_LENGTH)

        x_train = train_data.reshape(train_data.shape[0], train_data.shape[1],
                                     train_data.shape[2], INPUT_DIMENSION_CONV)
        # x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)
        x_test = test_data.reshape(test_data.shape[0], test_data.shape[1],
                                   test_data.shape[2], INPUT_DIMENSION_CONV)
        x_val = val_data.reshape(val_data.shape[0], val_data.shape[1],
                                 val_data.shape[2], INPUT_DIMENSION_CONV)

        # x_val = x_test_all[-VAL_SIZE:]
        # y_val = y_test[-VAL_SIZE:]
        #
        # x_test = x_test_all[:-VAL_SIZE]
        # y_test = y_test[:-VAL_SIZE]

        # SS Residual Network 4 with BN
        model_res4_SS_BN = res4_model_ss(img_rows, img_cols, img_channels,
                                         nb_classes, lr=lr)

        earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience, verbose=1,
                                                  mode='auto')
        saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_RES_path_ss4,
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    mode='auto')

        tic6 = time.clock()
        # tic6 = time.time()
        logger.info('#################### train #############################')
        print(x_train.shape, x_test.shape)
        history_res4_SS_BN = model_res4_SS_BN.fit(
            x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],
                            x_train.shape[3], 1), y_train,
            validation_data=(
            x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2],
                          x_val.shape[3], 1), y_val),
            batch_size=batch_size,
            nb_epoch=nb_epoch, shuffle=True,
            callbacks=[earlyStopping6, saveBestModel6])
        # toc6 = time.time()
        toc6 = time.clock()

        # tic7 = time.time()
        tic7 = time.clock()
        loss_and_metrics_res4_SS_BN = model_res4_SS_BN.evaluate(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                           x_test.shape[3], 1), y_test,
            batch_size=batch_size)
        # toc7 = time.time()
        toc7 = time.clock()

        print('3D RES_SS4 without BN Training Time: ', toc6 - tic6)
        print('3D RES_SS4 without BN Test time:', toc7 - tic7)

        print('3D RES_SS4 without BN Test score:', loss_and_metrics_res4_SS_BN[0])
        print('3D RES_SS4 without BN Test accuracy:',
              loss_and_metrics_res4_SS_BN[1])

        print(history_res4_SS_BN.history.keys())

        pred_test_res4 = model_res4_SS_BN.predict(
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],
                           x_test.shape[3], 1)).argmax(axis=1)
        collections.Counter(pred_test_res4)
        gt_test = gt[test_indices] - 1

        # overall_acc_res4 = metrics.accuracy_score(pred_test_res4, gt_test[:-VAL_SIZE])
        # confusion_matrix_res4 = metrics.confusion_matrix(pred_test_res4, gt_test[:-VAL_SIZE])
        # each_acc_res4, average_acc_res4 = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix_res4)
        # kappa = metrics.cohen_kappa_score(pred_test_res4, gt_test[:-VAL_SIZE])

        overall_acc_res4 = metrics.accuracy_score(pred_test_res4, gt_test)
        confusion_matrix_res4 = metrics.confusion_matrix(pred_test_res4, gt_test)
        each_acc_res4, average_acc_res4 = AA_andEachClassAccuracy(
            confusion_matrix_res4)
        kappa = metrics.cohen_kappa_score(pred_test_res4, gt_test)

        KAPPA_RES_SS4.append(kappa)
        OA_RES_SS4.append(overall_acc_res4)
        AA_RES_SS4.append(average_acc_res4)
        TRAINING_TIME_RES_SS4.append(toc6 - tic6)
        TESTING_TIME_RES_SS4.append(toc7 - tic7)
        ELEMENT_ACC_RES_SS4[index_iter, :] = each_acc_res4

        print("3D RESNET_SS4 without BN training finished.")
        # print("# %d Iteration" % (index_iter + 1))

        # modelStatsRecord.outputStats(KAPPA_RES_SS4, OA_RES_SS4, AA_RES_SS4, ELEMENT_ACC_RES_SS4,
        #                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
        #                              history_res4_SS_BN, loss_and_metrics_res4_SS_BN, CATEGORY,
        #                              '/home/zilong/SSRN/records/IN_train_SS_10.txt',
        #                              '/home/zilong/SSRN/records/IN_train_SS_element_10.txt')

        # ##################### predict all data #########################################
        logger.info('#################### predict #############################')
        all_assign = indexToAssignment(range(ALL_SIZE), whole_data.shape[0],
                                       whole_data.shape[1], PATCH_LENGTH)
        for i in range(len(all_assign)):
            all_data[i] = selectNeighboringPatch(padded_data, all_assign[i][0],
                                                 all_assign[i][1], PATCH_LENGTH)

        tic8 = time.clock()
        # tic8 = time.time()
        pred_test_conv1_ = model_res4_SS_BN.predict(
            all_data.reshape(all_data.shape[0], all_data.shape[1],
                             all_data.shape[2], all_data.shape[3], 1))
        # toc8 = time.time()
        toc8 = time.clock()
        pred_test_conv1 = pred_test_conv1_.argmax(axis=1)

        # x = np.ravel(pred_test_conv1)

        # map_dir = '{}.png'.format(
        #     os.path.basename(os.path.splitext(best_weights_RES_path_ss4)[0]))
        # mat_dir = '{}.mat'.format(
        #     os.path.basename(os.path.splitext(best_weights_RES_path_ss4)[0]))
        # prob_mat = '{}.mat'.format(os.path.basename(
        #     os.path.splitext(best_weights_RES_path_ss4)[0]) + '_prob')
        # map_dir = os.path.join(map_save, map_dir)
        # mat_dir = os.path.join(map_save, mat_dir)
        # prob_mat = os.path.join(map_save, prob_mat)

        # map_dir = check_path(map_dir)
        # mat_dir = check_path(mat_dir)
        # prob_mat = check_path(prob_mat)

        # print('\nmap_dir: %s' % map_dir)
        # print('mat_dir: %s' % mat_dir)
        # print('prob_mat: %s\n' % prob_mat)

        # display_map_and_save(pred_test_conv1.reshape(data_IN.shape[:2]) + 1,
        #                      map_dir)
        logger.info('save soft label ...')
        sio.savemat(soft_dir,
                    {'soft_label': pred_test_conv1_.reshape(data_IN.shape[:2] + (-1,))})
        # sio.savemat(mat_dir,
        #             {'map': pred_test_conv1.reshape(data_IN.shape[:2]) + 1})

    path1 = os.path.join(result_dir, 'ssrn_acc.txt')
    path2 = os.path.join(result_dir, 'ssrn_ele_acc.txt')

    # path1 = check_path(path1)
    # path2 = check_path(path2)
    #
    # print('\npath1: %s' % path1)
    # print('path2: %s\n' % path2)

    outputStats(KAPPA_RES_SS4, OA_RES_SS4, AA_RES_SS4,
                 ELEMENT_ACC_RES_SS4,
                 TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                 history_res4_SS_BN, loss_and_metrics_res4_SS_BN,
                 CATEGORY,
                 path1,
                 path2)

    # 2020-08-04, LiKui, record time
    # time_txt = os.path.join(records_dir, 'time.txt')
    train_time = toc6 - tic6
    train_time_mean = train_time / nb_epoch
    predict_time = toc8 - tic8
    logger.info('training time: %.4fs' % train_time)
    logger.info('training time mean: %.4fs' % train_time_mean)
    logger.info('predicted time: %.4fs' % predict_time)

    # if not os.path.exists(time_txt):
    #     with open(time_txt, 'w') as f:
    #         f.write('training time\ttraining time mean\tpredicted time\n')
    #         f.write('%s\t%s\t%s\n' % (train_time, train_time_mean, predict_time))
    # else:
    #     with open(time_txt, 'a') as f:
    #         f.write('%s\t%s\t%s\n' % (train_time, train_time_mean, predict_time))

    pred = pred_test_conv1.reshape(data_IN.shape[:2]) + 1

    logger.info('save classification maps etc. ...')
    train_records = {
        'train_time': '%.4f' % train_time,
        'train_time_mean': '%.4f' % train_time_mean,
        'pred_time': '%.4f' % predict_time
    }

    ro = ResultOutput(pred, data_IN, gt_IN, train_mask, val_mask, test_mask,
                      result_dir, acc_dir, hyper_params=params,
                      train_records=train_records)
    ro.output()


