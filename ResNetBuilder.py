from __future__ import print_function

import os
import uuid
import numpy as np
import tensorflow.keras
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Layers import MaskedConv2D, MaskedDense
from MyBiasedLayers import MaskedConv2DBiased, MaskedDenseBias
import tensorflow

print("TF version:        ", tensorflow.__version__)
print("TF.keras version:  ", tensorflow.keras.__version__)


def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True, layersconfig=None):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    # conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    # config = {
    #     "name": "LeNet",
    #     "data": "MNIST",
    #     "arch": dense_arch,
    #     "seed": myseed,
    #     "initializer": initializer,
    #     "activation": activation,
    #     "masktype": masktype,
    #     "trainW": trainW,
    #     "trainM": trainM,
    #     "p1": p1,
    #     "abg": alphabetagamma
    # }

    myseed = layersconfig['seed']
    trainW, trainM = layersconfig['trainW'], layersconfig['trainM']
    initializer = layersconfig['initializer']
    masktype = layersconfig['masktype']
    alpha = layersconfig['abg']
    p1 = layersconfig['p1']

    conv = MaskedConv2D((kernel_size, kernel_size), num_filters, activation, myseed, initializer, strides, masktype, trainW, trainM, p1, alpha)

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10, initializer='heconstant', layersconfig=None):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, layersconfig=layersconfig)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides, layersconfig=layersconfig)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None, layersconfig=layersconfig)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False, layersconfig=layersconfig)

            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    # np.random.seed(None)
    # myseed = np.random.randint(2 ** 31 - 1)
    # trainW, trainM = False, True

    myseed = layersconfig['seed']
    trainW, trainM = layersconfig['trainW'], layersconfig['trainM']
    initializer = layersconfig['initializer']
    masktype = layersconfig['masktype']
    alpha = layersconfig['abg']
    p1 = layersconfig['p1']

    dense = MaskedDense(num_classes, 'softmax', seed=myseed, initializer=initializer, masktype=masktype, trainweights=trainW, trainmask=trainM, p1=p1, alpha=alpha)

    outputs = dense(y)
    outputs = Activation('softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def MakeResNet(input_shape, version=1, n=3, layersconfig=None):
    # Computed depth from supplied model parameter n
    depth = 1
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2

    model = resnet_v1(input_shape=input_shape, depth=depth, layersconfig=layersconfig)

    return model
