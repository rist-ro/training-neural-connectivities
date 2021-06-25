import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import numpy as np
from Functions import heconstant, activate, mask, mask_rs, flip, binary

minval = 0.01
maxval = 0.1


class MaskedConv2D(Layer):
    def __init__(self, ksize, filters, activation, seed, initializer, stride, masktype, trainweights, trainmask, p1, alpha, **kwargs):

        self.filters = filters
        self.seed = seed
        self.stride = 1
        self.p1 = p1
        self.alpha = alpha

        if stride is not None:
            self.stride = stride

        self.initializer = initializer

        if masktype == "flip":
            self.masktype = flip

        if masktype == "mask":
            self.masktype = mask

        if masktype == "mask_rs":
            self.masktype = mask_rs

        self.trainW = trainweights
        self.trainM = trainmask
        self.kernelsize = ksize

        self.activation = activation

        super(MaskedConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.initializer == 'normal':
            ki = tf.compat.v1.keras.initializers.RandomNormal(mean=0., stddev=0.05, seed=self.seed)

        if self.initializer == 'glorot':
            ki = tf.compat.v1.keras.initializers.glorot_normal(self.seed)

        if self.initializer == 'he':
            ki = tf.compat.v1.keras.initializers.he_normal(self.seed)

        if self.initializer == "heconstant":
            ki = heconstant(self.p1, self.seed)

        if self.initializer == "binary":
            ki = binary(self.p1, self.seed)

        kshape = list(self.kernelsize) + [input_shape.as_list()[-1], self.filters]
        self.kernel = self.add_weight(name='kernel', shape=kshape, initializer=ki, trainable=self.trainW)

        si = tf.compat.v1.keras.initializers.RandomUniform(minval=minval, maxval=maxval, seed=self.seed)
        self.score = self.add_weight(name='score', shape=kshape, initializer=si, trainable=self.trainM)

        if self.alpha != 0:
            self.add_loss(self.alpha * tf.reduce_mean(self.masktype(self.score)))

        super(MaskedConv2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        """
        THis is the layer's logic
        :param x: input
        :return: output
        """

        act = K.conv2d(x, self.kernel * self.masktype(self.score), strides=(self.stride, self.stride), padding='same')
        # act = activate(act, self.activation)

        return act

    # needed for keras to calculate the outputshape of an operation
    def compute_output_shape(self, input_shape):
        return (input_shape.as_list()[1], self.output_dim)

    # called for a layer's weights
    def get_weights(self):
        return K.eval(self.kernel)

    def get_pruneamount(self):
        weights_mask = K.eval(self.masktype(self.score))
        nz = np.count_nonzero(weights_mask)
        total = weights_mask.size
        return nz, total

    def get_score(self):
        return K.eval(self.score)

    def get_mask(self):
        return K.eval(self.masktype(self.score))

    def get_nzpmasks(self):
        k = K.eval(self.masktype(self.score))
        neg = np.count_nonzero(k < 0)
        zeros = np.count_nonzero(k == 0)
        pos = np.count_nonzero(k > 0)
        return neg, zeros, pos

    def get_kernel(self):
        return K.eval(self.kernel)

    def get_seed(self):
        return self.seed

    def set_weights(self, weights):
        super(MaskedConv2D, self).set_weights(weights)


class MaskedDense(Layer):

    def __init__(self, output_dim, activation, seed, initializer, masktype, trainweights, trainmask, p1, alpha, **kwargs):
        self.output_dim = output_dim
        self.seed = seed
        self.p1 = p1
        self.alpha = alpha
        self.initializer = initializer

        if masktype == "flip":
            self.masktype = flip

        if masktype == "mask":
            self.masktype = mask

        if masktype == "mask_rs":
            self.masktype = mask_rs

        self.trainW = trainweights
        self.trainM = trainmask
        self.activation = activation

        super(MaskedDense, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.initializer == 'normal':
            ki = tf.compat.v1.keras.initializers.RandomNormal(mean=0.1, stddev=0.05, seed=self.seed)

        if self.initializer == 'glorot':
            ki = tf.compat.v1.keras.initializers.glorot_normal(self.seed)

        if self.initializer == 'he':
            ki = tf.compat.v1.keras.initializers.he_normal(self.seed)

        if self.initializer == "heconstant":
            ki = heconstant(self.p1, self.seed)

        if self.initializer == "binary":
            ki = binary(self.p1, self.seed)

        kshape = (input_shape.as_list()[1], self.output_dim)

        # define weights using the API available method (self.add_weights)
        self.kernel = self.add_weight(name='kernel', shape=kshape, initializer=ki, trainable=self.trainW)

        si = tf.compat.v1.keras.initializers.RandomUniform(minval=minval, maxval=maxval, seed=self.seed)
        self.score = self.add_weight(name='score', shape=kshape, initializer=si, trainable=self.trainM)

        if self.alpha != 0:
            self.add_loss(self.alpha * tf.reduce_mean(self.masktype(self.score)))

        super(MaskedDense, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        """
        THis is the layer's logic
        :param x: input
        :return: output
        """

        act = K.dot(x, self.kernel * self.masktype(self.score))
        # act = activate(act, self.activation)

        return act

    # needed for keras to calculate the outputshape of an operation
    def compute_output_shape(self, input_shape):
        return (input_shape.as_list()[1], self.output_dim)

    # called for a layer's weights
    def get_weights(self):
        return K.eval(self.kernel)

    def get_pruneamount(self):
        weights_mask = K.eval(self.masktype(self.score))
        nz = np.count_nonzero(weights_mask)
        total = weights_mask.size
        return nz, total

    def get_score(self):
        return K.eval(self.score)

    def get_mask(self):
        return K.eval(self.masktype(self.score))

    def get_nzpmasks(self):
        k = K.eval(self.masktype(self.score))
        neg = np.count_nonzero(k < 0)
        zeros = np.count_nonzero(k == 0)
        pos = np.count_nonzero(k > 0)

        return neg, zeros, pos

    def get_kernel(self):
        return K.eval(self.kernel)

    def get_seed(self):
        return self.seed

    def set_weights(self, weights):
        super(MaskedDense, self).set_weights(weights)
