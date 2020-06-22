import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import uuid


def heconstant(p1, myseed):
    def initializer(shape, dtype=None):
        a = np.sqrt(2 / np.prod(shape[:-1]))
        p2 = 1. - p1
        np.random.seed(myseed)
        distribution = np.random.choice([1., -1.], shape, p=[p1, p2])
        return tf.Variable(a * distribution, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def binary(p1, myseed):
    def initializer(shape, dtype=None):
        p2 = 1. - p1
        np.random.seed(myseed)
        distribution = np.random.choice([1., -1.], shape, p=[p1, p2])
        return tf.Variable(distribution, dtype=dtype, name=uuid.uuid4().hex)

    return initializer


def activate(x, activationtype):
    if 'relu' in activationtype:
        return tf.keras.activations.relu(x)

    if 'softmax' in activationtype:
        return tf.keras.activations.softmax(x)

    if 'sigmoid' in activationtype:
        return tf.keras.activations.sigmoid(x)

    if 'swish' in activationtype:
        return tf.keras.activations.sigmoid(x) * x

    if "elu" in activationtype:
        return tf.keras.activations.elu(x)

    if "selu" in activationtype:
        return tf.keras.activations.selu(x)

    if activationtype is None:
        return x

    return x


@tf.custom_gradient
def mask(x):
    y = K.sign(tf.keras.activations.relu(x))

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def mask_rs(x):
    y = K.sign(tf.keras.activations.relu(x))

    # some papers (arXiv:1905.01067v4 and arXiv:1911.13299v1) do a
    # rescaling of the weights/masks while backpropagating, we can do it here as well
    scalefactor = tf.compat.v1.size(y, out_type=tf.dtypes.float32) / (1 + tf.math.count_nonzero(y, dtype=tf.dtypes.float32))
    y *= scalefactor

    def grad(dy):
        return dy

    return y, grad


@tf.custom_gradient
def flip(x):
    y = K.sign(x)

    def grad(dy):
        return dy

    return y, grad
