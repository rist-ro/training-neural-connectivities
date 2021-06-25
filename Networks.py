import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Input, Flatten
from Layers import MaskedDense, MaskedConv2D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation


def makeMaskedCNN(inshape, cnn_arch, dense_arch, activation, myseed, initializer, masktype, trainW, trainM, p1, alpha):
    inputshape = inshape
    arch = dense_arch

    input_img = Input(shape=inputshape)
    np.random.seed(myseed)
    seeds = np.random.randint(1, np.iinfo(np.int32).max, len(cnn_arch) + len(dense_arch))

    # add the first cnn layer
    CI = MaskedConv2D(cnn_arch[0][:2], cnn_arch[0][-1], activation, seeds[0], initializer, 1, masktype, trainW, trainM, p1, alpha)(input_img)
    CI = Activation(activation)(CI)

    # add the next layers
    for i in range(1, len(cnn_arch)):

        # this is a cnn layer
        if len(cnn_arch[i]) != 0:
            CI = MaskedConv2D(cnn_arch[i][:2], cnn_arch[i][-1], activation, seeds[i], initializer, 1, masktype, trainW, trainM, p1, alpha)(CI)
            CI = Activation(activation)(CI)

        # this is a maxpool layer
        if len(cnn_arch[i]) == 0:
            CI = MaxPooling2D(pool_size=(2, 2))(CI)

    LF = Flatten()(CI)
    for i in range(0, len(arch) - 1):
        LF = MaskedDense(dense_arch[i], activation, seeds[len(cnn_arch) + i + 1], initializer, masktype, trainW, trainM, p1, alpha)(LF)

    LF = MaskedDense(dense_arch[-1], 'softmax', seeds[-1], initializer, masktype, trainW, trainM, p1, alpha)(LF)

    # define the model, connecting the input to the last layer
    model = Model(input_img, LF)

    # set a network name
    import uuid
    ID = uuid.uuid4().hex

    model._name = "CNN_" + ID[len(ID) - 7:] + "_S" + str(myseed)

    return model


def makeMaskedMLP(fullarch, activation, myseed, initializer, masktype, trainW, trainM, p1, alpha):
    inputshape = fullarch[0]
    outshape = fullarch[-1]
    arch = fullarch[1:-1]

    input_img = Input(shape=(inputshape,))
    np.random.seed(myseed)
    seeds = np.random.randint(1, np.iinfo(np.int32).max, len(fullarch))

    # if there are no hidden layers then add just
    # the last layer connected to the input (input_img)
    if len(arch) == 0:
        LN = MaskedDense(outshape, 'softmax', seeds[0], initializer, masktype, trainW, trainM, p1, alpha)(input_img)
        LN = Activation('softmax')(LN)


    # if there are hidden layers then
    else:
        # add the first hidden layer and connect it to the input
        Li = MaskedDense(arch[0], activation, seeds[0], initializer, masktype, trainW, trainM, p1, alpha)(input_img)
        Li = Activation(activation)(Li)

        # add the rest of the hidden layers (if any) and connect
        # them to the previous ones
        for i in range(1, len(arch)):
            Li = MaskedDense(arch[i], activation, seeds[i], initializer, masktype, trainW, trainM, p1, alpha)(Li)
            Li = Activation(activation)(Li)

        # here is the last layer, connected to the one before
        # (either the ones from the loop or the one before)
        LN = MaskedDense(outshape, 'softmax', seeds[-1], initializer, masktype, trainW, trainM, p1, alpha)(Li)
        LN = Activation('softmax')(LN)

    # define the model, connecting the input to the last layer (LN)
    model = Model(input_img, LN)

    # set a network name
    import uuid
    ID = uuid.uuid4().hex
    model._name = "MLP_" + ID[len(ID) - 7:] + "_S" + str(myseed)

    return model
