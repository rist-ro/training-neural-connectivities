import numpy as np


def getNZPmasks(net):
    nsum = 0
    zsum = 0
    psum = 0

    for l in range(1, len(net.layers)):
        if "masked" not in net.layers[l].name:
            continue

        neg, zero, pos = net.layers[l].get_nzpmasks()
        nsum += neg
        zsum += zero
        psum += pos

    return nsum, zsum, psum


def SetMyData(datatype, w=1):
    if "CIFAR" == datatype:
        # data = cifar10()  # PrepareCIFAR10Data()
        Xtrain, Ytrain, Xtest, Ytest, nclasses = cifar10()  # SplitDataTrnValTst(data)
        Xval, Yval = Xtest, Ytest

        return Xtrain * w, Ytrain, Xval * w, Yval, Xtest * w, Ytest, nclasses

    if "MNIST" == datatype:
        data = mnist()  # PrepareMNISTData()
        # imx = data[0].shape[1]
        # imy = data[0].shape[2]
        # nchannels = data[0].shape[3]

        Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, nclasses = mnist()

        # Xtrain = Xtrain.reshape(-1, imx * imy * nchannels)
        # Xval = Xval.reshape(-1, imx * imy * nchannels)
        # Xtest = Xtest.reshape(-1, imx * imy * nchannels)

        return Xtrain * w, Ytrain, Xval * w, Yval, Xtest * w, Ytest, nclasses


def PrepareCIFAR10Data():
    print("\n\n\n")
    from tensorflow.keras.datasets import cifar10

    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

    TrainInput = xtrain / 255.
    TestInput = xtest / 255.

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    TrainLabels = np.zeros((len(ytrain), 10))
    for i in range(0, len(ytrain)):
        TrainLabels[i, ytrain[i]] = 1

    TestLabels = np.zeros((len(ytest), 10))
    for i in range(0, len(ytest)):
        TestLabels[i, ytest[i]] = 1

    return np.ascontiguousarray(TrainInput), np.ascontiguousarray(TrainLabels), np.ascontiguousarray(TestInput), np.ascontiguousarray(TestLabels), 10


def PrepareMNISTData():
    from tensorflow.keras.datasets import mnist

    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    TrainInput = xtrain.reshape(-1, 28, 28, 1) / 255.
    TestInput = xtest.reshape(-1, 28, 28, 1) / 255.

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    TrainLabels = np.zeros((len(ytrain), 10))
    for i in range(0, len(ytrain)):
        TrainLabels[i, ytrain[i]] = 1

    TestLabels = np.zeros((len(ytest), 10))
    for i in range(0, len(ytest)):
        TestLabels[i, ytest[i]] = 1

    return TrainInput, TrainLabels, TestInput, TestLabels, 10


def SplitDataTrnValTst(data):
    Xtrain = data[0]
    Ytrain = data[1]
    Xtst = data[2]
    Ytst = data[3]

    nval = 5000

    XVal = Xtrain[:nval]
    YVal = Ytrain[:nval]

    Xtrn = Xtrain[nval:]
    Ytrn = Ytrain[nval:]

    return Xtrn, Ytrn, XVal, YVal, Xtst, Ytst, Ytst.shape[1]


def mnist():
    with np.load("data/mnist.npz") as data:
        Xtrain = data['training_images']
        Ytrain = data['training_labels']
        Xtest = data['test_images']
        Ytest = data['test_labels']
        Xval = data['validation_images']
        Yval = data['validation_labels']

        # print(Xtrain.shape)
        # input()

        Xtrain -= np.mean(Xtrain, axis=0)
        Xval -= np.mean(Xval, axis=0)
        Xtest -= np.mean(Xtest, axis=0)

        Xtrain /= (np.std(Xtrain))
        Xval /= (np.std(Xval))
        Xtest /= (np.std(Xtest))

        # w = .0005942073791483592  # paper original value
        # w = .05942073791483592
        w = 1

        Xtrain *= w
        Xtest *= w
        Xval *= w

        # print(Xtrain.shape)
        # print(Xval.shape)
        # print(Xtest.shape)
        # input()

        # return w * Xtrain[:, :, 0], Ytrain[:, :, 0], w * Xtest[:, :, 0], Ytest[:, :, 0], w * Xval[:, :, 0], Yval[:, :, 0], 10
        return Xtrain[..., 0], Ytrain[..., 0], Xtest[..., 0], Ytest[..., 0], Xval[..., 0], Yval[..., 0], 10


def cifar10(ss=None):
    mypath = 'data/cifar-10-python/'
    xtrain = np.load(mypath + "xtrain.npy")
    ytrain = np.load(mypath + "ytrain.npy")
    xtest = np.load(mypath + "xtest.npy")
    ytest = np.load(mypath + "ytest.npy")

    TrainInput = xtrain / 255.
    TestInput = xtest / 255.

    TrainInput -= np.mean(TrainInput, axis=0)
    TestInput -= np.mean(TestInput, axis=0)

    TrainInput /= (np.std(TrainInput))
    TestInput /= (np.std(TestInput))

    TrainLabels = np.zeros((len(ytrain), 10))
    for i in range(0, len(ytrain)):
        TrainLabels[i, ytrain[i]] = 1

    TestLabels = np.zeros((len(ytest), 10))
    for i in range(0, len(ytest)):
        TestLabels[i, ytest[i]] = 1

    # print("Data set: CIFAR10 ")
    # print(" train data shape:  ", TrainInput.shape)
    # print(" train label shape: ", TrainLabels.shape)
    # print(" test data shape:   ", TestInput.shape)
    # print(" test label shape:  ", TestLabels.shape)

    Xtrain, Ytrain, Xtest, Ytest, nclasses = np.ascontiguousarray(TrainInput), np.ascontiguousarray(TrainLabels), np.ascontiguousarray(TestInput), np.ascontiguousarray(
        TestLabels), 10

    # Xtrain, Ytrain, Xtest, Ytest, nclasses = SetLocalCIFAR10('data/cifar-10-python/')
    data = Xtrain[:ss], Ytrain[:ss], Xtest[:ss], Ytest[:ss], nclasses
    return data
