import numpy as np


def SetMyData(datatype, w=1):
    if "CIFAR" == datatype:
        data = PrepareCIFAR10Data()
        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, nclasses = SplitDataTrnValTst(data)
        return Xtrain * w, Ytrain, Xval * w, Yval, Xtest * w, Ytest, nclasses

    if "MNIST" == datatype:
        data = PrepareMNISTData()
        imx = data[0].shape[1]
        imy = data[0].shape[2]
        nchannels = data[0].shape[3]

        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, nclasses = SplitDataTrnValTst(data)

        Xtrain = Xtrain.reshape(-1, imx * imy * nchannels)
        Xval = Xval.reshape(-1, imx * imy * nchannels)
        Xtest = Xtest.reshape(-1, imx * imy * nchannels)

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
