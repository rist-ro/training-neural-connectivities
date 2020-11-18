import glob
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


def makelistoffiles(mypath, pattern):
    netfiles = glob.glob(mypath + pattern)

    for i in range(len(netfiles)):
        netfiles[i] = netfiles[i].replace('\\', "/")

    return netfiles


def MergeTrainLogs(mypath):
    allfiles = makelistoffiles(mypath, "TrainLogs*.pkl")

    Log0 = pickle.load(open(allfiles[0], "rb"))
    nepochs = Log0["trainLoss"].shape[-1]
    nruns = len(list(allfiles))
    nlayers = len(pickle.load(open(allfiles[0], "rb"))["remainingWeightsPerLayer"][0])

    TstAcc = np.zeros((nepochs, nruns))
    TrnAcc = np.zeros((nepochs, nruns))
    ValAcc = np.zeros((nepochs, nruns))

    TstLoss = np.zeros((nepochs, nruns))
    TrnLoss = np.zeros((nepochs, nruns))
    ValLoss = np.zeros((nepochs, nruns))

    RemainingWeights = np.zeros((nepochs, nruns))
    RemainingWeightsPerLayer = np.zeros((nepochs, nruns, nlayers, 6))

    for i, file in enumerate(allfiles):
        Logs = pickle.load(open(file, "rb"))

        TrnLoss[:, i] = Logs["trainLoss"]
        ValLoss[:, i] = Logs["valLoss"]
        TstLoss[:, i] = Logs["testLoss"]

        TrnAcc[:, i] = Logs["trainAccuracy"]
        ValAcc[:, i] = Logs["valAccuracy"]
        TstAcc[:, i] = Logs["testAccuracy"]

        RemainingWeights[:, i] = Logs["remainingWeights"]
        RemainingWeightsPerLayer[:, i, ...] = np.asarray(Logs["remainingWeightsPerLayer"])

    np.save(mypath + "MergedTrainAcc.npy", TrnAcc)
    np.save(mypath + "MergedValAcc.npy", ValAcc)
    np.save(mypath + "MergedTestAcc.npy", TstAcc)
    np.save(mypath + "MergedTrainLoss.npy", TrnLoss)
    np.save(mypath + "MergedValLoss.npy", ValLoss)
    np.save(mypath + "MergedTestLoss.npy", TstLoss)
    np.save(mypath + "MergedRemainingWeights.npy", RemainingWeights)
    np.save(mypath + "MergedRemainingWeightsPerLayer.npy", RemainingWeightsPerLayer)

    return 0


def PlotAccuracy(mypath):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=100)
    AxAccuracy = axes[0]
    AxSparsity = axes[1]

    trnAcc = np.load(mypath + "MergedTrainAcc.npy")
    tstAcc = np.load(mypath + "MergedTestAcc.npy")
    tstLoss = np.load(mypath + "MergedTestLoss.npy")
    wj = np.load(mypath + "MergedRemainingWeights.npy")
    scale = np.max(wj)

    AxAccuracy.plot(np.mean(tstAcc, axis=1), linewidth=2, c="black")
    AxAccuracy.plot(np.min(tstAcc, axis=1), linewidth=1, c="black", alpha=0.2)
    AxAccuracy.plot(np.max(tstAcc, axis=1), linewidth=1, c="black", alpha=0.2)
    AxAccuracy.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), facecolor="black", alpha=0.1)

    AxSparsity.plot(1 - np.mean(wj, axis=1) / scale, linewidth=2, c="black", )
    AxSparsity.plot(1 - np.min(wj, axis=1) / scale, linewidth=1, c="black", alpha=0.2)
    AxSparsity.plot(1 - np.max(wj, axis=1) / scale, linewidth=1, c="black", alpha=0.2)
    AxSparsity.fill_between(np.arange(wj.shape[0]), 1 - np.min(wj, axis=1) / scale, 1 - np.max(wj, axis=1) / scale, facecolor="black", alpha=0.1)

    AxAccuracy.grid(True)
    AxSparsity.grid(True)
    AxSparsity.set_xlabel("Epochs", fontsize=20)
    AxAccuracy.set_ylabel("Test Accuracy", fontsize=18)
    AxSparsity.set_ylabel("Pruned Weights", fontsize=18)

    AxAccuracy.set_ylim((.90, 1.007))

    fig.tight_layout(pad=1)

    fig.savefig(mypath + "Accuracy_Sparsity.pdf")
    fig.savefig(mypath + "Accuracy_Sparsity.png")
    print("Figures saved in", mypath)

    plt.show()

    return 0


def main():
    mypath = "Outputs/FreePruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/"
    mypath = "Outputs/FreePruning/LeNet/P1_0.5/mask_relu_he_LR0.001/"
    mypath = "Outputs/MaxPruning/LeNet/P1_0.5/mask_relu_he_LR0.001/"
    MergeTrainLogs(mypath)
    PlotAccuracy(mypath)

    return 0


if __name__ == '__main__':
    main()
