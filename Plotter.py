import os, sys

# from RandomNets.RandomNets_keras0 import continuetraining

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from numpy.random import seed
import Utilities as utils
from shutil import copy2
import DataBaseManager, time, math
import pickle
import matplotlib
import socket
import MaskedCNN2
import MaskedCNN3
import random

if socket.gethostname() != "DESKTOP-UBRVFON":
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import localutils

# from tensorflow.keras import backend as KB

np.set_printoptions(threshold=sys.maxsize)

WinRatio = 1.9
WinFormFactor = 0.6
WinWidth = 16 * WinFormFactor
WinHeight = WinRatio * WinWidth * WinFormFactor
WinHeight = 9
WinWidth = 16

dpi_server = 350
dpi_laptop = 100


def makelistoffiles(mypath, pattern):
    import glob
    netfiles = glob.glob(mypath + pattern)

    for i in range(len(netfiles)):
        netfiles[i] = netfiles[i].replace('\\', "/")

    return netfiles


def DPI():
    return dpi_laptop if socket.gethostname() == "DESKTOP-UBRVFON" else dpi_server


def makeselectionby(mypath, lof, n):
    selection = {}
    for file in lof:
        s = file[len(mypath):-4].split("_")[n][2:]
        if s not in selection.keys():
            selection[s] = len(selection.keys()) + 1

    return selection


def PlotWeightMagnitudes(mypath, ID):
    print(mypath)
    lon = makelistoffiles(mypath, "NetworkWeights*.pkl")

    ID = lon[0][len(mypath):-4].split("_")[1][2:]
    SD = lon[0][len(mypath):-4].split("_")[2][2:]

    lof = makelistoffiles(mypath, "TrainLogs*ID" + ID + "*.pkl")
    print(lof)

    unique_ID = makeselectionby(mypath, lof, 3)
    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    print(unique_Wj.keys())
    scaling = 1
    for i in unique_Wj.keys():
        scaling = max(scaling, int(i))

    lom = []
    for f in lof:
        lom.append(mypath + f[len(mypath):].replace("TrainLogs", "Masks"))

    low = makelistoffiles(mypath, "NetworkWeights_ID" + ID + "*.pkl")
    print(low)

    W0 = pickle.load(open(mypath + "NetworkWeights_ID" + ID + "_SD" + SD + ".pkl", "rb"))

    M0 = pickle.load(open(lom[0], "rb"))

    # print(W0[0])

    # for w in W0:
    #     if len(w) > 0:
    #         print(w[0])
    # print(w[0].shape,np.mean(w[0]),np.mean(np.abs(w[0])))

    # print(W0[0][0].shape)
    # plt.hist(W0[0][0].reshape(-1))
    # plt.hist(W0[0][1].reshape(-1))
    # plt.show()
    #
    # input()

    lins, cols = utils.SplitToCompactGrid(len(M0) - 4)
    fig, ax = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI())
    colormap = 'viridis'
    ax = ax.flatten(-1)

    for c, ps in zip(range(len(list(unique_PS.keys()))), unique_PS.keys()):
        lom = makelistoffiles(mypath, "Masks_Wj*ID" + ID + "*PS" + ps + ".pkl")

        for file in lom:
            Masks = pickle.load(open(file, "rb"))
            params = int(file[len(mypath):-4].split("_")[1][2:])
            TrainLogs = pickle.load(open(file.replace("Masks", "TrainLogs"), "rb"))
            # print(len(TrainLogs['testAccuracy']))
            acc = TrainLogs['testAccuracy'][-1]

            # print(params, np.sum(Masks[0]))
            lm = 0
            for m in range(len(W0)):
                if isinstance(W0[m], list):
                    continue

                else:
                    # if len(W0[m]) > 0:
                    # print(np.sum(Masks[lm][0])/Masks[lm][0].size)
                    # print("W",W0[m][0].shape)
                    # print("M",Masks[lm].shape)
                    # print(Masks[lm])
                    W = W0[m]
                    w = W0[m] * Masks[m]
                    # wp = w[w > 0]
                    # wm = w[w < 0]

                    # print(np.mean(W0[m][0]))
                    # w = w[w > 0]
                    m = np.mean(np.abs(w[w != 0]))
                    s = np.std(np.abs(w[w != 0]))

                    #
                    # ax[m].hist(w.reshape(-1), bins=300)
                    # ax[lm].scatter(params / scaling, m/s, c='black')
                    ax[lm].scatter(int(ps), m, c='black')
                    # ax[lm].scatter(int(ps), np.max(np.abs(w[w != 0])), c='red')
                    # ax[lm].scatter(int(ps), np.min(np.abs(w[w != 0])), c='blue')

                    # h = np.histogram(w.reshape(-1))

                    # ax[lm].scatter(params / scaling, np.max(W), c='red')
                    # ax[lm].scatter(params / scaling, np.min(W), c='blue')
                    # ax[lm].scatter(int(ps), acc, c='red')
                    # ax[lm].scatter(m, acc, c='red')
                    # ax[lm].scatter(params/scaling, s, c='red')
                    # ax[lm].scatter(int(ps), np.sum(Masks[lm][0])/Masks[lm][0].size)
                    # ax[lm].scatter(params/scaling, np.sum(Masks[lm][0])/Masks[lm][0].size)

                    lm += 1

        # plt.pause(0.1)
        # fig.savefig(mypath + "AllWeights.png")

    for a in ax:
        a.set_xlabel("% overall weights left")
        a.set_ylabel("% weights left per layer")
        # a.set_xlim((0.5, 1))
        # a.set_ylim((0, 1))

        # a.set_xscale('log')
        # a.set_yscale('log')
        a.grid()

    # plt.show()

    plt.grid(True)
    fig.tight_layout(pad=0)
    fig.savefig(mypath + "AllWeights.png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.close()

    return 0


def PlotMaskEvolution(mypath):
    print(mypath)
    lon = makelistoffiles(mypath, "NetworkWeights*.pkl")

    ID = lon[0][len(mypath):-4].split("_")[1][2:]
    SD = lon[0][len(mypath):-4].split("_")[2][2:]

    print(ID)

    lof = makelistoffiles(mypath, "TrainLogs*ID" + ID + "*_SD" + SD + "*.pkl")
    # print(lof)

    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    # print(unique_Wj.keys())
    maxw = -1
    minw = 100000000

    scaling = 1
    for i in unique_Wj.keys():
        maxw = max(maxw, int(i))
        minw = min(minw, int(i))
        scaling = max(scaling, int(i))

    lom = []
    for f in lof:
        lom.append(mypath + f[len(mypath):].replace("TrainLogs", "Masks"))

    # low = makelistoffiles(mypath, "NetworkWeights_ID" + ID + "*.pkl")
    W0 = pickle.load(open(mypath + "NetworkWeights_ID" + ID + "_SD" + SD + ".pkl", "rb"))
    M0 = pickle.load(open(lom[0], "rb"))

    nonemptylayers = 0

    for m in range(len(W0)):
        if isinstance(W0[m], list):
            continue
        nonemptylayers += 1

    lins, cols = utils.SplitToCompactGrid(len(M0) - 4)
    fig, ax = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI())
    # fig.suptitle(mypath, fontsize=3)

    colormap = 'viridis'
    ax = ax.flatten(-1)

    lom = makelistoffiles(mypath, "Masks_Wj*ID" + ID + "*.pkl")

    # here we have all masks of a particular run (ID)

    # loop thourgh all masks and count the number
    # of nonzero elements for each layer

    nonzeros = np.zeros((nonemptylayers, len(lom)))

    for i, f in enumerate(lom):
        Masks = pickle.load(open(f, "rb"))
        ps = int(f[len(mypath):-4].split("_")[8][2:])

        lm = 0
        for m in range(len(W0)):
            if isinstance(W0[m], list):
                continue
            else:
                nonzeros[lm, ps] = np.count_nonzero(Masks[m]) / Masks[m].size

            lm += 1

    for i in range(len(ax)):
        ax[i].plot(nonzeros[i], color='tab:blue', linestyle=':', linewidth=.5, marker='o', markersize=4, markerfacecoloralt='tab:red')
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("%Weights, Layer " + str(i), fontsize=8)

        # a.set_title("Layer" + str(l), fontsize=8)
        # a.legend(fontsize=5)
        # a.set_xlim((0.5, 1))
        # a.set_xscale('log')
        # a.set_yscale('log')

        ax[i].set_yticks(np.linspace(0., 1, 5))
        ax[i].set_ylim((0, 1.1))
        ax[i].grid(alpha=0.5, linewidth=0.5)

    fig.tight_layout(pad=1)
    fig.savefig(mypath + "1_Mask_Evolution.png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.close()
    return 0


def PlotFinalPercentageOfWeights(mypath):
    print(mypath)
    lon = makelistoffiles(mypath, "NetworkWeights*.pkl")

    ID = lon[0][len(mypath):-4].split("_")[1][2:]
    SD = lon[0][len(mypath):-4].split("_")[2][2:]

    print(ID)

    lof = makelistoffiles(mypath, "TrainLogs*ID" + ID + "*_SD" + SD + "*.pkl")

    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    # print(unique_Wj.keys())
    maxw = -1
    minw = 100000000

    scaling = 1
    for i in unique_Wj.keys():
        maxw = max(maxw, int(i))
        minw = min(minw, int(i))
        scaling = max(scaling, int(i))

    minps = 0
    maxps = -1
    for i in unique_PS.keys():
        maxps = max(maxps, int(i))
    # maxps=0

    # find the mas file with ID and maximum PS
    lom = makelistoffiles(mypath, "Masks*ID" + ID + "*PS" + str(maxps) + ".pkl")

    W0 = pickle.load(open(mypath + "NetworkWeights_ID" + ID + "_SD" + SD + ".pkl", "rb"))
    M0 = pickle.load(open(lom[0], "rb"))

    lom = makelistoffiles(mypath, "Masks*ID" + ID + "*PS" + str(minps) + ".pkl")
    ML = pickle.load(open(lom[0], "rb"))

    # print(M0)
    # input()

    nonemptylayers = 0

    for m in range(len(W0)):
        if isinstance(W0[m], list):
            continue
        nonemptylayers += 1

    lins, cols = utils.SplitToCompactGrid(len(M0) - 4)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI())
    # fig.suptitle(mypath, fontsize=3)

    colormap = 'viridis'
    # ax = ax.flatten(-1)

    points = []
    allweights = []
    maskedweights = []
    lm = 0
    means0 = []
    meansl = []
    stds0 = []
    stdsl = []

    for m in range(len(M0)):
        if isinstance(M0[m], list):
            continue

        # print(W0[m].shape,W0[m].size)
        allweights.extend(W0[m].reshape(-1).tolist())
        maskedweights.extend((W0[m] * M0[m]).reshape(-1).tolist())
        points.append(np.count_nonzero(M0[m]) / M0[m].size)
        print("m0={:f}, s0={:f}, ml={:f}, sl={:f},".format(np.mean(W0[m] * M0[m]), np.std(W0[m] * M0[m]), np.mean(W0[m] * ML[m]),
                                                           np.std(W0[m] * ML[m])))
        # means0.append(np.mean(W0[m] * M0[m]))
        # meansl.append(np.mean(W0[m] * ML[m]))
        # stds0.append(np.std(W0[m] * M0[m]))
        # stdsl.append(np.std(W0[m] * ML[m]))

        means0.append(np.mean(W0[m][M0[m] == 1]))
        meansl.append(np.mean(W0[m][ML[m] == 1]))
        stds0.append(np.std(W0[m][M0[m] == 1]))
        stdsl.append(np.std(W0[m][ML[m] == 1]))
        print(np.mean(W0[m]))

    # ax.plot(np.asarray(points))
    ax.plot(np.asarray(points) * 100, color='tab:blue', linestyle=':', linewidth=2, marker='o', markersize=14, markerfacecoloralt='tab:red')

    # for a in ax:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_yminorticklabels()):
        item.set_fontsize(20)

    fig_allweights, ax_allweights = plt.subplots(figsize=(6, 6), dpi=DPI())
    # print(len(allweights))
    # print(len(maskedweights))

    # ax_allweights.scatter(np.asarray(stds0), np.asarray(stdsl))
    # ax_allweights.scatter(np.asarray(means0), np.asarray(meansl))
    # ax_allweights.scatter(np.abs(np.asarray(means0)), np.abs(np.asarray(meansl)))
    ax_allweights.scatter(np.arange(len(means0)), (np.asarray(means0) / np.asarray(meansl)))
    ax_allweights.scatter(np.arange(len(stds0)), (np.asarray(stds0) / np.asarray(stdsl)))

    # ax.set_title("Layer" + str(l), fontsize=8)
    # ax.legend(fontsize=5)

    # a.set_xlim((0.5, 1))
    ax.set_yticks(np.linspace(40, 100, 7))
    ax.set_ylim((40, 100))

    # a.set_xscale('log')
    # a.set_yscale('log')
    ax.grid(alpha=0.5, linewidth=0.5)
    ax.set_xlabel("Layer ID", fontsize=20)
    ax.set_ylabel("Remaining weights (%)", fontsize=20)

    fig.tight_layout(pad=0.5)
    fig.savefig(mypath + "1_WeightRemainingPerLayer.pdf")

    plt.show()

    return 0


def PlotWeightMagnitudes2(mypath):
    print(mypath)
    lon = makelistoffiles(mypath, "NetworkWeights*.pkl")

    ID = lon[0][len(mypath):-4].split("_")[1][2:]
    SD = lon[0][len(mypath):-4].split("_")[2][2:]

    print(ID)

    lof = makelistoffiles(mypath, "TrainLogs*ID" + ID + "*_SD" + SD + "*.pkl")
    # print(lof)

    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    # print(unique_Wj.keys())
    maxw = -1
    minw = 100000000

    scaling = 1
    for i in unique_Wj.keys():
        maxw = max(maxw, int(i))
        minw = min(minw, int(i))
        scaling = max(scaling, int(i))

    lom = []
    for f in lof:
        lom.append(mypath + f[len(mypath):].replace("TrainLogs", "Masks"))

    # low = makelistoffiles(mypath, "NetworkWeights_ID" + ID + "*.pkl")
    W0 = pickle.load(open(mypath + "NetworkWeights_ID" + ID + "_SD" + SD + ".pkl", "rb"))
    M0 = pickle.load(open(lom[0], "rb"))

    lins, cols = utils.SplitToCompactGrid(len(M0) - 4)
    fig, ax = plt.subplots(lins, cols, figsize=(12, 8), dpi=DPI())
    # fig.suptitle(mypath, fontsize=3)

    colormap = 'viridis'
    ax = ax.flatten(-1)

    epoch = 0
    for c, ps in zip(range(len(list(unique_PS.keys()))), unique_PS.keys()):
        lom = makelistoffiles(mypath, "Masks_Wj*ID" + ID + "*PS" + ps + ".pkl")

        # if (c) % (len(list(unique_PS.keys())) // 2) != 0:
        #     continue

        if c != 0 and c != len(list(unique_PS.keys())) - 1:
            continue

        for file in lom:
            Masks = pickle.load(open(file, "rb"))
            params = int(file[len(mypath):-4].split("_")[1][2:])
            TrainLogs = pickle.load(open(file.replace("Masks", "TrainLogs"), "rb"))
            # print(len(TrainLogs['testAccuracy']))
            acc = TrainLogs['testAccuracy'][-1]

            # print(params, np.sum(Masks[0]))
            lm = 0
            sumall = 0
            nonzeros = np.zeros((len(W0), len(list(unique_PS.keys()))))
            # print(nonzeros.shape)
            # input()

            for m in range(len(W0)):
                if isinstance(W0[m], list):
                    continue
                else:
                    w = W0[m]
                    w = W0[m] * Masks[m]
                    # w = W0[m][Masks[m]]
                    # w[w==0]=np.min(w)
                    # print(Masks[m])
                    # wp = w[w > 0]
                    # wm = w[w < 0]

                    # ax[m].hist(w.reshape(-1), bins=300)
                    # ax[lm].scatter(params / scaling, m/s, c='black')
                    # ax[lm].scatter(int(ps), m, c='black')
                    # ax[lm].scatter(int(ps), np.max(np.abs(w[w != 0])), c='red')
                    # ax[lm].scatter(int(ps), np.min(np.abs(w[w != 0])), c='blue')

                    # ax[lm].grid()
                    if int(ps) == 0:
                        label = "initialization"
                    if int(ps) == len(list(unique_PS.keys())) - 1:
                        label = "last epoch"

                    # ax[lm].scatter(int(ps), np.count_nonzero(Masks[m]) / Masks[m].size, c='red', s=3)
                    # nonzeros.append(np.count_nonzero(Masks[m]) / Masks[m].size)
                    # nonzeros[lm, int(ps)] = np.count_nonzero(Masks[m]) / Masks[m].size
                    # h = np.histogram(w.reshape(-1))
                    # ax[lm].hist(w[w != 0].reshape(-1), histtype='step', label=label, bins=max(60, int(w[w != 0].reshape(-1).shape[-1] / 3000)))
                    # ax[lm].hist(w[w != 0].reshape(-1), alpha=0.5,label=label, bins=max(60, int(w[w != 0].reshape(-1).shape[-1] / 3000)))
                    # ax[lm].hist(w[w != 0].reshape(-1), histtype='step', label=ps)
                    # ax[lm].hist(w.reshape(-1), histtype='step', label=label, bins=max(20, int(w[w != 0].reshape(-1).shape[-1] / 3000)))

                    ax[lm].hist(w.reshape(-1), histtype='step', label=label, linewidth=0.5,
                                bins=200)  # , bins=max(100, int(w.reshape(-1).shape[-1] / 3000)))

                    # ax[lm].hist(w[w!=0].reshape(-1), histtype='step', label=ps)
                    # print(np.sum(Masks[m]))
                    # ax[lm].scatter(int(ps), np.sum(w)/w.size, c='red')
                    # sumall += np.sum(w)

                    # ax[lm].matshow()
                    # print(m)
                    # print(Masks[m].shape)
                    # ax[lm].scatter(params / scaling, np.max(W), c='red')
                    # ax[lm].scatter(params / scaling, np.min(W), c='blue')
                    # ax[lm].scatter(int(ps), acc, c='red')
                    # ax[lm].scatter(m, acc, c='red')
                    # ax[lm].scatter(params/scaling, s, c='red')
                    # ax[lm].scatter(int(ps), np.sum(Masks[lm][0])/Masks[lm][0].size)
                    # ax[lm].scatter(params/scaling, np.sum(Masks[lm][0])/Masks[lm][0].size)

                    lm += 1

        epoch += 1

        # print(sumall)
        # plt.pause(0.1)
        # fig.savefig(mypath + "AllWeights.png")

    for a, l in zip(ax, range(len(ax))):
        # a.set_xlabel("Weight value")
        # a.set_ylabel("Entries in layer " + str(l), fontsize=8)
        a.set_title("Layer" + str(l), fontsize=8)
        a.legend(fontsize=5)

        # a.set_xlim((0.5, 1))
        # a.set_yticks(np.linspace(0., 1, 5))
        # a.set_ylim((0, 1.1))

        # a.set_xscale('log')
        a.set_yscale('log')
        a.grid(alpha=0.5, linewidth=0.5)

    for a in ax.reshape(-1):
        for item in ([a.title, a.xaxis.label, a.yaxis.label] + a.get_xticklabels() + a.get_yticklabels() + a.get_yminorticklabels()):
            item.set_fontsize(8)

    fig.tight_layout(pad=0.5)
    fig.savefig(mypath + "1_Weights_Evolution.png")
    fig.savefig(mypath + "1_Weights_Evolution.pdf")

    # fig.savefig(mypath + "1_Weights_Evolution_NonZero.png")
    # fig.savefig(mypath + "1_Weights_Evolution_NonZero.pdf")
    # fig.savefig(mypath + "1_Weights_Scatter_BeforeAfter.png")
    # fig.savefig(mypath + "1_Weights_Scatter_BeforeAfter.pdf")

    # if socket.gethostname() == "DESKTOP-UBRVFON":
    #     plt.show()

    plt.close()

    return 0


def MergeFinalTrainLogs(mypath):
    print(mypath)
    allfiles = makelistoffiles(mypath, "TrainLogs*.pkl")

    Log0 = pickle.load(open(allfiles[0], "rb"))
    nepochs = Log0["trainLoss"].shape[-1]
    nruns = len(list(allfiles))
    print(nruns, nepochs)

    TstAcc = np.zeros((nepochs, nruns))
    TrnAcc = np.zeros((nepochs, nruns))
    ValAcc = np.zeros((nepochs, nruns))

    TstLoss = np.zeros((nepochs, nruns))
    TrnLoss = np.zeros((nepochs, nruns))
    ValLoss = np.zeros((nepochs, nruns))

    RemainingWeights = np.zeros((nepochs, nruns))

    for i, file in enumerate(allfiles):
        Logs = pickle.load(open(file, "rb"))

        TrnLoss[:, i] = Logs["trainLoss"]
        ValLoss[:, i] = Logs["valLoss"]
        TstLoss[:, i] = Logs["testLoss"]

        TrnAcc[:, i] = Logs["trainAccuracy"]
        ValAcc[:, i] = Logs["valAccuracy"]
        TstAcc[:, i] = Logs["testAccuracy"]

        RemainingWeights[:, i] = Logs["remainingWeights"]

    np.save(mypath + "MergedTrainAcc.npy", TrnAcc)
    np.save(mypath + "MergedValAcc.npy", ValAcc)
    np.save(mypath + "MergedTestAcc.npy", TstAcc)
    np.save(mypath + "MergedTrainLoss.npy", TrnLoss)
    np.save(mypath + "MergedValLoss.npy", ValLoss)
    np.save(mypath + "MergedTestLoss.npy", TstLoss)
    np.save(mypath + "MergedRemainingWeights.npy", RemainingWeights)

    print(TrnAcc)
    print(RemainingWeights)

    return 0


def MergeTrainLogs(mypath):
    print(mypath)
    lof = makelistoffiles(mypath, "TrainLogs*.pkl")
    # lof = makelistoffiles(mypath, "TransferTrainLogs*.pkl")
    # print(lof)
    # input()

    unique_ID = makeselectionby(mypath, lof, 3)
    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    # how many runs were done
    nruns = len(list(unique_ID))
    nepochs = len(list(unique_PS))
    nlayers = len(pickle.load(open(lof[0], "rb"))["remainingWeigtsPerLayer"][0])
    print("nlyaers", nlayers)
    # input()

    for i in unique_Wj.keys():
        scaling = max(scaling, int(i))

    print(nruns, nepochs)

    TstAcc = np.zeros((nepochs, nruns))
    TrnAcc = np.zeros((nepochs, nruns))
    ValAcc = np.zeros((nepochs, nruns))

    TstLoss = np.zeros((nepochs, nruns))
    TrnLoss = np.zeros((nepochs, nruns))
    ValLoss = np.zeros((nepochs, nruns))

    RemainingWeights = np.zeros((nepochs, nruns))
    RemainingWeightsPerLayer = np.zeros((nepochs, nruns, nlayers, 6))

    # run over pruning steps (epochs)
    for ps in unique_PS.keys():

        # run over all runs (IDs)
        for i, id in zip(np.arange(nruns), unique_ID.keys()):

            lot = makelistoffiles(mypath, "TrainLogs_Wj*ID" + id + "*PS" + ps + ".pkl")

            for file in lot:
                Logs = pickle.load(open(file, "rb"))
                params = float(file[len(mypath):-4].split("_")[1][2:])

                epoch = int(ps)
                # print(int(ps), i)

                TrnLoss[epoch, i] = Logs["trainLoss"][-1]
                ValLoss[epoch, i] = Logs["valLoss"][-1]
                TstLoss[epoch, i] = Logs["testLoss"][-1]

                TrnAcc[epoch, i] = Logs["trainAccuracy"][-1]
                ValAcc[epoch, i] = Logs["valAccuracy"][-1]
                TstAcc[epoch, i] = Logs["testAccuracy"][-1]
                # print(ps,id,Logs["trainAccuracy"][-1])

                RemainingWeights[int(ps), i] = params  # / scaling
                RemainingWeightsPerLayer[epoch, i, ...] = np.asarray(Logs["remainingWeigtsPerLayer"][epoch])

    # for f in lom:
    #     Logs = pickle.load(open(f, "rb"))
    #     params = float(f[len(mypath):-4].split("_")[7][2:])
    #
    #     TrnLoss.append(Logs["trainLoss"])
    #     ValLoss.append(Logs["valLoss"])
    #     TstLoss.append(Logs["testLoss"])
    #     TrnAcc.append(Logs["trainAccuracy"])
    #     ValAcc.append(Logs["valAccuracy"])
    #     TstAcc.append(Logs["testAccuracy"])
    #
    #     print(TstAcc)

    # print(TrnAcc)
    # print(TrnAcc[0])
    # print(TrnAcc[1])
    # TrnAcc[1] = TrnAcc[1][np.argsort(TrnAcc[0])]
    # TrnAcc[0] = TrnAcc[0][np.argsort(TrnAcc[0])]
    #
    #
    # print(TrnAcc[0])
    # print(TrnAcc[1])
    #
    # input()

    # print(TstAcc)
    # input()

    np.save(mypath + "MergedTrainAcc.npy", TrnAcc)
    np.save(mypath + "MergedValAcc.npy", ValAcc)
    np.save(mypath + "MergedTestAcc.npy", TstAcc)
    np.save(mypath + "MergedTrainLoss.npy", TrnLoss)
    np.save(mypath + "MergedValLoss.npy", ValLoss)
    np.save(mypath + "MergedTestLoss.npy", TstLoss)
    np.save(mypath + "MergedRemainingWeights.npy", RemainingWeights)
    np.save(mypath + "MergedRemainingWeightsPerLayer.npy", RemainingWeightsPerLayer)

    # np.save(mypath + "MergedTransferTrainAcc.npy", TrnAcc)
    # np.save(mypath + "MergedTransferValAcc.npy", ValAcc)
    # np.save(mypath + "MergedTransferTestAcc.npy", TstAcc)
    # np.save(mypath + "MergedTransferTrainLoss.npy", TrnLoss)
    # np.save(mypath + "MergedTransferValLoss.npy", ValLoss)
    # np.save(mypath + "MergedTransferTestLoss.npy", TstLoss)
    # np.save(mypath + "MergedTransferRemainingWeights.npy", RemainingWeights)
    # np.save(mypath + "MergedTransferRemainingWeightsPerLayer.npy", RemainingWeightsPerLayer)
    # print(RemainingWeightsPerLayer.shape)
    # print(RemainingWeightsPerLayer)
    # print(np.mean(RemainingWeightsPerLayer, axis=1))
    # input()

    return 0


def MergeFullRuns(mypath):
    print(mypath)
    lof = makelistoffiles(mypath, "TrainLogs*.pkl")
    # print(lof)

    unique_ID = makeselectionby(mypath, lof, 3)
    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    # how many runs were done
    nruns = len(list(unique_ID))
    # nepochs = len(list(unique_PS))

    Logs0 = pickle.load(open(lof[0], "rb"))
    nepochs = len(Logs0[list(Logs0.keys())[0]])

    scaling = 0
    for i in unique_Wj.keys():
        scaling = max(scaling, int(i))

    # scaling = 2257983

    print(nruns, nepochs)

    TstAcc = []
    TrnAcc = []
    ValAcc = []

    TstLoss = []
    TrnLoss = []
    ValLoss = []

    # RemainingWeights = np.zeros((nepochs, nruns))

    for f in lof:
        Logs = pickle.load(open(f, "rb"))

        TrnLoss.append(Logs["trainLoss"])
        ValLoss.append(Logs["valLoss"])
        TstLoss.append(Logs["testLoss"])
        TrnAcc.append(Logs["trainAccuracy"])
        ValAcc.append(Logs["valAccuracy"])
        TstAcc.append(Logs["testAccuracy"])

    # print(np.vstack(TstAcc).transpose())
    # input()

    np.save(mypath + "MergedTrainAcc.npy", np.vstack(TrnAcc).transpose())
    np.save(mypath + "MergedValAcc.npy", np.vstack(ValAcc).transpose())
    np.save(mypath + "MergedTestAcc.npy", np.vstack(TstAcc).transpose())

    np.save(mypath + "MergedTrainLoss.npy", np.vstack(TrnLoss).transpose())
    np.save(mypath + "MergedValLoss.npy", np.vstack(ValLoss).transpose())
    np.save(mypath + "MergedTestLoss.npy", np.vstack(TstLoss).transpose())

    return 0


def GetBaselinePerformances(mypaths):
    lins, cols = 1, 3
    nplots = len(mypaths)

    if nplots != 3:
        lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 6), dpi=100, sharey=True)
    axes = axes.flatten(-1)

    for p, ax in zip(mypaths, axes):
        trnAcc = np.load(p + "MergedTrainAcc.npy")
        tstAcc = np.load(p + "MergedTestAcc.npy")

        maxman = np.max(np.mean(tstAcc, axis=1))

        ax.plot(np.mean(tstAcc, axis=1), label=p + "/n" + str(maxman), linewidth=1)
        ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.4)
        ax.grid(True)
        ax.legend(fontsize=8, loc='lower right')  # fontsize=5)

    plt.show()
    return


def PlotPruningPercentagesAll(mypath):
    print(mypath)
    lof = makelistoffiles(mypath, "TrainLogs*.pkl")
    # data = localutils.SetMyData("CIFAR")

    unique_ID = makeselectionby(mypath, lof, 3)
    unique_Wj = makeselectionby(mypath, lof, 1)
    unique_PS = makeselectionby(mypath, lof, 8)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    ID_dict = {}
    count = 0
    for i in unique_ID.keys():
        ID_dict[i] = count
        count += 1

    fig_MeanAccs, ax = plt.subplots(figsize=(10, 6), dpi=DPI())
    ax.set_title(mypath)

    print(unique_Wj.keys())
    scaling = 0
    for i in unique_Wj.keys():
        scaling = max(scaling, int(i))
    scaling = 1

    rebuildNet = False

    for ps in unique_PS.keys():
        lot = makelistoffiles(mypath, "TrainLogs_Wj*PS" + ps + ".pkl")
        print(lot)

        clusterx_testset = []
        clustery_testset = []

        clusterx_trainset = []
        clustery_trainset = []

        clusterx_valset = []
        clustery_valset = []

        clusterx_params = []
        clustery_params = []

        for file in lot:
            Logs = pickle.load(open(file, "rb"))
            # id = file[len(mypath):-4].split("_")[3][2:]
            # params = int(file[len(mypath):-4].split("_")[1][2:])
            params = float(file[len(mypath):-4].split("_")[7][2:])

            TrnLoss = Logs["trainLoss"]
            ValLoss = Logs["valLoss"]
            TstLoss = Logs["testLoss"]
            TrnAcc = Logs["trainAccuracy"]
            ValAcc = Logs["valAccuracy"]
            TstAcc = Logs["testAccuracy"]
            epoch = int(ps)

            myseed = int(file[len(mypath):-4].split("_")[5][2:])

            # ax.scatter(epoch, ValLoss[-1], c="black", s=5, alpha=0.1)
            # ax.scatter(epoch, TstLoss[-1], c="tab:cyan", s=5, alpha=0.1)
            # ax.scatter(epoch, TrnLoss[-1], c="tab:purple", s=5, alpha=0.1)

            ax.scatter(epoch, ValAcc[-1], c="black", s=5, alpha=0.2)
            ax.scatter(epoch, TstAcc[-1], c="tab:cyan", s=5, alpha=0.2)
            ax.scatter(epoch, TrnAcc[-1], c="tab:purple", s=5, alpha=0.2)

            clusterx_trainset.append(epoch)
            clustery_trainset.append(TrnAcc[-1])

            clusterx_testset.append(epoch)
            clustery_testset.append(TstAcc[-1])

            clusterx_valset.append(epoch)
            clustery_valset.append(ValAcc[-1])

            clusterx_params.append(epoch)
            clustery_params.append(params / scaling)

        clusterxavg = np.mean(np.asarray(clusterx_testset))
        clusterxstd = np.std(np.asarray(clusterx_testset))
        clusteryavg = np.mean(np.asarray(clustery_testset))
        clusterystd = np.std(np.asarray(clustery_testset))
        # yerr[condition], label = l, marker = markers[c], linewidth = .5, markersize = 2, c = colors[c], linestyle = styles[0])

        ax.errorbar(clusterxavg, clusteryavg, xerr=clusterxstd, yerr=clusterystd,
                    marker='o', linewidth=2, markersize=3, c='tab:blue', label="test set", alpha=0.95)

        clusterxavg = np.mean(np.asarray(clusterx_valset))
        clusterxstd = np.std(np.asarray(clusterx_valset))
        clusteryavg = np.mean(np.asarray(clustery_valset))
        clusterystd = np.std(np.asarray(clustery_valset))

        ax.errorbar(clusterxavg, clusteryavg, xerr=clusterxstd, yerr=clusterystd,
                    marker='o', linewidth=1, markersize=3, c='black', label="validation set", alpha=0.95)

        clusterxavg = np.mean(np.asarray(clusterx_trainset))
        clusterxstd = np.std(np.asarray(clusterx_trainset))
        clusteryavg = np.mean(np.asarray(clustery_trainset))
        clusterystd = np.std(np.asarray(clustery_trainset))

        ax.errorbar(clusterxavg, clusteryavg, xerr=clusterxstd, yerr=clusterystd,
                    marker='o', linewidth=1, markersize=3, c='red', label="train set", alpha=0.95)

        clusterxavg = np.mean(np.asarray(clusterx_params))
        clusterxstd = np.std(np.asarray(clusterx_params))
        clusteryavg = np.mean(np.asarray(clustery_params))
        clusterystd = np.std(np.asarray(clustery_params))

        ax.errorbar(clusterxavg, clusteryavg, xerr=clusterxstd, yerr=clusterystd,
                    marker='o', linewidth=1, markersize=3, c='green', label="parameters", alpha=0.95)

    # ax.legend(fontsize=8)
    ax.set_yticks(np.linspace(0, 1, 21))
    # ax.set_xscale('log')
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy/parameters")
    ax.set_ylim((0, 1))
    plt.grid(True)
    fig_MeanAccs.tight_layout(pad=0)
    fig_MeanAccs.savefig(mypath + "AllAcc.png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.close()

    return 0


def PlotMergedFiles(mypaths, fnames, savepath, savename):
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI())
    # ax.set_title(mypath)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    pc = 0
    for p in mypaths:
        c = 0
        for n in fnames:
            f = np.load(p + n)
            if c % 2 == 1:
                ax.plot(np.mean(f, axis=1), label=p + n, c=colors[pc % 10], linewidth=2)
            else:
                ax.plot(np.mean(f, axis=1), label=p + n, c=colors[pc % 10], linewidth=2, linestyle='--')

            ax.fill_between(np.arange(f.shape[0]), np.min(f, axis=1), np.max(f, axis=1), alpha=0.1, facecolor=colors[pc % 10])
            # ax.fill_between(np.arange(f.shape[0]), np.mean(f, axis=1)-np.std(f, axis=1)/2, np.mean(f, axis=1)+np.std(f, axis=1)/2, alpha=0.1, facecolor=colors[pc % 10])
            c += 1
        pc += 1

    ax.set_yticks(np.linspace(0., 1, 11))
    ax.set_xlabel("epochs", fontsize=28)
    ax.set_ylabel("Accuracy  -  Sparsity", fontsize=28)
    ax.set_ylim((0., 1))
    # ax.legend(fontsize=5)
    ax.grid(True)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_yminorticklabels()):
        item.set_fontsize(20)

    fig.tight_layout(pad=0)
    fig.savefig(savepath + savename)

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.close()

    return 0


def PlotAccForDiffInitializations(mypaths, axtitles, saveas=None):
    nplots = len(mypaths)

    lins, cols = 1, 3

    if nplots != 3:
        lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 6), dpi=DPI(), sharey=True)
    axes = axes.flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    for p, ax, axt in zip(mypaths, axes, axtitles):
        trnAcc = np.load(p + "MergedTrainAcc.npy")
        tstAcc = np.load(p + "MergedTestAcc.npy")
        wj = np.load(p + "MergedRemainingWeights.npy")

        # if c % 2 == 1:
        ax.plot(np.mean(trnAcc, axis=1), label="Traindata", c='tab:blue', linewidth=1)
        ax.fill_between(np.arange(trnAcc.shape[0]), np.min(trnAcc, axis=1), np.max(trnAcc, axis=1), alpha=0.4, facecolor="tab:blue")

        ax.plot(np.mean(tstAcc, axis=1), label="Testdata", c='tab:green', linewidth=1)
        ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.4, facecolor="tab:green")

        ax.plot(np.mean(wj, axis=1), label="%weights remaining", c='tab:red', linewidth=2)
        ax.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1), np.max(wj, axis=1), alpha=0.4, facecolor="tab:red")

        ax.set_title(axt, fontsize=18)
        ax.set_xlabel("Epochs", fontsize=18)

        ax.set_yticks(np.linspace(0., 1, 11))
        ax.set_ylim((0., 1))
        ax.grid(True)

        # else:
        #     ax.plot(np.mean(f, axis=1), label=p + n, c=colors[pc % 10], linewidth=1, linestyle='--')
        #
    axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    axes[0].set_ylabel("Accuracy,  %weights remaining", fontsize=18)

    # fig_MeanAccs, ax = plt.subplots(figsize=(10, 6), dpi=DPI())
    # ax.set_title(mypath)

    fig.tight_layout(pad=1)

    # savepath = "Outputs/ZMI/Plots/"
    # savename = "Initializations.pdf"

    if saveas is not None:
        fig.savefig(saveas)

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.close()

    return 0


def PlotAccVsP1(data, labels, saveas=None):
    lins = 2
    cols = 1
    Fig, Axes = plt.subplots(lins, cols, figsize=(9, 9), dpi=DPI(), sharey=False, sharex=True)
    # Fig, Axes = plt.subplots(lins, cols, figsize=(9,5), dpi=DPI(), sharey=False, sharex=True)

    ncurves = len(data)
    # ndatapoints = len(data[0])

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # colors = colors[:ncurves]

    for i in range(ncurves):
        data_i = data[i]
        curve_test_acc = np.zeros((len(data_i), 7))
        curve_train_acc = np.zeros((len(data_i), 7))

        for j in range(len(data_i)):
            # read path and p1 value
            N = -1
            path = data_i[j][0]
            trnAcc = np.load(path + "MergedTrainAcc.npy")[:N]
            trnAcc = np.load(path + "MergedTrainAcc.npy")[:N]
            tstAcc = np.load(path + "MergedTestAcc.npy")[:N]
            tstLoss = np.load(path + "MergedTestLoss.npy")[:N]
            wj = np.load(path + "MergedRemainingWeights.npy")[:N]
            wjpl = wj
            if os.path.exists(path + "MergedRemainingWeightsPerLayer.npy"):
                wjpl = np.load(path + "MergedRemainingWeightsPerLayer.npy")[:N]
                # print(path, wjpl.shape)
                x = 1 - data_i[j][1]
                # y = np.abs(wjpl[-1, 0, 0, 3] - wjpl[-1, 0, 0, 5]) / np.sum(wjpl[-1, 0, 0, 3:-1])
                # Axes[1].scatter(x,y,c="red")
                # Axes[1].scatter(1 - data_i[j][1], wjpl[-1, 0, 1, 3] / np.sum(wjpl[-1, 0, 1, 3:-1]), c="green")
                # Axes[1].scatter(1 - data_i[j][1], wjpl[-1, 0, 1, 5] / np.sum(wjpl[-1, 0, 1, 3:-1]), c="red")
                # Axes[1].scatter(1 - data_i[j][1], wjpl[-1, 0, 1, 4] / np.sum(wjpl[-1, 0, 1, 3:-1]), c="black")
                # Axes[1].scatter(1-data_i[j][1], wjpl[-1, 0, 2, 3] / np.max(wj),c="black")
                # print(wjpl[:,0,0,5])

            p1 = data_i[j][1]

            # tstAcc=trnAcc-tstAcc

            meantest_curve = np.mean(tstAcc, axis=1)
            amax = np.argmax(meantest_curve)

            avgv = np.mean(tstAcc[amax])
            maxv = np.max(tstAcc[amax])
            minv = np.min(tstAcc[amax])

            # curve_test_acc.append([1 - p1, avgv, minv, maxv])
            curve_test_acc[j, 0] = 1 - p1
            curve_test_acc[j, 1] = avgv
            curve_test_acc[j, 2] = minv
            curve_test_acc[j, 3] = maxv
            curve_test_acc[j, 4] = np.mean(wj[amax]) / np.max(wj)
            curve_test_acc[j, 5] = np.max(wj[amax]) / np.max(wj)
            curve_test_acc[j, 6] = np.min(wj[amax]) / np.max(wj)

            curve_train_acc[j, 0] = 1 - p1
            curve_train_acc[j, 1] = np.mean(trnAcc[np.argmax(np.mean(trnAcc, axis=1))])
            curve_train_acc[j, 2] = np.min(trnAcc[np.argmax(np.mean(trnAcc, axis=1))])
            curve_train_acc[j, 3] = np.max(trnAcc[np.argmax(np.mean(trnAcc, axis=1))])

            # curve_test_acc[j, 4] = np.mean(wj[amax]) / np.max(wj)
            # curve_test_acc[j, 5] = np.max(wj[amax]) / np.max(wj)
            # curve_test_acc[j, 6] = np.min(wj[amax]) / np.max(wj)

            # Axes.plot(meantest_curve, c=colors[i], linewidth=2)
            # Axes.plot(np.min(tstAcc, axis=1), c=colors[i], alpha=0.3)
            # Axes.plot(np.max(tstAcc, axis=1), c=colors[i], alpha=0.3)
            # print(meantest_curve)

            # Axes.scatter(1 - p1, np.max(meantest_curve), c='red',s=10)
            # Axes.scatter(1 - p1, maxv, c='blue')
            # Axes.scatter(amax, avgv, c=colors[i])
            # Axes.scatter(amax, maxv, c=colors[i])
            # Axes.scatter(amax, minv, c=colors[i])

        Axes[0].plot(np.asarray(curve_test_acc)[:, 0], np.asarray(curve_test_acc)[:, 1], c=colors[i], linewidth=2, marker="o", label=labels[i])
        Axes[0].plot(np.asarray(curve_test_acc)[:, 0], np.asarray(curve_test_acc)[:, 2], c=colors[i], linewidth=1, alpha=0.2)
        Axes[0].plot(np.asarray(curve_test_acc)[:, 0], np.asarray(curve_test_acc)[:, 3], c=colors[i], linewidth=1, alpha=0.2)
        Axes[0].fill_between(np.asarray(curve_test_acc)[:, 0], np.asarray(curve_test_acc)[:, 3], np.asarray(curve_test_acc)[:, 2], alpha=0.1, facecolor=colors[i])

        # Axes[0].plot(np.asarray(curve_train_acc)[:, 0], np.asarray(curve_train_acc)[:, 1], c=colors[i], linestyle="-.", linewidth=1, marker="^")  # , label="Train")
        # Axes[0].plot(np.asarray(curve_train_acc)[:, 0], np.asarray(curve_train_acc)[:, 2], c=colors[i], linestyle="-.", linewidth=0.5, alpha=0.2)
        # Axes[0].plot(np.asarray(curve_train_acc)[:, 0], np.asarray(curve_train_acc)[:, 3], c=colors[i], linestyle="-.", linewidth=0.5, alpha=0.2)
        # Axes[0].fill_between(np.asarray(curve_train_acc)[:, 0], np.asarray(curve_train_acc)[:, 3], np.asarray(curve_train_acc)[:, 2], alpha=0.1, facecolor=colors[i])

        Axes[1].plot(np.asarray(curve_test_acc)[:, 0], 1 - np.asarray(curve_test_acc)[:, 4], c=colors[i], linewidth=2, marker="o", label=labels[i])
        Axes[1].plot(np.asarray(curve_test_acc)[:, 0], 1 - np.asarray(curve_test_acc)[:, 5], c=colors[i], linewidth=1, alpha=0.2)
        Axes[1].plot(np.asarray(curve_test_acc)[:, 0], 1 - np.asarray(curve_test_acc)[:, 6], c=colors[i], linewidth=1, alpha=0.2)
        Axes[1].fill_between(np.asarray(curve_test_acc)[:, 0], 1 - np.asarray(curve_test_acc)[:, 5], 1 - np.asarray(curve_test_acc)[:, 6], alpha=0.1, facecolor=colors[i])
        # print(np.asarray(curve_test_acc).shape)
        # print(path, p1)

    Axes[0].grid(True)
    Axes[0].legend(fontsize=15)  # , loc='lower left')
    # Axes[0].legend(fontsize=15,bbox_to_anchor=(0.1, .02, .82, .5), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)#, loc='lower left')
    # Axes[0].set_xlabel("probability of chosing a positive value", fontsize=15)
    Axes[0].set_ylabel("Accuracy", fontsize=15)
    # Axes[0].set_xticks(np.log(np.asarray(curve_test_acc)[:, 0]))
    # Axes[0].set_yticks(np.linspace(0.88, 1, 6))
    # Axes[0].set_ylim((0.8805, 1.01))

    # Axes[0].set_yticks(np.linspace(0.90, 0.98, 5))
    # Axes[0].set_ylim((0.883, 0.987))

    Axes[0].set_xscale('log')
    # Axes[0].set_yscale('log')

    Axes[1].grid(True)
    # Axes[1].legend(fontsize=15,bbox_to_anchor=(0.1, .02, .82, .5), loc='lower center', ncol=3, mode="expand", borderaxespad=0.)#, loc='lower left')

    Axes[1].set_xlabel("Probability of chosing a positive value", fontsize=15)
    Axes[1].set_ylabel("Pruned Weights", fontsize=15)
    # Axes[1].set_xticks(np.asarray(curve_test_acc)[:, 0])
    Axes[1].set_xscale('log')
    # Axes[1].set_yscale('log')

    Fig.tight_layout(pad=0.5)

    if saveas is not None:
        Fig.savefig(saveas + ".pdf")
        Fig.savefig(saveas + ".png")

    plt.show()

    return 0


def PlotAccVsWj(mypaths, axtitles, plotlabels, baseline=None, saveas=None):
    nplots = len(mypaths)

    lins, cols = 1, 3

    if nplots != 3:
        lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 6), dpi=DPI(), sharey=True, sharex=True)
    axes = axes.flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    colors = colors[:len(mypaths[0])] * 100

    styles = ["-", "-.", "--", ":", "-"]
    styles = ["-", "-", "-", "-", "-"]
    styles = styles[:len(mypaths[0])] * 100
    # random.shuffle(styles)
    # random.shuffle(colors)
    # styles = ["-"] * len(mypaths[0]) * 100

    scale = 2261184
    scale = 256476
    # 256476

    for p1, ax, axt1, PL in zip(mypaths, axes, axtitles, plotlabels):

        for p, lstyle, pl, lc in zip(p1, styles, PL, colors):
            trnAcc = np.load(p + "MergedTrainAcc.npy")
            tstAcc = np.load(p + "MergedTestAcc.npy")
            tstLoss = np.load(p + "MergedTestLoss.npy")
            wj = np.load(p + "MergedRemainingWeights.npy")
            # print(tstAcc)

            # tstAcc = tstAcc[:100]
            # trnAcc = trnAcc[:100]
            # wj = wj[:100]
            # tstLoss=tstLoss[:100]
            #
            # ax.plot(np.mean(trnAcc, axis=1), label="trainAcc_" + pl, c=lc, linewidth=1, linestyle=lstyle)
            # ax.fill_between(np.arange(trnAcc.shape[0]), np.min(trnAcc, axis=1), np.max(trnAcc, axis=1), alpha=0.4, facecolor=lc)
            #
            ax.plot(np.mean(tstAcc, axis=1), label=pl[0:], c=lc, linewidth=3, linestyle=lstyle)
            ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #
            #
            # # ax.scatter(np.arange(len(np.mean(tstAcc, axis=1))), np.mean(tstAcc, axis=1), label=pl[:], c=lc, linewidth=2, linestyle=lstyle)
            #
            # # ax.plot(np.mean(tstLoss, axis=1), label=pl[8:-4], c=lc, linewidth=1, linestyle=lstyle)
            # # ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #
            scale = np.max(wj)
            ax.plot(np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle="--")
            ax.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1) / scale, np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)

            # ax.scatter(np.mean(wj, axis=1) / scale,np.mean(tstAcc, axis=1),label=pl[0:])

            # ax.set_xlabel("Epochs", fontsize=18)

        ax.set_title(axt1, fontsize=18)
        # ax.set_yticks(np.linspace(0.925, 0.985, 11))
        # ax.set_ylim((0.925, 0.985))

        ax.set_yticks(np.linspace(0.5, 1., 11))
        ax.set_ylim((0.5, 1.01))

        # ax.set_yticks(np.linspace(0.93, 0.98, 11))
        # ax.set_ylim((0.929, 0.981))

        # ax.set_yticks(np.linspace(0.82, 0.93, 11))
        # ax.set_ylim((0.82, 0.93))

        # ax.set_yticks(np.linspace(0.48, 0.81, 11))
        # ax.set_ylim((0.48, 0.81))

        ax.grid(True)
        ax.legend(fontsize=8, loc='right')  # fontsize=5)

        # ax.set_yscale('log',basey=2)

    if baseline:
        for ax, bl in zip(axes, baseline):
            ax.plot((0, 25), (bl, bl), c="black")
            ax.set_xlabel("Epochs", fontsize=18)
            # ax.legend(fontsize=18, loc='lower right')

    # axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    axes[0].set_ylabel("Test Accuracy", fontsize=18)
    # axes[0].set_ylabel("Accuracy  -  Sparsity", fontsize=18)
    # axes[1].legend(fontsize=18, loc='lower right')
    # fig_MeanAccs, ax = plt.subplots(figsize=(10, 6), dpi=DPI())
    # ax.set_title(mypath)

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas)

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    plt.show()

    return 0


def PlotAccAndSparsityPerEpoch(mypaths, axtitles, plotlabels, accylims=None, sparylims=None, saveas=None):
    nplots = len(mypaths)

    lins, cols = 2, 3

    # if nplots != 3:
    #     lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 7), dpi=DPI(), sharey=False, sharex=True)
    AxAccuracy = axes[0, :].flatten(-1)
    AxSparsity = axes[1, :].flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    colors = colors[:len(mypaths[0])] * 100

    styles = ["-", "-.", "--", ":", "-"]
    styles = ["-", "-", "-", "-", "-"]
    styles = styles[:len(mypaths[0])] * 100
    # random.shuffle(styles)
    # random.shuffle(colors)
    # styles = ["-"] * len(mypaths[0]) * 100

    scale = 2261184
    scale = 256476
    # 256476

    for p1, axacc, axt1, PL, axspar in zip(mypaths, AxAccuracy, axtitles, plotlabels, AxSparsity):

        for p, lstyle, pl, lc in zip(p1, styles, PL, colors):
            trnAcc = np.load(p + "MergedTrainAcc.npy")
            tstAcc = np.load(p + "MergedTestAcc.npy")
            tstLoss = np.load(p + "MergedTestLoss.npy")
            wj = np.load(p + "MergedRemainingWeights.npy")
            # wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")
            # print(tstAcc)

            # tstAcc = tstAcc[:100]
            # trnAcc = trnAcc[:100]
            # tstLoss = tstLoss[:100]

            # wj = wj[:100]
            # wjpl = wjpl[:100]
            #
            # axacc.plot(np.mean(trnAcc, axis=1), label="trainAcc_" + pl, c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(trnAcc.shape[0]), np.min(trnAcc, axis=1), np.max(trnAcc, axis=1), alpha=0.4, facecolor=lc)
            #
            axacc.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=3, linestyle=lstyle)
            axacc.plot(np.min(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axacc.plot(np.max(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)

            meancurve = np.mean(tstAcc, axis=1)
            mincurve = np.min(tstAcc, axis=1)
            maxcurve = np.max(tstAcc, axis=1)

            posmax = np.argmax(meancurve)
            print(axt1, pl, "mean {:.2f}, min {:.2f}, max {:.2f}, pm=".format(meancurve[posmax] * 100, mincurve[posmax] * 100, maxcurve[posmax] * 100),
                  meancurve[posmax] * 100 - mincurve[posmax] * 100)
            #
            #
            # # axacc.scatter(np.arange(len(np.mean(tstAcc, axis=1))), np.mean(tstAcc, axis=1), label=pl[:], c=lc, linewidth=2, linestyle=lstyle)
            #
            # axacc.plot(np.mean(tstLoss, axis=1), label=pl[8:-4], c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #

            axacc.grid(True)
            # if axt1 == "He Initialization":
            axacc.legend(fontsize=16, loc='lower right')  # fontsize=5)

            scale = np.max(wj)
            # scalepl=np.max(wjpl[:,])
            # print(np.mean(wjpl, axis=1).shape)
            # print(np.mean(wjpl, axis=1)[:,0,2])
            # axspar.plot(np.mean(wjpl, axis=1)[:,0,2], c=lc, linewidth=2, linestyle=lstyle)
            axspar.plot(np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle=lstyle)
            axspar.plot(np.min(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axspar.plot(np.max(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axspar.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1) / scale, np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)
            # axspar.set_aspect(300)

            # axacc.scatter(np.mean(wj, axis=1) / scale,np.mean(tstAcc, axis=1),label=pl[0:])

            # axacc.set_xlabel("Epochs", fontsize=18)

            axacc.set_title(axt1, fontsize=20)
            # axacc.set_yticks(np.linspace(0.925, 0.985, 11))
            # axacc.set_ylim((0.925, 0.985))

            mlp = False
            cnn = False

            # ylow = accylims[0]
            # yup = accylims[1]

            if accylims != None:
                axacc.set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
                axacc.set_ylim(accylims[1])

            if sparylims != None:
                axspar.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
                axspar.set_ylim(sparylims[1])

            # axspar.set_yscale('log')

            if mlp:
                # axacc.set_yticks(np.linspace(0.934, 0.984, 5))
                # axacc.set_ylim((0.932, .986))

                axacc.set_yticks(np.linspace(0.956, 0.984, 9))
                axacc.set_ylim((0.956, .986))

                axspar.set_yticks(np.linspace(0.655, 1, 9))
                axspar.set_ylim((0.55, 1.02))

            if cnn:
                axacc.set_yticks(np.linspace(0.51, 0.81, 9))
                axacc.set_ylim((0.50, 0.82))

                axspar.set_yticks(np.linspace(0.53, 0.63, 5))
                axspar.set_ylim((0.52, .64))

                axspar.set_yticks(np.linspace(0.53, 1, 5))
                axspar.set_ylim((0.52, 1))

                axspar.set_yticks(np.linspace(0.54, 1, 9))
                axspar.set_ylim((0.54, 1))

            # axacc.set_yticks(np.linspace(0.55, 0.81, 9))
            # axacc.set_ylim((0.55,0.81))

            # axacc.set_yticks(np.linspace(0.93, 0.98, 11))
            # axacc.set_ylim((0.929, 0.981))

            # axacc.set_yticks(np.linspace(0.82, 0.93, 11))
            # axacc.set_ylim((0.82, 0.93))

            axspar.grid(True)
            axspar.set_xlabel("Epochs", fontsize=20)
            # axacc.legend(fontsize=8, loc='lower right')  # fontsize=5)

        # axacc.set_yscale('log',basey=2)

    # axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    AxAccuracy[0].set_ylabel("Test Accuracy", fontsize=20)
    AxSparsity[0].set_ylabel("Sparsity", fontsize=20)

    # axes[0].set_ylabel("Accuracy  -  Sparsity", fontsize=18)
    # axes[1].legend(fontsize=18, loc='lower right')
    # fig_MeanAccs, axacc = plt.subplots(figsize=(10, 6), dpi=DPI())
    # axacc.set_title(mypath)

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas + ".pdf")
        fig.savefig(saveas + ".png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    # plt.show()

    return 0


def PlotAccVsChangedWeightsPerEpoch(mypaths, axtitles, plotlabels, accylims=None, sparylims=None, saveas=None, axlabels=None):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), dpi=DPI(), sharey=False, sharex=True)
    AxAccuracy = axes[0]
    AxSparsity = axes[1]

    trnAcc = np.load(mypaths + "MergedTrainAcc.npy")
    tstAcc = np.load(mypaths + "MergedTestAcc.npy")
    tstLoss = np.load(mypaths + "MergedTestLoss.npy")
    wj = np.load(mypaths + "MergedRemainingWeights.npy")
    scale = np.max(wj)

    meancurve = np.mean(tstAcc, axis=1)
    mincurve = np.min(tstAcc, axis=1)
    maxcurve = np.max(tstAcc, axis=1)

    posmax = np.argmax(meancurve)

    AxAccuracy.plot(np.mean(tstAcc, axis=1), linewidth=3, c="black")
    AxAccuracy.plot(np.min(tstAcc, axis=1), linewidth=1, c="black", alpha=0.2)
    AxAccuracy.plot(np.max(tstAcc, axis=1), linewidth=1, c="black", alpha=0.2)
    AxAccuracy.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1),  facecolor="black", alpha=0.1)

    AxSparsity.plot(np.mean(wj, axis=1) / scale, linewidth=2, c="black", )
    AxSparsity.plot(np.min(wj, axis=1) / scale, linewidth=1, c="black", alpha=0.2)
    AxSparsity.plot(np.max(wj, axis=1) / scale, linewidth=1, c="black", alpha=0.2)
    AxSparsity.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1) / scale, np.max(wj, axis=1) / scale,  facecolor="black", alpha=0.1)

    AxAccuracy.grid(True)
    AxAccuracy.legend(fontsize=15)

    AxAccuracy.set_title(AxAccuracy, fontsize=20)

    AxSparsity.grid(True)
    AxSparsity.set_xlabel("Epochs", fontsize=20)

    axlabels = ("Test Accuracy", "Pruned Weights")

    AxAccuracy.set_ylabel(axlabels[0], fontsize=18)
    AxSparsity.set_ylabel(axlabels[1], fontsize=18)

    # if accylims != None:
    #     AxAccuracy[0].set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
    #     AxAccuracy[0].set_ylim(accylims[1])
    #
    # if sparylims != None:
    #     AxSparsity[0].set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
    #     AxSparsity[0].set_ylim(sparylims[1])

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas + ".pdf")
        fig.savefig(saveas + ".png")

    plt.show()

    return 0


def PlotAccVsChangedWeightsPerEpochWeightless(mypaths, axtitles, plotlabels, accylims=None, sparylims=None, saveas=None):
    nplots = len(mypaths)

    lins, cols = 1, 3

    # if nplots != 3:
    #     lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 4), dpi=DPI(), sharey=False, sharex=True)
    AxAccuracy = axes

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    lenghts = [len(mypaths[0]), len(mypaths[1]), len(mypaths[2])]

    colors = colors[:np.max(np.asarray(lenghts))] * 100

    styles = ["-", "-.", "--", ":"]
    # styles = ["-", "-", "-", "-", "-"]
    styles = styles[:np.max(np.asarray(lenghts))] * 100
    # random.shuffle(styles)
    # random.shuffle(colors)
    # styles = ["-"] * len(mypaths[0]) * 100

    scale = 2261184
    scale = 256476
    # 256476

    for i, p1, axacc, axt1, PL in zip(np.arange(len(mypaths)), mypaths, AxAccuracy, axtitles, plotlabels):

        for p, lstyle, pl, lc in zip(p1, styles, PL, colors):
            trnAcc = np.load(p + "MergedTrainAcc.npy")
            tstAcc = np.load(p + "MergedTestAcc.npy")
            tstLoss = np.load(p + "MergedTestLoss.npy")
            wj = np.load(p + "MergedRemainingWeights.npy")
            scale = np.max(wj)

            # if os.path.exists(p + "MergedTransferTestAcc.npy"):
            #     transfertstAcc = np.load(p + "MergedTransferTestAcc.npy")
            #     axacc.plot(np.mean(transfertstAcc, axis=1), label=pl, c="black", linewidth=2, linestyle=lstyle)
            #     axacc.plot(np.min(transfertstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axacc.plot(np.max(transfertstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axacc.fill_between(np.arange(transfertstAcc.shape[0]), np.min(transfertstAcc, axis=1), np.max(transfertstAcc, axis=1), alpha=0.1, facecolor=lc)

            if os.path.exists(p + "MergedRemainingWeightsPerLayer.npy"):
                wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")

            meancurve = np.mean(tstAcc, axis=1)
            mincurve = np.min(tstAcc, axis=1)
            maxcurve = np.max(tstAcc, axis=1)

            posmax = np.argmax(meancurve)
            print(axt1, pl, "mean {:.2f}, min {:.2f}, max {:.2f}, pm=".format(meancurve[posmax] * 100, mincurve[posmax] * 100, maxcurve[posmax] * 100),
                  meancurve[posmax] * 100 - mincurve[posmax] * 100)

            r, f = 1, -1
            if "Flipping" in pl or "MinFlipping" in pl:
                wj = np.sum(wjpl[:, :, :, 0], axis=2)
                r, f = 0, 1

            # if "Baseline" not in pl:
            #     axspar.plot(r + f * np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle=lstyle)
            #     axspar.plot(r + f * np.min(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.plot(r + f * np.max(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.fill_between(np.arange(wj.shape[0]), r + f * np.min(wj, axis=1) / scale, r + f * np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)

            # wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")
            # print(wj.shape, wjpl.shape)
            # print(np.sum(wjpl[:,:,:,0],axis=2).shape)
            # wj=minus1Masks
            # print(wj)

            # print(wjpl==0)
            # print(tstAcc)

            # tstAcc = tstAcc[:100]
            # trnAcc = trnAcc[:100]
            # tstLoss = tstLoss[:100]

            # wj = wj[:100]
            # wjpl = wjpl[:100]
            #
            # axacc.plot(np.mean(trnAcc, axis=1), label="trainAcc_" + pl, c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(trnAcc.shape[0]), np.min(trnAcc, axis=1), np.max(trnAcc, axis=1), alpha=0.4, facecolor=lc)
            #
            axacc.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=3, linestyle=lstyle)
            axacc.plot(np.min(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axacc.plot(np.max(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #
            #
            # # axacc.scatter(np.arange(len(np.mean(tstAcc, axis=1))), np.mean(tstAcc, axis=1), label=pl[:], c=lc, linewidth=2, linestyle=lstyle)
            # axacc.plot(np.mean(tstLoss, axis=1), label=pl[8:-4], c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #

            axacc.grid(True)
            if i == 1:
                # axacc.legend(fontsize=10, loc='lower right')  # fontsize=5)
                # axacc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                axacc.legend(fontsize=15, bbox_to_anchor=(0.14, .02, .8, .5), loc='lower center', ncol=1, mode="expand", borderaxespad=0.)

            # scalepl=np.max(wjpl[:,])
            # print(np.mean(wjpl, axis=1).shape)
            # print(np.mean(wjpl, axis=1)[:,0,2])
            # axspar.plot(np.mean(wjpl, axis=1)[:,0,2], c=lc, linewidth=2, linestyle=lstyle)
            # if "Baseline" in PL:
            #     axspar.plot(np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle=lstyle)
            #     axspar.plot(np.min(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.plot(np.max(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1) / scale, np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)

            # axspar.set_aspect(300)

            # axacc.scatter(np.mean(wj, axis=1) / scale,np.mean(tstAcc, axis=1),label=pl[0:])

            # axacc.set_xlabel("Epochs", fontsize=18)

            axacc.set_title(axt1, fontsize=20)

            if accylims != None:
                axacc.set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
                axacc.set_ylim(accylims[1])

            # if sparylims != None:
            #     axspar.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
            #     axspar.set_ylim(sparylims[1])

            # if "Glorot" not in axt1:
            #     axacc.set_yticklabels([])
            #     axspar.set_yticklabels([])

            # axspar.set_yscale('log')

            # axspar.grid(True)
            # axspar.set_xlabel("Epochs", fontsize=20)
            # axacc.legend(fontsize=8, loc='lower right')  # fontsize=5)

        # axacc.set_yscale('log',basey=2)

    # axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    AxAccuracy[0].set_ylabel("Test Accuracy", fontsize=20)
    # AxSparsity[0].set_ylabel("Changed Weights", fontsize=20)

    if accylims != None:
        AxAccuracy[0].set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
        AxAccuracy[0].set_ylim(accylims[1])

    # if sparylims != None:
    #     AxSparsity[0].set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
    #     AxSparsity[0].set_ylim(sparylims[1])

    # AxAccuracy[1].set_yticklabels([])
    # AxAccuracy[2].set_yticklabels([])
    # AxSparsity[1].set_yticklabels([])
    # AxSparsity[2].set_yticklabels([])

    # axes[0].set_ylabel("Accuracy  -  Sparsity", fontsize=18)
    # axes[1].legend(fontsize=18, loc='lower right')
    # fig_MeanAccs, axacc = plt.subplots(figsize=(10, 6), dpi=DPI())
    # axacc.set_title(mypath)

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas + ".pdf")
        fig.savefig(saveas + ".png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    # plt.show()

    return 0


def PlotAccVsTransferAccuracy(mypaths, axtitles, plotlabels, accylims=None, sparylims=None, saveas=None):
    nplots = len(mypaths)

    lins, cols = 1, 3

    # if nplots != 3:
    #     lins, cols = utils.SplitToCompactGrid(nplots)

    fig, axes = plt.subplots(lins, cols, figsize=(18, 4), dpi=DPI(), sharey=False, sharex=True)
    AxAccuracy = axes
    # AxSparsity = axes[1, :].flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    lenghts = [len(mypaths[0]), len(mypaths[1]), len(mypaths[2])]

    colors = colors[:np.max(np.asarray(lenghts))] * 100

    styles = ["-", "-.", "--", ":"]
    # styles = ["-", "-", "-", "-", "-"]
    styles = styles[:np.max(np.asarray(lenghts))] * 100
    # random.shuffle(styles)
    # random.shuffle(colors)
    # styles = ["-"] * len(mypaths[0]) * 100

    for i, p1, axacc, axt1, PL in zip(np.arange(len(mypaths)), mypaths, AxAccuracy, axtitles, plotlabels):

        for p, lstyle, pl, lc in zip(p1, styles, PL, colors):
            N = 1
            trnAcc = np.load(p + "MergedTrainAcc.npy")[N:]
            tstAcc = np.load(p + "MergedTestAcc.npy")[N:]
            tstLoss = np.load(p + "MergedTestLoss.npy")[N:]
            wj = np.load(p + "MergedRemainingWeights.npy")[N:]
            scale = np.max(wj)

            transfertrnAcc = np.load(p + "MergedTransferTrainAcc.npy")[N:]
            transfertstAcc = np.load(p + "MergedTransferTestAcc.npy")[N:]

            y1 = np.mean(tstAcc, axis=1)
            y2 = np.mean(transfertstAcc, axis=1)

            x = np.arange(len(y1))
            y = (y1 - y2)
            # y = y2
            axacc.plot(x, y, c=lc, label=pl, linewidth=2, linestyle=lstyle)
            # axacc.plot(x, y2, c=lc, label=pl, linewidth=1, linestyle=lstyle)

            # if os.path.exists(p + "MergedTransferTestAcc.npy"):
            #     transfertstAcc = np.load(p + "MergedTransferTestAcc.npy")
            #     axacc.plot(np.mean(transfertstAcc, axis=1), label=pl, c="black", linewidth=2, linestyle=lstyle)
            #     axacc.plot(np.min(transfertstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axacc.plot(np.max(transfertstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axacc.fill_between(np.arange(transfertstAcc.shape[0]), np.min(transfertstAcc, axis=1), np.max(transfertstAcc, axis=1), alpha=0.1, facecolor=lc)

            # if os.path.exists(p + "MergedRemainingWeightsPerLayer.npy"):
            #     wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")

            # r, f = 1, -1
            # if "Flipping" in pl or "MinFlipping" in pl:
            #     wj = np.sum(wjpl[:, :, :, 0], axis=2)
            #     r, f = 0, 1
            #
            # if "Baseline" not in pl:
            #     axspar.plot(r + f * np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle=lstyle)
            #     axspar.plot(r + f * np.min(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.plot(r + f * np.max(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.fill_between(np.arange(wj.shape[0]), r + f * np.min(wj, axis=1) / scale, r + f * np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)

            # wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")
            # print(wj.shape, wjpl.shape)
            # print(np.sum(wjpl[:,:,:,0],axis=2).shape)
            # wj=minus1Masks
            # print(wj)

            # print(wjpl==0)
            # print(tstAcc)

            # tstAcc = tstAcc[:100]
            # trnAcc = trnAcc[:100]
            # tstLoss = tstLoss[:100]

            # wj = wj[:100]
            # wjpl = wjpl[:100]
            #
            # axacc.plot(np.mean(trnAcc, axis=1), label="trainAcc_" + pl, c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(trnAcc.shape[0]), np.min(trnAcc, axis=1), np.max(trnAcc, axis=1), alpha=0.4, facecolor=lc)
            #
            # axacc.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=3, linestyle=lstyle)
            # axacc.plot(np.min(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            # axacc.plot(np.max(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            # axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #
            #
            # axacc.scatter(np.mean(tstAcc, axis=1), np.mean(transfertstAcc, axis=1), label=pl[:], c=lc)
            # axacc.plot(np.mean(tstLoss, axis=1), label=pl[8:-4], c=lc, linewidth=1, linestyle=lstyle)
            # axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)
            #

            axacc.grid(True)
            # if axt1 == "He Initialization":
            if i == 1:
                axacc.legend(fontsize=15)  # , loc='lower right')  # fontsize=5)
            # axacc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # axacc.legend(fontsize=15, bbox_to_anchor=(0.17, .02, .82, .5), loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
            # axacc.plot((0, 1), (0, 1))

            # scalepl=np.max(wjpl[:,])
            # print(np.mean(wjpl, axis=1).shape)
            # print(np.mean(wjpl, axis=1)[:,0,2])
            # axspar.plot(np.mean(wjpl, axis=1)[:,0,2], c=lc, linewidth=2, linestyle=lstyle)
            # if "Baseline" in PL:
            #     axspar.plot(np.mean(wj, axis=1) / scale, c=lc, linewidth=2, linestyle=lstyle)
            #     axspar.plot(np.min(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.plot(np.max(wj, axis=1) / scale, c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            #     axspar.fill_between(np.arange(wj.shape[0]), np.min(wj, axis=1) / scale, np.max(wj, axis=1) / scale, alpha=0.1, facecolor=lc)

            # axspar.set_aspect(300)

            # axacc.scatter(np.mean(wj, axis=1) / scale,np.mean(tstAcc, axis=1),label=pl[0:])

            # axacc.set_xlabel("Epochs", fontsize=18)

            axacc.set_title(axt1, fontsize=20)
            axacc.set_ylabel("Mask Flipped Signs Accuracy", fontsize=20)
            axacc.set_ylabel("Difference", fontsize=20)
            axacc.set_xlabel("Epochs", fontsize=20)

            if accylims != None:
                axacc.set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
                axacc.set_ylim(accylims[1])

            # if sparylims != None:
            #     axspar.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
            #     axspar.set_ylim(sparylims[1])

            # if "Glorot" not in axt1:
            #     axacc.set_yticklabels([])
            #     axspar.set_yticklabels([])

            # axspar.set_yscale('log')

            # axspar.grid(True)
            # axspar.set_xlabel("Epochs", fontsize=20)
            # axacc.legend(fontsize=8, loc='lower right')  # fontsize=5)

        # axacc.set_yscale('log')
        # axacc.set_xscale('log')#,basey=2)

    # axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    # AxAccuracy[0].set_ylabel("Test Accuracy", fontsize=20)
    # AxSparsity[0].set_ylabel("Changed Weights", fontsize=20)

    # if accylims != None:
    #     AxAccuracy[0].set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
    #     AxAccuracy[0].set_ylim(accylims[1])
    #
    # if sparylims != None:
    #     AxSparsity[0].set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
    #     AxSparsity[0].set_ylim(sparylims[1])

    # AxAccuracy[1].set_yticklabels([])
    # AxAccuracy[2].set_yticklabels([])
    # AxSparsity[1].set_yticklabels([])
    # AxSparsity[2].set_yticklabels([])

    # axes[0].set_ylabel("Accuracy  -  Sparsity", fontsize=18)
    # axes[1].legend(fontsize=18, loc='lower right')
    # fig_MeanAccs, axacc = plt.subplots(figsize=(10, 6), dpi=DPI())
    # axacc.set_title(mypath)

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas + ".pdf")
        fig.savefig(saveas + ".png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    # plt.show()

    return 0


def PlotAccFscan(mypaths, plotlabels, saveas=None):
    nplots = len(mypaths)

    # lins, cols = 1, 3
    #
    # if nplots != 3:
    #     lins, cols = utils.SplitToCompactGrid(nplots)

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=DPI())
    fig, ax = plt.subplots(figsize=(12, 6), dpi=DPI())
    # axes = axes.flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    colors = colors[:len(mypaths[0])] * 10

    # styles = ["-", "-.", "--", ":", "-"]
    # styles = ["-", "-", "-", "-", "-"]
    # styles = styles[:len(mypaths[0])]
    styles = ["-"] * len(mypaths)

    maxacc = []
    Wj = []
    scale = 2261184
    scale = 4300992
    # scale = 266200
    for p, pl, lc in zip(mypaths, plotlabels, colors):
        trnAcc = np.load(p + "MergedTrainAcc.npy")
        tstAcc = np.load(p + "MergedTestAcc.npy")
        tstLoss = np.load(p + "MergedTestLoss.npy")
        trnLoss = np.load(p + "MergedTrainLoss.npy")
        valLoss = np.load(p + "MergedValLoss.npy")
        wj = np.load(p + "MergedRemainingWeights.npy")

        ax.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=1)
        ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)

        # ax.plot(np.mean(tstLoss, axis=1), label=pl, c=lc, linewidth=2)
        # ax.plot(np.mean(trnLoss, axis=1), label=pl, c=lc, linewidth=2)
        # ax.plot(np.mean(valLoss, axis=1), label=pl, c=lc, linewidth=2)
        ax.plot(np.mean(wj, axis=1) / scale, label=pl, c=lc, linewidth=1)
        # ax.fill_between(np.arange(tstAcc.shape[0]), np.min(wj, axis=1)/ scale, np.max(wj, axis=1)/ scale, alpha=0.1, facecolor='red')

        maxacc.append(np.mean(tstAcc, axis=1)[-1])

    ax.set_xlabel("Epochs", fontsize=18)
    # ax.set_yticks(np.linspace(0.965, 0.985, 11))
    # ax.set_ylim((0.965, .985))

    # ax.set_yticks(np.linspace(0.6, 0.71, 11))
    # ax.set_ylim((0.6, .71))

    # ax.set_yticks(np.linspace(0.73, 0.81, 11))
    # ax.set_ylim((0.73, .81))

    # ax.set_yticks(np.linspace(0., 1, 21))
    # ax.set_ylim((0.0, 1))

    ax.grid(True)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylabel("Test Accuracy", fontsize=18)

    fig.tight_layout(pad=0)

    if saveas is not None:
        fig.savefig(saveas)

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    return 0


def PlotEverything(mypaths, plotlabels, saveas=None):
    nplots = len(mypaths)

    # lins, cols = 1, 3
    #
    # if nplots != 3:
    #     lins, cols = utils.SplitToCompactGrid(nplots)

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=DPI())
    fig, ax = plt.subplots(figsize=(12, 5), dpi=DPI())
    # axes = axes.flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    colors = colors[:len(mypaths[0])] * 10

    # styles = ["-", "-.", "--", ":", "-"]
    # styles = ["-", "-", "-", "-", "-"]
    # styles = styles[:len(mypaths[0])]
    styles = ["-"] * len(mypaths)

    maxacc = []
    Wj = []
    scale = 2261184
    scale = 266200

    for p, pl, lc in zip(mypaths, plotlabels, colors):
        trnAcc = np.load(p + "MergedTrainAcc.npy")
        tstAcc = np.load(p + "MergedTestAcc.npy")
        tstLoss = np.load(p + "MergedTestLoss.npy")
        wj = np.load(p + "MergedRemainingWeights.npy")

        ax.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=1)
        # ax.plot(np.mean(tstLoss, axis=1), label=pl, c=lc, linewidth=2)
        # ax.plot(np.mean(wj, axis=1) / scale, label=pl, c=lc, linewidth=1)
        # ax[1].scatter(np.mean(wj, axis=1)[-1], np.max(np.mean(tstAcc, axis=1)), label=pl, c=lc, linewidth=2)
        # ax[1].scatter(np.mean(tstAcc, axis=1)[-1],np.mean(wj, axis=1)[-1], label=pl, c=lc, linewidth=1)
        ax.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)

        maxacc.append(np.mean(tstAcc, axis=1)[-1])

    ax.set_xlabel("Epochs", fontsize=18)
    ax.set_yticks(np.linspace(0.96, 0.985, 11))
    ax.set_ylim((0.96, .985))
    ax.grid(True)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylabel("Test Accuracy", fontsize=18)

    fig.tight_layout(pad=0)

    if saveas is not None:
        fig.savefig(saveas)

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    return 0


def LoadNetwork(mypath, netname):
    def PrepareConv6(data, myseed, initializer, factor, activation, lr):
        in_shape = data[0][0].shape

        cnn_arch = [[3, 3, 3, int(64 * factor)], [3, 3, int(64 * factor), int(64 * factor)], [],
                    [3, 3, int(64 * factor), int(128 * factor)], [3, 3, int(128 * factor), int(128 * factor)], [],
                    [3, 3, int(128 * factor), int(256 * factor)], [3, 3, int(256 * factor), int(256 * factor)], []]

        dense_arch = [int(255 * factor), int(255 * factor), data[-1]]
        network = MaskedCNN3.makeFullyMaskedCNN(in_shape, cnn_arch, dense_arch, activation, myseed, initializer, lr, True, False)

        return network

    data = localutils.SetMyData("CIFAR")
    myseed = 1234
    initializer = "heconstant"
    factor = 1
    activation = "swish"
    lr = 3e-3

    network = PrepareConv6(data, myseed, initializer, factor, activation, lr)

    # load weights
    ID = ""
    SD = ""
    mypath = "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/"
    netname = "NetworkWeights_ID4b9d68a_SD20988944.pkl"

    # W0 = pickle.load(open(mypath + "NetworkWeights_ID" + ID + "_SD" + SD + ".pkl", "rb"))
    # W0 = pickle.load(open(mypath + "NetworkWeights_ID4b9d68a_SD20988944.pkl", "rb"))
    # M = pickle.load(open(mypath + "Masks_Wj1178012_BS25_ID4b9d68a_PTOnTheFly_SD20988944_AR300_PP0.5220_PS24.pkl", "rb"))

    W0 = pickle.load(open(mypath + "NetworkWeights_ID4dc5cab_SD97354499.pkl", "rb"))
    M = pickle.load(open(mypath + "Masks_Wj1175806_BS25_ID4dc5cab_PTOnTheFly_SD97354499_AR300_PP0.5211_PS24.pkl", "rb"))

    W = []
    lw = 0
    for l in range(1, len(network.layers)):
        w = network.layers[l].get_weights()
        if isinstance(w, list):
            continue
        else:
            # network.layers[l].set_weights([W0[l - 1]*M[l - 1], np.ones_like(M[l - 1])])
            network.layers[l].set_weights([W0[l - 1] * M[l - 1], M[l - 1]])
            # network.layers[l].set_weights([M[l - 1], W0[l - 1] * M[l - 1]])

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, nclasses = data

    TrainL0, TrainA0 = network.evaluate(Xtrain, Ytrain, batch_size=200, verbose=2)
    TestL0, TestA0 = network.evaluate(Xtest, Ytest, batch_size=200, verbose=2)
    ValL0, ValA0 = network.evaluate(Xval, Yval, batch_size=200, verbose=2)

    fit_history = network.fit(Xtrain, Ytrain, batch_size=25, epochs=100, verbose=2, shuffle=True, validation_data=(Xtest, Ytest))

    return 0


def SetMypath():
    mypath = "Outputs/Conv6_relu_heconstant/f1/"
    mypath = "Outputs/LeNet_relu_heconstant_LR0.005/f1/"
    mypath = "Outputs/Conv6_relu_he_LR0.005/f1/"
    mypath = "Outputs/Test_swish_he_LR0.005/f1/"
    mypath = "Outputs/Test_relu_he_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/"
    # mypath = "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f0.5/"
    # mypath = "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f0.5/"
    mypath = "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI__he_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI__heconstant_LR0.005/f1/"
    # mypath = "Outputs/ZMI/Conv6_ZMI__glorot_LR0.005/f1/"
    # mypath = "Z:/r2/gitlabDNN/MaskTrainer/Outputs/Conv6_relu_he_LR0.005/f1/"

    return mypath


def WriteEquations():
    Nhidden_layers = 2

    inputnodes = 2
    outputnodes = 2

    nodes_per_hidden_layer = 3

    available_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
                       "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
                       "W", "X", "Y", "Z"]

    hidden_layer_names = available_names[:Nhidden_layers]

    net = [["X"] * inputnodes]

    for i in range(Nhidden_layers):
        net.append([hidden_layer_names[i]] * nodes_per_hidden_layer)

    net.append(["O"] * outputnodes)
    print(net)

    # input()

    def f(depth, pos, hln):
        print(depth, pos)
        # print(hln[depth][pos])

        if depth < len(hln) - 1:
            f(depth + 1, pos, hln)

        if pos < len(hln[depth]) - 1:
            f(depth, pos + 1, hln)

    f(0, 0, net)

    return 0


def CompareLRScans():
    # conv4_flip = [
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0002/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0003/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0004/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0005/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0006/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0007/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0008/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.0009/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.001/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.002/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.003/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.004/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.005/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.006/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.007/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.008/f1/",
    #     "Outputs/Conv4/flip_relu_heconstant_LR0.009/f1/"
    # ]

    conv4_mask_relu_heconstant = [
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0002/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0003/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0004/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0005/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0006/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0007/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0008/f1/",
        # "Outputs/Conv4/mask_relu_heconstant_LR0.0009/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.001/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.002/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.004/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.005/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.006/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.007/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.008/f1/",
        "Outputs/Conv4/mask_relu_heconstant_LR0.009/f1/"
    ]

    conv4_mask_relu_glorot = [
        # "Outputs/Conv4/mask_relu_glorot_LR0.0002/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0003/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0004/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0005/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0006/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0007/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0008/f1/",
        # "Outputs/Conv4/mask_relu_glorot_LR0.0009/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.002/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.004/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.005/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.006/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.007/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.008/f1/",
        "Outputs/Conv4/mask_relu_glorot_LR0.009/f1/"
    ]
    conv4_mask_relu_he = [
        # "Outputs/Conv4/mask_relu_he_LR0.0002/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0003/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0004/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0005/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0006/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0007/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0008/f1/",
        # "Outputs/Conv4/mask_relu_he_LR0.0009/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.002/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.004/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.005/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.006/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.007/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.008/f1/",
        "Outputs/Conv4/mask_relu_he_LR0.009/f1/"
    ]

    conv2_mask_relu_he = [
        "Outputs/Conv2/mask_relu_he_LR0.0002/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0003/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0004/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0005/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0006/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0007/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0008/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.0009/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.002/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.003/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.004/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.005/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.006/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.007/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.008/f1/",
        "Outputs/Conv2/mask_relu_he_LR0.009/f1/"
    ]
    conv2_mask_relu_glorot = [
        "Outputs/Conv2/mask_relu_glorot_LR0.0002/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0003/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0004/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0005/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0006/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0007/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0008/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.0009/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.002/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.004/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.005/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.006/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.007/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.008/f1/",
        "Outputs/Conv2/mask_relu_glorot_LR0.009/f1/"
    ]
    conv2_mask_relu_heconst = [
        "Outputs/Conv2/mask_relu_heconstant_LR0.0002/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0003/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0004/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0005/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0006/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0007/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0008/f1/",
        "Outputs/Conv2/mask_relu_heconstant_LR0.0009/f1/"
        # "Outputs/Conv2/mask_relu_heconstant_LR0.002/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.004/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.005/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.006/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.007/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.008/f1/",
        # "Outputs/Conv2/mask_relu_heconstant_LR0.009/f1/"
    ]
    # MergeJob(np.asarray(conv2_mask_relu_heconst).reshape(-1))
    PlotAccFscan(conv2_mask_relu_heconst, conv2_mask_relu_heconst)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")
    PlotAccFscan(conv2_mask_relu_glorot, conv2_mask_relu_glorot)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")
    PlotAccFscan(conv2_mask_relu_he, conv2_mask_relu_he)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")

    PlotAccFscan(conv4_mask_relu_he, conv4_mask_relu_he)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")
    PlotAccFscan(conv4_mask_relu_glorot, conv4_mask_relu_glorot)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")
    PlotAccFscan(conv4_mask_relu_heconstant, conv4_mask_relu_heconstant)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")

    # MergeJob(np.asarray(conv4_mask).reshape(-1))
    # PlotAccFscan(conv4_mask, conv4_mask)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")
    # PlotAccFscan(conv4_flip, conv4_flip)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")

    return 0


def CompareOneWeightMaskerPermuter():
    paths = [
        # "Outputs/OneWeight/OneWeight/Masker/LeNet/relu_heconstant_LR0.003/f1/",
        # "Outputs/LeNet/InitActScan100Epochs_Norescaling/relu_heconstant_LR0.005/f1/",
        "Outputs/LeNet/InitActScan100Epochs/relu_heconstant_LR0.005/f1/",
        "../Permuter/Outputs/LeNet/flipper/InitActScan100Epochs/relu_heconstant_LR0.005/f1/",
        "Outputs/OneWeight/OneWeight/Permuter/LeNet/relu_heconstant_LR0.003/f1/"
    ]

    labels = [
        # "One weight network,  0/+1 mask, no rescaling",
        # "He Constant network, 0/+1 mask, no rescaling",
        "He Constant network, all weights,  0/+1 mask - pruning",
        "He Constant network, all weights, -1/+1 mask - sign flipping",
        "He Constant network, one weight,  -1/+1 mask - sign flipping"

    ]

    paths_MRB_MLP = [
        # "Outputs/OneWeight/OneWeight/Masker/LeNet/relu_heconstant_LR0.003/f1/",
        # "Outputs/LeNet/InitActScan100Epochs_Norescaling/relu_heconstant_LR0.005/f1/",
        "Outputs/LeNet/Standard/nobias_relu_glorot_normal_LR0.0012/f1/",
        "Outputs/LeNet/Standard/nobias_relu_he_normal_LR0.0012/f1/",
        "Outputs/LeNet/Standard/nobias_relu_heconstant_LR0.0012/f1/",
        "Outputs/LeNet/Standard/bias_relu_glorot_normal_LR0.0012/f1/",
        "Outputs/LeNet/Standard/bias_relu_he_normal_LR0.0012/f1/",
        "Outputs/LeNet/Standard/bias_relu_heconstant_LR0.0012/f1/",
        "Outputs/LeNet/mask_relu_binary_LR0.0012/f1/",
        "Outputs/LeNet/flip_relu_binary_LR0.0012/f1/",
        "Outputs/LeNet/flip_relu_heconstant_LR0.0012/f1/",
        "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f1/"
    ]

    paths_MRB_Conv6 = [
        # "Outputs/OneWeight/OneWeight/Masker/LeNet/relu_heconstant_LR0.003/f1/",
        # "Outputs/LeNet/InitActScan100Epochs_Norescaling/relu_heconstant_LR0.005/f1/",
        # "Outputs/Conv6/mask_relu_binary_LR0.005/f1/",
        # "Outputs/Conv6/mask_relu_heconstant_LR0.005/f1/",
        # "Outputs/Conv6/mask_relu_binary_LR0.003/f1/",
        # "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
        "Outputs/Conv6/mask_relu_binary_LR0.001/f1/",
        # "Outputs/Conv6/mask_relu_heconstant_LR0.0003/f1/",
        "Outputs/Conv6/flip_relu_binary_LR0.001/f1/"
        # "Outputs/Conv6/flip_relu_heconstant_LR0.0003/f1/"
    ]

    paths_LRscan_Conv6 = [
        # "Outputs/OneWeight/OneWeight/Masker/LeNet/relu_heconstant_LR0.003/f1/",
        # "Outputs/LeNet/InitActScan100Epochs_Norescaling/relu_heconstant_LR0.005/f1/",
        # "Outputs/Conv6/mask_relu_binary_LR0.005/f1/",
        # "Outputs/Conv6/mask_relu_heconstant_LR0.005/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0003/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0004/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0005/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0006/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0007/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0008/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.0009/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.001/f1/",
        "Outputs/Conv6/lrscan/mask_relu_binary_LR0.003/f1/"
        # "Outputs/Conv6/lrscan/flip_relu_binary_LR0.006/f1/",
        # "Outputs/Conv6/lrscan/flip_relu_binary_LR0.009/f1/",
        # "Outputs/Conv6/lrscan/flip_relu_binary_LR0.01/f1/"
    ]

    conv4 = [
        "Outputs/Conv4/mask_relu_heconstant_LR0.001/f1/",
        "Outputs/Conv4/flip_relu_heconstant_LR0.001/f1/"
    ]

    labels = [
        # "One weight network,  0/+1 mask, no rescaling",
        # "He Constant network, 0/+1 mask, no rescaling",
        "Binary Weights, 0/+1 mask"
    ]

    # MergeJob(np.asarray(paths_MRB_MLP).reshape(-1))
    # PlotAccFscan(paths_MRB_MLP, paths_MRB_MLP, saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")

    # MergeJob(np.asarray(paths_MRB_Conv6).reshape(-1))
    # PlotAccFscan(paths_MRB_Conv6, paths_MRB_Conv6)#, saveas="Outputs/PlotsForPaper/1_MRB_Conv6_Acc.pdf")

    # epochs1000 = ["Outputs/Conv6/1000Epochs/flip_relu_binary_LR0.001/f1/"]

    # PlotAccFscan(epochs1000, epochs1000)  # , saveas="Outputs/PlotsForPaper/1_MRB_Conv6_Acc.pdf")
    # PlotAccFscan(paths_LRscan_Conv6, paths_LRscan_Conv6)  # , saveas="Outputs/PlotsForPaper/1_MRB_Conv6_Acc.pdf")

    MergeJob(np.asarray(conv4).reshape(-1))
    PlotAccFscan(conv4, conv4)  # , saveas="Outputs/PlotsForPaper/1_MRB_MLP_Acc.pdf")

    return 0


def PlotsAccuracyMLP():
    Baselines_FreePruning_MinPruning = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            # "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/",
            # "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
            # "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.25/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            # "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/",
            # "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"

            # "Outputs/LeNet/mask_relu_he_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_he_LR0.0012/f0.25/"

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"

            # "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.25/"
        ]
    ]
    Baselines_FreePruning_MinPruning_labels = [
        ["Baseline", "FreePruning", "MinimalPruning"],
        ["Baseline", "FreePruning", "MinimalPruning"],
        ["Baseline", "FreePruning", "MinimalPruning"]
    ]
    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    accylims = [(.95, .99, 5), (.94, .99)]
    sparylims = [(0.55, 1, 9), (0.55, 1.02)]
    # PlotAccAndSparsityPerEpoch(Baselines_FreePruning_MinPruning, axtitles, Baselines_FreePruning_MinPruning_labels, accylims, sparylims,
    #                            saveas="Outputs/PlotsForPaper/MLP/0_BLVsFP")

    Baseline_SignFlipping_MinFlipping = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"
        ]
    ]
    Baseline_SignFlipping_MinFlipping_labels = [
        ["Baseline", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreeFlipping", "MinFlipping"]
    ]

    Baselines_FreePruning_MinPruning_SignFlip_MinFlip = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
            # "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.25/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"

            # "Outputs/LeNet/mask_relu_he_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_he_LR0.0012/f0.25/"

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/BinaryFreePruning/LeNet/P1_0.5/mask_rs_relu_binary_LR0.001/f1/",
            # "Outputs/Allruns/BinarySignFlipping/LeNet/P1_0.5/flip_relu_binary_LR0.001/f1/"

            #
            # "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.5/",
            # "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.25/"
        ]
    ]
    Baselines_FreePruning_MinPruning_SignFlip_MinFlip_labels = [
        ["Baseline", "FreePruning", "MinimalPruning", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning", "FreeFlipping", "MinFlipping", "BinaryFreePruning", "BinaryFreeFlipping"]
    ]

    accylims = [(.95, .99, 9), (.95, .985)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
    # PlotAccVsChangedWeightsPerEpoch(Baselines_FreePruning_MinPruning, axtitles,
    #                                 Baselines_FreePruning_MinPruning_labels,
    #                                 accylims, sparylims, saveas="Outputs/PlotsForPaper/MLP/0_Baselines_FreePruning_MinPruning", axlabels=("Test Accuracy", "Pruned Weights"))
    #
    # PlotAccVsChangedWeightsPerEpoch(Baseline_SignFlipping_MinFlipping, axtitles,
    #                                 Baseline_SignFlipping_MinFlipping_labels,
    #                                 accylims, sparylims, saveas="Outputs/PlotsForPaper/MLP/0_Baseline_SignFlipping_MinFlipping", axlabels=("Test Accuracy", "Flipped Weights"))

    FreePruning_MLP = [
        "Outputs/FreePruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/"
    ]

    PlotAccVsChangedWeightsPerEpoch(FreePruning_MLP[0], axtitles,
                                    FreePruning_MLP[0],
                                    None, None, saveas=None)
    return 0


def PlotsAccuracyConvX():
    Baselines_Free_Min_Pruning_C6 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/BinaryFreePruning/Conv6/P1_0.5/mask_rs_relu_binary_LR0.003/f1/"

            # "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

        ]
    ]
    Baselines_Free_Min_Pruning_C4 = [
        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    Baselines_Free_Min_Pruning_C2 = [
        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.0002/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_he_LR0.0002/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

        ]
    ]

    Baselines_Free_Min_Pruning_C6_labels = [
        ["Baseline", "FreePruning", "MinimalPruning"],
        ["Baseline", "FreePruning", "MinimalPruning"],
        ["Baseline", "FreePruning", "MinimalPruning"]  # ,"BinaryFreePrune"]
    ]
    Baselines_Free_Min_Pruning_C4_labels = [
        ["Baseline", "FreePruning", "MinimalPruning"],  # "SignFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning"],  # "SignFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning"]  # "SignFlipping"]
    ]
    Baselines_Free_Min_Pruning_C2_labels = [
        ["Baseline", "FreePruning", "MinimalPruning"],  # , "SignFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning"],  # , "SignFlipping"],
        ["Baseline", "FreePruning", "MinimalPruning"]  # , "SignFlipping"]
    ]

    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    accylims = [(.50, 0.80, 6), (.50, 0.82)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]

    # Plots for Pruning
    PlotAccVsChangedWeightsPerEpoch(Baselines_Free_Min_Pruning_C6, axtitles,
                                    Baselines_Free_Min_Pruning_C6_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Baselines_Free_Min_Pruning_C6", axlabels=("Test Accuracy", "Pruned Weights"))
    PlotAccVsChangedWeightsPerEpoch(Baselines_Free_Min_Pruning_C4, axtitles,
                                    Baselines_Free_Min_Pruning_C4_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Baselines_Free_Min_Pruning_C4", axlabels=("Test Accuracy", "Pruned Weights"))
    PlotAccVsChangedWeightsPerEpoch(Baselines_Free_Min_Pruning_C2, axtitles,
                                    Baselines_Free_Min_Pruning_C2_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Baselines_Free_Min_Pruning_C2", axlabels=("Test Accuracy", "Pruned Weights"))

    # input()
    # PlotAccAndSparsityPerEpoch(Baselines_Free_Min_Pruning_C6, axtitles, Baselines_Free_Min_Pruning_C6_labels, accylims, sparylims, saveas="Outputs/PlotsForPaper/Convx/0_BLVsFP_C6")
    # #
    # accylims = [(.50, 0.80, 6), (.50, 0.82)]
    # sparylims = [(0.50, 1, 6), (0.50, 1.02)]
    # PlotAccAndSparsityPerEpoch(Baselines_Free_Min_Pruning_C4, axtitles, Baselines_Free_Min_Pruning_C6_labels, accylims, sparylims, saveas="Outputs/PlotsForPaper/Convx/0_BLVsFP_C4")
    #
    # accylims = [(.58, 0.72, 8), (.56, 0.73)]
    # sparylims = [(0.50, 1, 6), (0.50, 1.02)]
    # PlotAccAndSparsityPerEpoch(Baselines_Free_Min_Pruning_C2, axtitles, Baselines_Free_Min_Pruning_C2_labels, accylims, sparylims, saveas="Outputs/PlotsForPaper/Convx/0_BLVsFP_C2")

    Baseline_SignFlipping_C6 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]

    Baseline_SignFlipping_C642 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]

    Baseline_FreePruning_SignFlipping_C642 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

        ]
    ]

    Baseline_SignFlip_MinFlip_C2 = [
        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.0002/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_he_LR0.0002/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    Baseline_SignFlip_MinFlip_C2_labels = [
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"]
    ]

    Baseline_SignFlip_MinFlip_C4 = [
        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]

    Baseline_SignFlip_MinFlip_C6 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    Baseline_SignFlip_MinFlip_C6_labels = [
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"]
    ]

    Baseline_SignFlipping_C6_labels = [
        ["Baseline", "SignFlipping"],
        ["Baseline", "SignFlipping"],
        ["Baseline", "SignFlipping"]
    ]
    Baseline_SignFlipping_C642_labels = [
        ["Baseline", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2"],
        ["Baseline", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2"],
        ["Baseline", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2"]
    ]
    Baseline_FreePruning_SignFlipping_C642_labels = [
        ["Baseline", "FreePuningC6", "FreePuningC4", "FreePuningC2", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2"],
        ["Baseline", "FreePuningC6", "FreePuningC4", "FreePuningC2", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2"],
        ["Baseline", "FreePuningC6", "FreePuningC4", "FreePuningC2", "SignFlippingC6", "SignFlippingC4", "SignFlippingC2", "MinFlippingC6"]
    ]

    Baseline_Binary_FreePruning_C6_4_2 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/BinaryFreePruning/Conv6/P1_0.5/mask_rs_relu_binary_LR0.003/f1/",

        ],
        [
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/BinaryFreePruning/Conv4/P1_0.5/mask_rs_relu_binary_LR0.003/f1/",
        ],
        [
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/BinaryFreePruning/Conv2/P1_0.5/mask_rs_relu_binary_LR0.003/f1/",
        ]
    ]
    Baseline_Binary_FreePruning_C6_4_2_labels = [
        ["Baseline", "Free Pruning", "Single Weight FreePruning"],
        ["Baseline", "Free Pruning", "Single Weight FreePruning"],
        ["Baseline", "Free Pruning", "Single Weight FreePruning"]
    ]

    axtitles_binary = [
        "Conv6 - He Constant",
        "Conv4 - He Constant",
        "Conv2 - He Constant"
    ]

    accylims = [(.50, 0.80, 6), (.50, 0.82)]
    # sparylims = [(0.0, 0.4, 6), (-0.02, 0.44)]
    # sparylims = [(0.0, 0.5, 6), (-0.02, 0.52)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]

    # PlotAccVsChangedWeightsPerEpoch(Baseline_SignFlipping_C6, axtitles,
    #                                 Baseline_SignFlipping_C6_labels, accylims, None,
    #                                 saveas="Outputs/PlotsForPaper/Convx/0_Acc_SignFlip_C6")

    # Plots for Flipping
    PlotAccVsChangedWeightsPerEpoch(Baseline_SignFlip_MinFlip_C6, axtitles, Baseline_SignFlip_MinFlip_C6_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_SignFlip_MinFlip_C6", axlabels=("Test Accuracy", "Flipped Weights"))
    PlotAccVsChangedWeightsPerEpoch(Baseline_SignFlip_MinFlip_C4, axtitles, Baseline_SignFlip_MinFlip_C6_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_SignFlip_MinFlip_C4", axlabels=("Test Accuracy", "Flipped Weights"))
    PlotAccVsChangedWeightsPerEpoch(Baseline_SignFlip_MinFlip_C2, axtitles, Baseline_SignFlip_MinFlip_C2_labels, accylims, sparylims,
                                    saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_SignFlip_MinFlip_C2", axlabels=("Test Accuracy", "Flipped Weights"))

    # accylims = [(.50, 0.75, 6), (.50, 0.75)]
    # PlotAccVsChangedWeightsPerEpoch(Baseline_Binary_FreePruning_C6_4_2, axtitles, Baseline_Binary_FreePruning_C6_4_2_labels, accylims, sparylims,
    #                                 saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_SignFlip_MinFlip_C2")

    accylims = [(.60, 0.80, 6), (.60, 0.82)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
    # PlotAccVsChangedWeightsPerEpochWeightless(Baseline_Binary_FreePruning_C6_4_2, axtitles_binary, Baseline_Binary_FreePruning_C6_4_2_labels, accylims, sparylims,
    #                                           saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_Binary_FreePruning_C6_4_2_weightless")

    # Full comparison Conv2
    fullComparisonConv2 = True
    if fullComparisonConv2:
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C2 = [
            [
                "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.0002/f1/",
                "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/"

            ],

            [
                "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_he_LR0.0002/f1/",
                "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/"

            ],

            [
                "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
                "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"

            ]
        ]
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C2_labels = [
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"]
        ]
        accylims = [(.60, 0.68, 6), (.60, 0.70)]
        sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
        PlotAccVsChangedWeightsPerEpoch(Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C2, axtitles,
                                        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C2_labels, accylims, sparylims,
                                        saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C2",
                                        axlabels=("Test Accuracy", "Pruned/Flipped Weights"))

    # Full comparison Conv4
    fullComparisonConv4 = True
    if fullComparisonConv4:
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C4 = [
            [
                "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
            ],

            [
                "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/"

            ],

            [
                "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
            ]
        ]
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C4_labels = [
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"]
        ]
        accylims = [(.60, 0.78, 6), (.60, 0.78)]
        sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
        PlotAccVsChangedWeightsPerEpoch(Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C4, axtitles,
                                        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C4_labels, accylims, sparylims,
                                        saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C4",
                                        axlabels=("Test Accuracy", "Pruned/Flipped Weights"))

    # Full comparison Conv6
    fullComparisonConv6 = True
    if fullComparisonConv6:
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C6 = [
            [
                "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
            ],

            [
                "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
            ],

            [
                "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
                "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
                "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
                "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
            ]
        ]
        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C6_labels = [
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"],
            ["Baseline", "FreePruning", "MinPruning", "FreeFlipping", "MinFlipping"]
        ]
        accylims = [(.56, 0.80, 6), (.56, 0.82)]
        sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
        PlotAccVsChangedWeightsPerEpoch(Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C6, axtitles,
                                        Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C6_labels, accylims, sparylims,
                                        saveas="Outputs/PlotsForPaper/Convx/0_Acc_Baseline_FreePruning_MinPruning_SignFlip_MinFlip_C6",
                                        axlabels=("Test Accuracy", "Pruned/Flipped Weights"))

    return 0


def PlotsTransferAccuracy():
    SignFlip_Transfer_C246 = [
        [
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    SignFlip_Transfer_C246_labels = [
        ["Conv2", "Conv4", "Conv6"],
        ["Conv2", "Conv4", "Conv6"],
        ["Conv2", "Conv4", "Conv6"]
    ]
    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    accylims = [(.0, .35, 11), (0, 0.35)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
    # PlotAccVsTransferAccuracy(SignFlip_Transfer_C246, axtitles, SignFlip_Transfer_C246_labels, accylims, sparylims,
    #                           saveas="Outputs/PlotsForPaper/Convx/0_Acc_TransferCorrelation_C6_4_2")

    SignFlip_Transfer_MLP = [
        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/",
        ],
        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/",
        ],
        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
        ]
    ]
    SignFlip_Transfer_MLP_labels = [
        ["LeNet"],
        ["LeNet"],
        ["LeNet"]
    ]

    accylims = [(.0, .15, 11), (0, 0.15)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
    # PlotAccVsTransferAccuracy(SignFlip_Transfer_MLP, axtitles, SignFlip_Transfer_MLP_labels, accylims, sparylims, saveas="Outputs/PlotsForPaper/MLP/0_Acc_TransferCorrelation")

    SignFlip_Transfer_C246_MLP = [
        [
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
        ],
        [
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"
        ],
        [
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv4/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/Conv2/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"
        ]
    ]
    SignFlip_Transfer_C246_MLP_labels = [
        ["Conv6", "Conv4", "Conv2", "LeNet"],
        ["Conv6", "Conv4", "Conv2", "LeNet"],
        ["Conv6", "Conv4", "Conv2", "LeNet"]
    ]
    accylims = [(.0, .30, 11), (0, 0.30)]
    sparylims = [(0.0, 0.50, 11), (0.0, 0.50)]
    PlotAccVsTransferAccuracy(SignFlip_Transfer_C246_MLP, axtitles,
                              SignFlip_Transfer_C246_MLP_labels, accylims, sparylims,
                              saveas="Outputs/PlotsForPaper/MLP/0_SignFlip_Transfer_C246_MLP")

    return 0


def PlotsP1Accuracies():
    MLP_p1 = [
        [
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0004/f1/", 0.0),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/", 0.6),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/", 0.9),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/", 0.95),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/", 0.98),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/", 0.99),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0004/f1/", 0.999)

        ],
        [
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_relu_heconstant_LR0.001/f1/", 0.0),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_relu_heconstant_LR0.001/f1/", 0.1),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_relu_heconstant_LR0.001/f1/", 0.2),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_relu_heconstant_LR0.001/f1/", 0.3),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_relu_heconstant_LR0.001/f1/", 0.4),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/f1/", 0.5),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_relu_heconstant_LR0.001/f1/", 0.6),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_relu_heconstant_LR0.001/f1/", 0.7),
            # ("Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_relu_heconstant_LR0.001/f1/", 0.8),
            # ("Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_relu_heconstant_LR0.001/f1/", 0.9),
            #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_relu_heconstant_LR0.001/f1/", 0.95),
            #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_relu_heconstant_LR0.001/f1/", 0.98),
            #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_relu_heconstant_LR0.001/f1/", 0.99),
            #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_relu_heconstant_LR0.001/f1/", 0.999)
            #
        ],
        # [
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/", 0.0),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/", 0.1),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/", 0.2),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/", 0.3),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/", 0.4),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/", 0.5),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/", 0.6),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/", 0.7),
        #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/", 0.8),
        #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/", 0.9),
        #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/", 0.95),
        #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/", 0.98),
        #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/", 0.99),
        #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/", 0.999)
        #
        # ],

        # [
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.001/f1/", 0.0),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/", 0.1),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/", 0.2),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/", 0.3),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/", 0.4),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/", 0.5),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/", 0.6),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/", 0.9),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/", 0.95),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/", 0.98),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/", 0.99),
        #     ("Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.001/f1/", 0.999)
        #
        # ],

    ]

    MLP_p1_labels = ["Free Pruning", "Minimal Pruning"]
    PlotAccVsP1(MLP_p1, MLP_p1_labels, saveas="Outputs/PlotsForPaper/MLP/0_Acc_P1_all")

    Conv6_4_2_p1 = [
        # [
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.003/f1/", 0.1),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.003/f1/", 0.2),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.003/f1/", 0.3),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.003/f1/", 0.4),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/", 0.5),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.003/f1/", 0.6),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.7/mask_rs_relu_heconstant_LR0.00025/f1/", 0.7),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/", 0.8),
        #
        # ],
        # [
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.001/f1/", 0.0),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.05/mask_rs_relu_heconstant_LR0.0009/f1/", 0.05),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.0009/f1/", 0.1),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.0009/f1/", 0.2),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.65/mask_rs_relu_heconstant_LR0.0009/f1/", 0.65),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
        #
        # ],

        # [
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/", 0.1),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/", 0.2),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.00025/f1/", 0.3),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.00025/f1/", 0.4),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.00025/f1/", 0.5),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.00025/f1/", 0.6),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.7/mask_rs_relu_heconstant_LR0.00025/f1/", 0.7),
        #     # ("Outputs/Allruns/FreePruning/Conv6/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
        #
        # ],

        [
            # ("Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
            # ("Outputs/Allruns/FreePruning/Conv6/P1_0.05/mask_rs_relu_heconstant_LR0.0009/f1/", 0.05),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
            # ("Outputs/Allruns/FreePruning/Conv6/P1_0.65/mask_rs_relu_heconstant_LR0.0009/f1/", 0.65),

        ],

        [
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.1/mask_rs_relu_heconstant_LR0.0009/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.2/mask_rs_relu_heconstant_LR0.0009/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/Conv4/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
            # ("Outputs/Allruns/FreePruning/Conv4/P1_0.7/mask_rs_relu_heconstant_LR0.003/f1/", 0.7),
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/", 0.8),
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.9/mask_rs_relu_heconstant_LR0.00025/f1/", 0.9)

        ],

        [
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.1/mask_rs_relu_heconstant_LR0.0009/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.2/mask_rs_relu_heconstant_LR0.0009/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/Conv2/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.7/mask_rs_relu_heconstant_LR0.00025/f1/", 0.7),
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/", 0.8),
            # ("Outputs/Allruns/FreePruning/Conv2/P1_0.9/mask_rs_relu_heconstant_LR0.00025/f1/", 0.9)

        ],

    ]
    Conv6_4_2_p1_labels = ["Conv6", "Conv4", "Conv2"]
    PlotAccVsP1(Conv6_4_2_p1, Conv6_4_2_p1_labels, saveas="Outputs/PlotsForPaper/Convx/0_Acc_P1_C6_4_2")

    MLP_Conv6_p1 = [
        [
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_relu_heconstant_LR0.001/f1/", 0.0),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_relu_heconstant_LR0.001/f1/", 0.1),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_relu_heconstant_LR0.001/f1/", 0.2),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_relu_heconstant_LR0.001/f1/", 0.3),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_relu_heconstant_LR0.001/f1/", 0.4),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/f1/", 0.5),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_relu_heconstant_LR0.001/f1/", 0.6),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_relu_heconstant_LR0.001/f1/", 0.7),
            ("Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_relu_heconstant_LR0.001/f1/", 0.8),
        ],
        [
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.001/f1/", 0.0),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/", 0.6),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/", 0.9),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/", 0.95),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/", 0.98),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/", 0.99),
        ],
        [
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
        ],
    ]
    MLP_Conv6_p1_labels = ["MLP MinPruning", "MLP FreePruning", "Conv6_mixed"]
    PlotAccVsP1(MLP_Conv6_p1, MLP_Conv6_p1_labels)  # , saveas="Outputs/PlotsForPaper/Convx/0_Acc_P1_C6_2")

    # MLP_Conv6_4_2_p1=[
    #     [
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_relu_heconstant_LR0.001/f1/", 0.0),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_relu_heconstant_LR0.001/f1/", 0.1),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_relu_heconstant_LR0.001/f1/", 0.2),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_relu_heconstant_LR0.001/f1/", 0.3),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_relu_heconstant_LR0.001/f1/", 0.4),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/f1/", 0.5),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_relu_heconstant_LR0.001/f1/", 0.6),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_relu_heconstant_LR0.001/f1/", 0.7),
    #         ("Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_relu_heconstant_LR0.001/f1/", 0.8),
    #         # ("Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_relu_heconstant_LR0.001/f1/", 0.9),
    #         #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_relu_heconstant_LR0.001/f1/", 0.95),
    #         #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_relu_heconstant_LR0.001/f1/", 0.98),
    #         #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_relu_heconstant_LR0.001/f1/", 0.99),
    #         #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_relu_heconstant_LR0.001/f1/", 0.999)
    #         #
    #     ],
    #     # [
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/", 0.0),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/", 0.1),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/", 0.2),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/", 0.3),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/", 0.4),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/", 0.5),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/", 0.6),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/", 0.7),
    #     #     ("Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/", 0.8),
    #     #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/", 0.9),
    #     #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/", 0.95),
    #     #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/", 0.98),
    #     #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/", 0.99),
    #     #     # ("Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/", 0.999)
    #     #
    #     # ],
    #
    #     [
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.001/f1/", 0.0),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/", 0.1),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/", 0.2),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/", 0.3),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/", 0.4),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/", 0.5),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/", 0.6),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/", 0.9),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/", 0.95),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/", 0.98),
    #         ("Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/", 0.99),
    #         # ("Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.001/f1/", 0.999)
    #
    #     ],
    #     [
    #         # ("Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
    #         # ("Outputs/Allruns/FreePruning/Conv6/P1_0.05/mask_rs_relu_heconstant_LR0.0009/f1/", 0.05),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/", 0.1),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/", 0.2),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.0009/f1/", 0.3),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.0009/f1/", 0.4),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/", 0.5),
    #         ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.0009/f1/", 0.6),
    #         # ("Outputs/Allruns/FreePruning/Conv6/P1_0.65/mask_rs_relu_heconstant_LR0.0009/f1/", 0.65),
    #
    #     ],
    #
    #     [
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.1/mask_rs_relu_heconstant_LR0.003/f1/", 0.1),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.2/mask_rs_relu_heconstant_LR0.003/f1/", 0.2),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.3/mask_rs_relu_heconstant_LR0.003/f1/", 0.3),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.4/mask_rs_relu_heconstant_LR0.003/f1/", 0.4),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/", 0.5),
    #         ("Outputs/Allruns/FreePruning/Conv4/P1_0.6/mask_rs_relu_heconstant_LR0.003/f1/", 0.6),
    #         # ("Outputs/Allruns/FreePruning/Conv4/P1_0.7/mask_rs_relu_heconstant_LR0.003/f1/", 0.7),
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/", 0.8),
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.9/mask_rs_relu_heconstant_LR0.00025/f1/", 0.9)
    #
    #     ],
    #
    #     [
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/", 0.0),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.1/mask_rs_relu_heconstant_LR0.003/f1/", 0.1),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.2/mask_rs_relu_heconstant_LR0.003/f1/", 0.2),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.3/mask_rs_relu_heconstant_LR0.003/f1/", 0.3),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.4/mask_rs_relu_heconstant_LR0.003/f1/", 0.4),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/", 0.5),
    #         ("Outputs/Allruns/FreePruning/Conv2/P1_0.6/mask_rs_relu_heconstant_LR0.003/f1/", 0.6),
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.7/mask_rs_relu_heconstant_LR0.00025/f1/", 0.7),
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/", 0.8),
    #         # ("Outputs/Allruns/FreePruning/Conv2/P1_0.9/mask_rs_relu_heconstant_LR0.00025/f1/", 0.9)
    #
    #     ],
    #
    # ]
    # MLP_Conv6_4_2_p1_labels=["MLP MinPruning", "MLP FreePruning","Conv6_mixed", "conv4", "conv2"]
    # PlotAccVsP1(MLP_Conv6_4_2_p1, MLP_Conv6_4_2_p1_labels)#, saveas="Outputs/PlotsForPaper/Convx/0_Acc_P1_C6_2")

    return 0


def SparsityPlots(mypaths, axtitles, plotlabels, accylims=None, sparylims=None, saveas=None, axlabels=None):
    nplots = len(mypaths)
    lins, cols = 3, 3
    fig, axes = plt.subplots(lins, cols, figsize=(15, 8), dpi=DPI(), sharey=False, sharex=True)
    AxAccuracy = axes[0, :].flatten(-1)
    AxRatioFreeFlipping = axes[1, :].flatten(-1)
    AxRatioMinFlipping = axes[2, :].flatten(-1)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    lenghts = [len(mypaths[0]), len(mypaths[1]), len(mypaths[2])]

    # colors = colors[:np.max(np.asarray(lenghts))] * 100

    styles = ["-", "-.", "--", ":"]
    # styles = ["-", "-", "-", "-", "-"]
    styles = styles[:np.max(np.asarray(lenghts))] * 100
    # random.shuffle(styles)
    # random.shuffle(colors)
    # styles = ["-"] * len(mypaths[0]) * 100

    scale = 2261184
    scale = 256476
    # 256476

    for i, path in zip(np.arange(len(mypaths)), mypaths):
        AxAccuracy[i].set_title(axtitles[i], fontsize=20)

        for j, p in enumerate(path):
            trnAcc = np.load(p + "MergedTrainAcc.npy")
            tstAcc = np.load(p + "MergedTestAcc.npy")
            tstLoss = np.load(p + "MergedTestLoss.npy")
            # wj = np.load(p + "MergedRemainingWeights.npy")

            if os.path.exists(p + "MergedRemainingWeightsPerLayer.npy"):
                wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")
            # wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")

            meancurve = np.mean(tstAcc, axis=1)
            mincurve = np.min(tstAcc, axis=1)
            maxcurve = np.max(tstAcc, axis=1)

            pl = plotlabels[i][j]
            lc = colors[j]
            lstyle = styles[j]

            # wj = np.sum(wjpl[:, :, :, 0], axis=2)
            AxAccuracy[i].plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=3, linestyle=lstyle)
            AxAccuracy[i].plot(np.min(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            AxAccuracy[i].plot(np.max(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
            AxAccuracy[i].fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)

            if accylims is not None:
                AxAccuracy[i].set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
                AxAccuracy[i].set_ylim(accylims[1])

            AxAccuracy[i].grid(True)
            AxRatioFreeFlipping[i].grid(True)
            AxRatioMinFlipping[i].grid(True)
            print(pl)

            ax = axes[j, :].flatten(-1)[i]

            if j > 0:

                # here we want to plot the number of negative, zero and positive weights - in total
                # loop through all layers
                nruns, nseeds, nlayers, nkpis = wjpl.shape
                # maskcounts.append([NegativeMasks, ZeroMasks, PositiveMasks, NegativeMW, ZeroMW, PositiveMW])
                for l in range(nlayers):
                    nnegativeWM = wjpl[:, :, l, 0]
                    npositiveWM = wjpl[:, :, l, 2]
                    print(nnegativeWM.shape)
                    # axrfreeflip.plot(r + f * np.mean(nnegativeWM, axis=1) / scale, label="L" + str(l) + " neg", c=colors[l], linewidth=2, linestyle="-")
                    # axrfreeflip.plot(r + f * np.mean(npositiveWM, axis=1) / scale, label="L" + str(l) + " pos", c=colors[l], linewidth=2, linestyle="--")
                    ax.plot(np.mean(nnegativeWM, axis=1) / np.mean(npositiveWM, axis=1), label="L" + str(l), c=colors[l], linewidth=2, linestyle="--")

                    # axrfreeflip.plot(r + f * np.min(nnegativeWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="-", alpha=0.5)
                    # axrfreeflip.plot(r + f * np.max(nnegativeWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="-", alpha=0.5)
                    #
                    # axrfreeflip.plot(r + f * np.min(npositiveWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="--", alpha=0.5)
                    # axrfreeflip.plot(r + f * np.max(npositiveWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="--", alpha=0.5)
                    # axrfreeflip.fill_between(np.arange(nnegativeWM.shape[0]), r + f * np.min(nnegativeWM, axis=1) / scale, r + f * np.max(nnegativeWM, axis=1) / scale, alpha=0.1, facecolor=lc)

                nnegativeWM = np.sum(wjpl[:, :, :, 0], axis=2)
                npositiveWM = np.sum(wjpl[:, :, :, 2], axis=2)
                # scale = np.mean(nnegativeWM + npositiveWM, axis=1)
                # print(nnegativeWM.shape, scale.shape)
                # axrfreeflip.plot(r + f * np.mean(nnegativeWM, axis=1) / scale, label="Total neg", c=colors[l], linewidth=3, linestyle="-")
                # axrfreeflip.plot(r + f * np.mean(npositiveWM, axis=1) / scale, label="Total pos", c=colors[l], linewidth=3, linestyle="--")
                ax.plot(np.mean(nnegativeWM, axis=1) / np.mean(npositiveWM, axis=1), label="Total", c="black", linewidth=2, linestyle="-")

                if sparylims is not None:
                    ax.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
                    ax.set_ylim(sparylims[1])

                ax.set_ylabel(axlabels[0], fontsize=18)
                ax.set_ylabel(axlabels[1], fontsize=18)

                ax.legend(fontsize=10, bbox_to_anchor=(0.14, .02, .8, .5), loc='lower center', ncol=2, mode="expand", borderaxespad=0.)

    oldstyle = False

    if oldstyle:
        for i, p1, axacc, axt1, PL, axrfreeflip, axrminflip in zip(np.arange(len(mypaths)), mypaths, AxAccuracy, axtitles, plotlabels, AxRatioFreeFlipping, AxRatioMinFlipping):

            for p, lstyle, pl, lc in zip(p1, styles, PL, colors):
                trnAcc = np.load(p + "MergedTrainAcc.npy")
                tstAcc = np.load(p + "MergedTestAcc.npy")
                tstLoss = np.load(p + "MergedTestLoss.npy")
                wj = np.load(p + "MergedRemainingWeights.npy")
                scale = np.max(wj)

                if os.path.exists(p + "MergedRemainingWeightsPerLayer.npy"):
                    wjpl = np.load(p + "MergedRemainingWeightsPerLayer.npy")
                    # print(wjpl.shape)

                meancurve = np.mean(tstAcc, axis=1)
                mincurve = np.min(tstAcc, axis=1)
                maxcurve = np.max(tstAcc, axis=1)

                posmax = np.argmax(meancurve)
                # print(axt1, pl, "mean {:.2f}, min {:.2f}, max {:.2f}, pm=".format(meancurve[posmax] * 100, mincurve[posmax] * 100, maxcurve[posmax] * 100),
                #       meancurve[posmax] * 100 - mincurve[posmax] * 100)

                r, f = 1, -1
                if "Flipping" in pl or "MinFlipping" in pl:
                    wj = np.sum(wjpl[:, :, :, 0], axis=2)
                    print(wjpl.shape)
                    r, f = 0, 1

                if "Baseline" not in pl:

                    # here we want to plot the number of negative, zero and positive weights - in total
                    # loop through all layers
                    nruns, nseeds, nlayers, nkpis = wjpl.shape
                    # maskcounts.append([NegativeMasks, ZeroMasks, PositiveMasks, NegativeMW, ZeroMW, PositiveMW])
                    for l in range(nlayers):
                        nnegativeWM = wjpl[:, :, l, 3]
                        npositiveWM = wjpl[:, :, l, 5]
                        scale = np.mean(nnegativeWM + npositiveWM, axis=1)
                        scale = 1
                        # print(nnegativeWM.shape, scale.shape)
                        # axrfreeflip.plot(r + f * np.mean(nnegativeWM, axis=1) / scale, label="L" + str(l) + " neg", c=colors[l], linewidth=2, linestyle="-")
                        # axrfreeflip.plot(r + f * np.mean(npositiveWM, axis=1) / scale, label="L" + str(l) + " pos", c=colors[l], linewidth=2, linestyle="--")
                        axrfreeflip.plot(r + f * (np.mean(nnegativeWM, axis=1) / np.mean(npositiveWM, axis=1)), label="L" + str(l), c=colors[l], linewidth=2, linestyle="--")

                        # axrfreeflip.plot(r + f * np.min(nnegativeWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="-", alpha=0.5)
                        # axrfreeflip.plot(r + f * np.max(nnegativeWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="-", alpha=0.5)
                        #
                        # axrfreeflip.plot(r + f * np.min(npositiveWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="--", alpha=0.5)
                        # axrfreeflip.plot(r + f * np.max(npositiveWM, axis=1) / scale, c=colors[l], linewidth=1, linestyle="--", alpha=0.5)
                        # axrfreeflip.fill_between(np.arange(nnegativeWM.shape[0]), r + f * np.min(nnegativeWM, axis=1) / scale, r + f * np.max(nnegativeWM, axis=1) / scale, alpha=0.1, facecolor=lc)

                    nnegativeWM = np.sum(wjpl[:, :, :, 3], axis=2)
                    npositiveWM = np.sum(wjpl[:, :, :, 5], axis=2)
                    # scale = np.mean(nnegativeWM + npositiveWM, axis=1)
                    # print(nnegativeWM.shape, scale.shape)
                    # axrfreeflip.plot(r + f * np.mean(nnegativeWM, axis=1) / scale, label="Total neg", c=colors[l], linewidth=3, linestyle="-")
                    # axrfreeflip.plot(r + f * np.mean(npositiveWM, axis=1) / scale, label="Total pos", c=colors[l], linewidth=3, linestyle="--")
                    axrfreeflip.plot(r + f * np.mean(nnegativeWM, axis=1) / np.mean(npositiveWM, axis=1), label="Total", c="black", linewidth=3, linestyle="--")

                axacc.plot(np.mean(tstAcc, axis=1), label=pl, c=lc, linewidth=3, linestyle=lstyle)
                axacc.plot(np.min(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
                axacc.plot(np.max(tstAcc, axis=1), c=lc, linewidth=1, linestyle=lstyle, alpha=0.2)
                axacc.fill_between(np.arange(tstAcc.shape[0]), np.min(tstAcc, axis=1), np.max(tstAcc, axis=1), alpha=0.1, facecolor=lc)

                axacc.grid(True)
                if i == 1:
                    # axacc.legend(fontsize=10, loc='lower right')  # fontsize=5)
                    # axacc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                    axacc.legend(fontsize=10, bbox_to_anchor=(0.14, .02, .8, .5), loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
                    axrfreeflip.legend(fontsize=9)  # , bbox_to_anchor=(0.14, .02, .8, .5), loc='center', ncol=3, mode="expand", borderaxespad=0.)

                axacc.set_title(axt1, fontsize=20)

                if accylims != None:
                    axacc.set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
                    axacc.set_ylim(accylims[1])

                if sparylims != None:
                    axrfreeflip.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
                    axrfreeflip.set_ylim(sparylims[1])
                    axrminflip.set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
                    axrminflip.set_ylim(sparylims[1])

                axrfreeflip.grid(True)
                axrfreeflip.set_xlabel("Epochs", fontsize=20)
                # axacc.legend(fontsize=8, loc='lower right')  # fontsize=5)

            # axacc.set_yscale('log',basey=2)

    # axes[1].legend(fontsize=18, loc='lower right')  # fontsize=5)
    if axlabels is None:
        axlabels = ("Test Accuracy", "Pruned Weights")

    AxAccuracy[0].set_ylabel(axlabels[0], fontsize=18)
    AxRatioFreeFlipping[0].set_ylabel(axlabels[1], fontsize=18)

    if accylims is not None:
        AxAccuracy[0].set_yticks(np.linspace(accylims[0][0], accylims[0][1], accylims[0][2]))
        AxAccuracy[0].set_ylim(accylims[1])

    # if sparylims is not None:
    #     AxRatioFreeFlipping[0].set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
    #     AxRatioFreeFlipping[0].set_ylim(sparylims[1])
    #
    #     AxRatioMinFlipping[0].set_yticks(np.linspace(sparylims[0][0], sparylims[0][1], sparylims[0][2]))
    #     AxRatioMinFlipping[0].set_ylim(sparylims[1])

    # AxAccuracy[1].set_yticklabels([])
    # AxAccuracy[2].set_yticklabels([])
    # AxRatioFreeFlipping[1].set_yticklabels([])
    # AxRatioFreeFlipping[2].set_yticklabels([])

    # axes[0].set_ylabel("Accuracy  -  Sparsity", fontsize=18)
    # axes[1].legend(fontsize=18, loc='lower right')
    # fig_MeanAccs, axacc = plt.subplots(figsize=(10, 6), dpi=DPI())
    # axacc.set_title(mypath)

    fig.tight_layout(pad=1)

    if saveas is not None:
        fig.savefig(saveas + ".pdf")
        fig.savefig(saveas + ".png")

    if socket.gethostname() == "DESKTOP-UBRVFON":
        plt.show()

    # plt.show()

    return 0


def PlotsSparsities():
    Baseline_SignFlipping_MinFlipping = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"
        ]
    ]
    Baseline_SignFlipping_MinFlipping_labels = [
        ["Baseline", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreeFlipping", "MinFlipping"],
        ["Baseline", "FreeFlipping", "MinFlipping"]
    ]

    SignFlipping_MLP = [
        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"

        ],

        [
            "Outputs/Allruns/SignFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"
        ]
    ]
    SignFlipping_MLP_labels = [
        ["FreeFlipping"],
        ["FreeFlipping"],
        ["FreeFlipping"]
    ]

    MinFlipping_MLP = [
        [
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_glorot_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_he_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/MinFlipping/LeNet/P1_0.5/flip_relu_heconstant_LR0.001/f1/"
        ]
    ]
    MinFlipping_MLP_labels = [
        ["MinFlipping"],
        ["MinFlipping"],
        ["MinFlipping"]
    ]

    SignFlipping_C6 = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/SignFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    SignFlipping_C6_labels = [
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"],
        ["Baseline", "SignFlipping", "MinFlipping"]
    ]

    MinFlipping_C6 = [
        [
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_he_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/MinFlipping/Conv6/P1_0.5/flip_relu_heconstant_LR0.0005/f1/"
        ]
    ]
    MinFlipping_C6_labels = [
        ["MinFlipping"],
        ["MinFlipping"],
        ["MinFlipping"]
    ]

    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    accylims = [(.95, .99, 9), (.95, .985)]
    sparylims = [(0.30, 0.70, 11), (0.30, 0.70)]
    # SparsityPlots(SignFlipping_MLP, axtitles,
    #               SignFlipping_MLP_labels,
    #               accylims, None, saveas=None, axlabels=("Test Accuracy", "Fraction of Weights"))

    # SparsityPlots(MinFlipping_MLP, axtitles,
    #               MinFlipping_MLP_labels,
    #               accylims, None, saveas=None, axlabels=("Test Accuracy", "Fraction of Weights"))

    accylims = [(.50, 0.80, 6), (.50, 0.82)]
    sparylims = [(0.7, 2.3, 11), (0.65, 2.5)]

    SparsityPlots(SignFlipping_C6, axtitles,
                  SignFlipping_C6_labels,
                  accylims, None, saveas=None, axlabels=("Test Accuracy", "Neg/Pos Weights"))

    # SparsityPlots(MinFlipping_C6, axtitles,
    #               MinFlipping_C6_labels,
    #               accylims, None, saveas=None, axlabels=("Test Accuracy", "Fraction of Weights"))
    return


def PlotAccForGlorotHeHeConstant():
    pathsCNN = [

        [
            # "Outputs/03.29/Norescaling_NLS/relu_glorot_LR0.001/f1/",
            # "Outputs/03.29/Norescaling_NLS/swish_glorot_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/relu_glorot_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/swish_glorot_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/relu_glorot_LR0.005/f1/"
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0008/f1/"
            # "Outputs/ZMI/Conv6_ZMI_relu_glorot_LR0.005/f1/"
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/swish_glorot_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/relu_heconstant_LR0.005/f1/"
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/swish_heconstant_LR0.005/f1/"

        ],

        [
            # "Outputs/03.29/Norescaling_NLS/relu_he_LR0.001/f1/",
            # "Outputs/03.29/Norescaling_NLS/swish_he_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/relu_he_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/swish_he_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/relu_he_LR0.005/f1/"
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0008/f1/"
            # "Outputs/ZMI/Conv6_ZMI_relu_he_LR0.005/f1/"

            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/swish_he_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/relu_heconstant_LR0.005/f1/"
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/swish_heconstant_LR0.005/f1/"
        ],

        [
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0008/f1/"
            # "Outputs/ZMI/Conv6_ZMI_relu_heconstant_LR0.005/f1/"

            # "Outputs/03.29/Norescaling_NLS/relu_heconstant_LR0.001/f1/",
            # "Outputs/03.29/Norescaling_NLS/swish_heconstant_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/relu_heconstant_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Rescaling_NLP/swish_heconstant_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/relu_heconstant_LR0.005/f1/"
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling/swish_heconstant_LR0.005/f1/",
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/relu_heconstant_LR0.005/f1/"
            # "Outputs/Conv6/InitActScan100Epochs_Norescaling_NLSInitHe/swish_heconstant_LR0.005/f1/"
        ]

    ]

    pathsMLP = [

        [
            "Outputs/LeNet/InitActScan100Epochs/relu_glorot_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_glorot_LR0.005/f1/"

        ],

        [
            "Outputs/LeNet/InitActScan100Epochs/relu_he_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_he_LR0.005/f1/"
        ],

        [
            "Outputs/LeNet/InitActScan100Epochs/relu_heconstant_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_heconstant_LR0.005/f1/"
        ]

    ]

    mypaths_Masker_Flipper_LeNet = [

        [
            "../Permuter/Outputs/LeNet/flipper/InitActScan100Epochs/relu_glorot_LR0.005/f1/",
            # "../Permuter/Outputs/LeNet/flipper/InitActScan100Epochs/swish_glorot_LR0.005/f1/",
            "Outputs/LeNet/InitActScan100Epochs/relu_glorot_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_glorot_LR0.005/f1/"
        ],

        [
            "../Permuter/Outputs/LeNet/flipper/InitActScan100Epochs/relu_he_LR0.005/f1/",
            # "../Permuter/Outputs/LeNet/flipper/InitActScan100Epochs/swish_he_LR0.005/f1/",
            "Outputs/LeNet/InitActScan100Epochs/relu_he_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_he_LR0.005/f1/"
        ],

        [
            "../Permuter/Outputs/Lenet/flipper/InitActScan100Epochs/relu_heconstant_LR0.005/f1/",
            # "../Permuter/Outputs/Lenet/flipper/InitActScan100Epochs/swish_heconstant_LR0.005/f1/",
            "Outputs/LeNet/InitActScan100Epochs/relu_heconstant_LR0.005/f1/"
            # "Outputs/LeNet/InitActScan100Epochs/swish_heconstant_LR0.005/f1/"
        ]
    ]

    conv2_4_6_mask_relu_glorot_he_heconst = [

        [
            # "Outputs/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",

            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/"
        ],

        [
            # "Outputs/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",

            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/"
        ],

        [
            # "Outputs/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",

            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/"
        ]
    ]

    pathsscanlrconv6 = [

        [
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0005/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.0008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_glorot_LR0.009/f1/"

        ],

        [
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0005/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.0008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_he_LR0.009/f1/"
        ],

        [
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.0008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.002/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.004/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.007/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.008/f1/",
            "Outputs/Conv6/Rescaling_NLP/relu_heconstant_LR0.009/f1/"]

    ]

    pathsscanlrconv6_swish = [

        [
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.0001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.0009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.01/f1/"
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.03/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.06/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.09/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.1/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.3/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.6/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_glorot_LR0.9/f1/"

        ],
        [
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.0001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.0009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.01/f1/"
            # "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.03/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.06/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.09/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.1/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_he_LR0.3/f1/"
        ],

        [
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.0001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.0003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.0006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.0009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.001/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.003/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.006/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.009/f1/",
            "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.01/f1/"
            # "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.03/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.06/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.09/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.1/f1/",
            # "Outputs/Conv6/Rescaling_NLP/swish_heconstant_LR0.3/f1/"
        ]

    ]

    # MergeJob(np.asarray(conv2_4_6_mask_relu_glorot_he_heconst).reshape(-1))
    # input()

    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    MLP_FreePruning_MinPruning_Skewed = [

        [
            # "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"

        ],

        [
            # "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"
        ]

    ]

    MLP_FreePruning = [

        [
            "Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0004/f1/"

        ],

        [
            "Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.001/f1/"
        ],

        [
            # "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"
        ]

    ]

    pathsConv6_Skewed = [

        [
            # "Outputs/Conv6/SkewedInit/P1_0.0/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.1/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.2/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.3/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.4/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.6/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.7/mask_rs_relu_heconstant_LR0.003/f1/"
            # "Outputs/Conv6/SkewedInit/P1_0.8/mask_rs_relu_heconstant_LR0.003/f1/"
            # "Outputs/Conv6/SkewedInit/P1_0.9/mask_rs_relu_heconstant_LR0.003/f1/"
            # "Outputs/Conv6/SkewedInit/P1_0.95/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.975/mask_rs_relu_heconstant_LR0.003/f1/"
            # "Outputs/Conv6/SkewedInit/P1_0.99/mask_rs_relu_heconstant_LR0.003/f1/"
            # "Outputs/Conv6/SkewedInit/P1_0.999/mask_rs_relu_heconstant_LR0.003/f1/"

        ],

        [
            # "Outputs/Conv6/SkewedInit/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.975/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/Conv6/SkewedInit/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"
        ],

        [
            "Outputs/Conv6/SkewedInit/P1_0.0/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.1/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.2/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.3/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.4/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.6/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.7/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.8/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.9/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.95/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.975/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.99/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Conv6/SkewedInit/P1_0.999/mask_rs_relu_heconstant_LR0.0005/f1/"
        ]

    ]

    # PlotAccVsWj(pathsscanlrconv6, axtitles, pathsscanlrconv6, None,
    #             saveas="Outputs/PlotsForPaper/0_CNN_ScanLR_relu_AccForDifferentInitializations.pdf")
    #
    # PlotAccVsWj(pathsscanlrconv6_swish, axtitles, pathsscanlrconv6_swish, None,
    #             saveas="Outputs/PlotsForPaper/0_CNN_ScanLR_swish_AccForDifferentInitializations.pdf")

    labels = [["Sign flipping", "Masking"], ["Sign flipping", "Masking"], ["Sign flipping", "Masking"]]

    # PlotAccVsWj(conv2_4_6_mask_relu_glorot_he_heconst, axtitles,
    #             conv2_4_6_mask_relu_glorot_he_heconst, None,
    #             saveas="Outputs/PlotsForPaper/0_conv2_4_6_mask_relu_glorot_he_heconst_AccForDifferentInitializations.pdf")

    path_mlp_relu_glorot_he_hc_f1xxx = [

        [
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.25/"

        ],
        [
            "Outputs/LeNet/mask_relu_he_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f0.25/"

        ],
        [
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.25/"
        ]
    ]

    path_mlp_relu_glorot_he_hc_baseline_f1 = [

        [
            "Outputs/LeNet/Baseline/bias_relu_glorot_normal_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f0.25/"

        ],
        [
            "Outputs/LeNet/Baseline/bias_relu_he_normal_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f0.25/"

        ],
        [
            "Outputs/LeNet/Baseline/bias_relu_heconstant_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.5/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f0.25/"
        ]
    ]

    path_vgg = [

        [
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0001/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0002/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0003/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0004/f1/",
            # "Outputs/VGG19_LTH/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/",
            # "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_swish_heconstant_LR0.0025/f1/",
            # "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0006/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0007/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0008/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0009/f1/"
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.002/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.004/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_rs_relu_heconstant_LR0.005/f1/"
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_glorot_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_glorot_LR0.0025/f1/"
        ],

        [
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0001/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0002/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0003/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0004/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0005/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0006/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0007/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0008/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.0009/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.002/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.004/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_rs_relu_heconstant_LR0.005/f1/"
            # "Outputs/VGG19_LTH/LRDecay/BS50/DataAugm/mask_relu_he_LR0.0025/f1/",
            # "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_swish_heconstant_LR0.0025/f1/",
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_he_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_he_LR0.0025/f1/"

        ],

        [
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0001/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0002/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0003/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0004/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0005/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0006/f1/",
            # "Outputs/VGG19/BS100/DataAugm/mask_relu_heconstant_LR0.0007/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.0008/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.0009/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.001/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.002/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.003/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.004/f1/",
            # "Outputs/VGG19/LRDecay/BS100/DataAugm/mask_relu_heconstant_LR0.005/f1/"
            # "Outputs/VGG19_LTH/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/",
            # "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_swish_heconstant_LR0.0025/f1/",
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/"
        ]
    ]

    conv2_4_6_vgg19 = [

        [
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_glorot_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_glorot_LR0.0025/f1/"
        ],

        [
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_he_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_he_LR0.0025/f1/"
        ],

        [
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/VGG19_BN/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/",
            "Outputs/VGG19_Baseline_BN/LRDecay/BS50/DataAugm/mask_relu_heconstant_LR0.0025/f1/"
        ]
    ]

    # MergeJob(np.asarray(path_vgg).reshape(-1), MergeFinalTrainLogs)
    # input()
    # MergeJob(np.asarray(path_mlp_relu_glorot_he_hc_f1xxx).reshape(-1))
    # PlotAccVsWj(path_mlp_relu_glorot_he_hc_f1xxx, axtitles, path_mlp_relu_glorot_he_hc_f1xxx, baseline=None,
    #             saveas="Outputs/PlotsForPaper/0_MLP_1_05_025_mask_relu_glorot_he_heconst_AccForDifferentInitializations.pdf")

    # MergeJob(np.asarray(path_mlp_relu_glorot_he_hc_baseline_f1).reshape(-1))

    vgg_lth = [

        [
            "Outputs/VGG19_LTH/BS50/mask_relu_he_LR0.0009/f1/"
        ],

        [
            "Outputs/VGG19_LTH/BS50/mask_relu_he_LR0.0009/f1/"
        ],

        [
            "Outputs/VGG19_LTH/BS50/mask_relu_heconstant_LR0.0009/f1/"
        ]
    ]

    MLP_MaxMask1 = [

        [
            "Outputs/LeNet/Baseline/bias_relu_glorot_normal_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_glorot_LR0.0012/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.001/f0.5/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.001/f0.25/",
            "Outputs/MaxMask/lenet/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/"
            # "Outputs/LeNet/mask_relu_glorot_LR0.0012/f1/",
            # "Outputs/LeNet/Standard/nobias_relu_glorot_normal_LR0.0012/f1/"

        ],

        [
            "Outputs/LeNet/Baseline/bias_relu_he_normal_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_he_LR0.0012/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.001/f0.5/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.001/f0.25/",
            "Outputs/MaxMask/lenet/P1_0.5/mask_rs_relu_he_LR0.0003/f1/"
            # "Outputs/LeNet/mask_relu_he_LR0.0012/f1/",
            # "Outputs/LeNet/Standard/nobias_relu_he_normal_LR0.0012/f1/"
        ],

        [
            "Outputs/LeNet/Baseline/bias_relu_heconstant_LR0.0012/f1/",
            "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f0.5/",
            # "Outputs/LeNet/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f0.25/",
            "Outputs/MaxMask/lenet/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/"
            # "Outputs/LeNet/mask_relu_heconstant_LR0.0012/f1/",
            # "Outputs/LeNet/Standard/nobias_relu_heconstant_LR0.0012/f1/"
        ]
    ]

    MLP_MaxMask1LRScan = [

        [
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"
        ],

        [
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/"
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"
        ],

        [
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            # "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/LeNet/LRScan/MinMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"

        ]
    ]

    Conv6_MaxMask1 = [

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.0009/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.0009/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.0009/f1/",

            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.0009/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.0009/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.0009/f1/",

            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/"
        ],

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",

            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/"
        ]
    ]

    Conv2_4_6_MaxMask1 = [

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",

            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",

            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/"
        ],

        [
            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",

            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/"
        ]
    ]

    Conv2_4_6_Baseline = [

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/"
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/"
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/"
        ]
    ]

    Conv2_4_6_Baseline_FreePruning = [

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/"
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/"
        ]
    ]

    Conv2_4_6_Baseline_FreePruning_MinPruning = [

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_glorot_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_glorot_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_glorot_LR0.003/f1/",

            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_glorot_LR0.004/f1/",
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_he_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_he_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_he_LR0.003/f1/",

            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_he_LR0.004/f1/",
        ],

        [
            "Outputs/Baseline/Conv6/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv4/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/Baseline/Conv2/Baseline/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",

            "Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv4/mask_relu_heconstant_LR0.003/f1/",
            "Outputs/Conv2/mask_relu_heconstant_LR0.003/f1/",

            "Outputs/MaxMask/Conv6/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv4/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv2/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
        ]
    ]

    Conv6_MaxMask1LRScan = [

        [
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"
        ],

        [
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/"
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"
        ],

        [
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0001/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0004/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0005/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0006/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0007/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0008/f1/",
            # "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.0009/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.002/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.004/f1/",
            "Outputs/MaxMask/Conv6/LRScan/MaxMask1/P1_0.5/mask_rs_relu_heconstant_LR0.005/f1/"
        ]
    ]

    Baselines = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.0002/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_he_LR0.0002/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/"
        ]
    ]

    FreePruning = [
        [
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/"

        ],

        [
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/"
        ]
    ]

    Conv6_FreePruning = [
        [
            "Outputs/Allruns/FreePruning/Conv6/P1_0.0/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.00025/f1/",
            "Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.00025/f1/"
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.7/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.8/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.9/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.95/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.98/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.99/mask_rs_relu_heconstant_LR0.00025/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.999/mask_rs_relu_heconstant_LR0.00025/f1/"
        ],

        [
            # "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/"

        ],

        [
            # "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            # "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/"
        ]
    ]

    MinPruning = [
        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/"

        ],

        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/"
        ]
    ]

    MinPruning_LR0_001 = [
        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.0/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.1/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.2/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.3/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.4/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.6/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.7/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.8/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.9/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.95/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.98/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.99/mask_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.999/mask_relu_heconstant_LR0.001/f1/"
        ],

        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/"
            # "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/"

        ],

        [
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/"
            # "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/"
        ]
    ]

    MLP_Baseline_FreePruning_MinPruning = [
        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_glorot_LR0.001/f1/",
        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_he_LR0.001/f1/",

        ],

        [
            "Outputs/Allruns/Baseline/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
            "Outputs/Allruns/MinPruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/",
        ]
    ]

    CNN_Baseline_FreePruning_MinPruning = [
        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.0002/f1/",

            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",

            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_glorot_LR0.003/f1/"
        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_he_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_he_LR0.0002/f1/",

            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/",

            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_he_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_he_LR0.003/f1/"

        ],

        [
            "Outputs/Allruns/Baseline/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.0003/f1/",
            # "Outputs/Allruns/Baseline/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.0002/f1/",

            "Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/FreePruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",

            "Outputs/Allruns/MinPruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv4/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/",
            # "Outputs/Allruns/MinPruning/Conv2/P1_0.5/mask_rs_relu_heconstant_LR0.003/f1/"
        ]
    ]

    # MergeJob(np.asarray(vgg_lth).reshape(-1), MergeTrainLogs)

    # MergeJob(np.asarray(MLP_FreePruning_MinPruning_Skewed[0]).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(pathsConv6_Skewed).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(Conv6_MaxMask1).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(Conv6_MaxMask1LRScan[0]).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(Conv2_4_6_MaxMask1).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(Conv2_4_6_Baseline).reshape(-1), MergeTrainLogs)
    # PlotAccVsWj(path_mlp_relu_glorot_he_hc_baseline_f1, axtitles, path_mlp_relu_glorot_he_hc_baseline_f1, baseline=None)  # , saveas="Outputs/PlotsForPaper/0_vgg_AccForDifferentInitializations")
    # PlotAccAndSparsityPerEpoch(path_mlp_relu_glorot_he_hc_baseline_f1, axtitles, path_mlp_relu_glorot_he_hc_baseline_f1, saveas="Outputs/PlotsForPaper/FreeMask/0_MLP")
    # PlotAccAndSparsityPerEpoch(conv2_4_6_mask_relu_glorot_he_heconst, axtitles, conv2_4_6_mask_relu_glorot_he_heconst, saveas="Outputs/PlotsForPaper/FreeMask/0_Conv")

    # PlotAccAndSparsityPerEpoch(MLP_MaxMask1, axtitles, MLP_MaxMask1)#, saveas="Outputs/PlotsForPaper/MaxMask/0_MLP")
    # PlotAccAndSparsityPerEpoch(Conv6_MaxMask1, axtitles, Conv6_MaxMask1, saveas="Outputs/PlotsForPaper/MaxMask/0_Conv2_4_6_NormalVSMaxMask")
    # PlotAccAndSparsityPerEpoch(Conv2_4_6_MaxMask1, axtitles, Conv2_4_6_MaxMask1)#, saveas="Outputs/PlotsForPaper/MaxMask/0_Conv2_4_6_NormalVSMaxMask")

    # MergeJob(np.asarray(Baselines).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(Conv6_FreePruning[0]).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(MinPruning_LR0_001[0]).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(MLP_FreePruning_MinPruning_Skewed).reshape(-1), MergeTrainLogs)
    # MergeJob(np.asarray(MLP_FreePruning[1]).reshape(-1), MergeTrainLogs)

    # PlotAccAndSparsityPerEpoch(Baselines, axtitles, Baselines)
    # PlotAccAndSparsityPerEpoch(FreePruning, axtitles, FreePruning)
    # PlotAccAndSparsityPerEpoch(MinPruning, axtitles, MinPruning)

    test = [
        # [
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.1/mask_rs_relu_heconstant_LR0.00025/f1/", 0.1),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.2/mask_rs_relu_heconstant_LR0.00025/f1/", 0.2),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.3/mask_rs_relu_heconstant_LR0.00025/f1/", 0.3),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.4/mask_rs_relu_heconstant_LR0.00025/f1/", 0.4),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.5/mask_rs_relu_heconstant_LR0.00025/f1/", 0.5),
        #     ("Outputs/Allruns/FreePruning/Conv6/P1_0.6/mask_rs_relu_heconstant_LR0.00025/f1/", 0.6)
        #
        # ]
        # ,

        [
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.0/mask_rs_relu_heconstant_LR0.0004/f1/", 0.0),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.1/mask_rs_relu_heconstant_LR0.001/f1/", 0.1),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.2/mask_rs_relu_heconstant_LR0.001/f1/", 0.2),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.3/mask_rs_relu_heconstant_LR0.001/f1/", 0.3),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.4/mask_rs_relu_heconstant_LR0.001/f1/", 0.4),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.5/mask_rs_relu_heconstant_LR0.001/f1/", 0.5),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.6/mask_rs_relu_heconstant_LR0.001/f1/", 0.6),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.7/mask_rs_relu_heconstant_LR0.001/f1/", 0.7),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.8/mask_rs_relu_heconstant_LR0.001/f1/", 0.8),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.9/mask_rs_relu_heconstant_LR0.001/f1/", 0.9),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.95/mask_rs_relu_heconstant_LR0.001/f1/", 0.95),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.98/mask_rs_relu_heconstant_LR0.001/f1/", 0.98),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.99/mask_rs_relu_heconstant_LR0.001/f1/", 0.99),
            ("Outputs/Allruns/FreePruning/LeNet/P1_0.999/mask_rs_relu_heconstant_LR0.0004/f1/", 0.999)

        ]
    ]
    # PlotAccVsP1(test)

    # PlotAccAndSparsityPerEpoch(MLP_Baseline_FreePruning_MinPruning, axtitles, MLP_Baseline_FreePruning_MinPruning)
    # PlotAccAndSparsityPerEpoch(Conv6_FreePruning, axtitles, Conv6_FreePruning)

    # PlotAccAndSparsityPerEpoch(conv2_4_6_mask_relu_glorot_he_heconst, axtitles, conv2_4_6_mask_relu_glorot_he_heconst)
    # PlotAccAndSparsityPerEpoch(Conv6_MaxMask1LRScan, axtitles, Conv6_MaxMask1LRScan)
    # PlotAccAndSparsityPerEpoch(Conv2_4_6_Baseline_FreePruning_MinPruning, axtitles, Conv2_4_6_Baseline_FreePruning_MinPruning)
    # PlotAccAndSparsityPerEpoch(Baselines, axtitles, Baselines)
    # PlotAccAndSparsityPerEpoch(FreePruning, axtitles, FreePruning)
    # PlotAccAndSparsityPerEpoch(MinPruning, axtitles, MinPruning)

    # PlotAccVsWj(path_vgg, axtitles, path_vgg, baseline=None)  # , saveas="Outputs/PlotsForPaper/0_vgg_AccForDifferentInitializations")
    PlotAccVsWj(vgg_lth, axtitles, vgg_lth, baseline=None)

    # PlotAccVsWj(path_mlp_relu_glorot_he_hc_baseline_f1, axtitles, path_mlp_relu_glorot_he_hc_baseline_f1, baseline=None,
    #             saveas="Outputs/PlotsForPaper/0_MLP_1_05_025_mask_relu_glorot_he_heconst_baseline_AccForDifferentInitializations")

    # PlotAccVsWj(pathsscanlrconv6, axtitles, pathsscanlrconv6, baseline=None)
    # PlotAccVsWj(pathsCNN, axtitles, pathsCNN, baseline=None, saveas="Outputs/PlotsForPaper/0_CNN_AccForDifferentInitializations")

    # done!
    # PlotAccVsWj(pathsMLP, axtitles, pathsMLP, baseline=None, saveas="Outputs/PlotsForPaper/0_MLP_AccForDifferentInitializations")

    return 0


def FiltersToInts(mypath, maskname, weightname, masktype):
    M = pickle.load(open(mypath + maskname, "rb"))
    W = pickle.load(open(mypath + weightname, "rb"))

    def bool2int(x):
        # print(x)
        y = 0
        for i, j in enumerate(x):
            y += j << i
        return y

    def threetoint(x):
        y = 0
        n = len(x) - 1
        for i in range(0, len(x)):
            y += x[i] * np.power(3, n - i)

        return int(y)

    def tonumber(x, mtype):
        if mtype == "mask":
            # print("mask")
            x += 1
            return threetoint(x)

        if mtype == "flip":
            # print("flip")
            x[x < 0] = 0

            return bool2int(x)

    images = []

    nconv = 0
    for w in W:
        if isinstance(w, list):
            continue
        nconv += 1

    lins, cols = utils.SplitToCompactGrid(nconv)
    lins = 2
    cols = 3

    fign, axn = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI(), sharex=True)
    fign.suptitle("Numbers", fontsize=15)
    axn = axn.reshape(-1)

    figm, axm = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI())
    figm.suptitle("Mask", fontsize=15)
    axm = axm.reshape(-1)

    figw, axw = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI())
    figw.suptitle("Weights", fontsize=15)
    axw = axw.reshape(-1)

    figwm, axwm = plt.subplots(lins, cols, figsize=(12, 6), dpi=DPI())
    figwm.suptitle("Weights*Mask", fontsize=15)
    axwm = axwm.reshape(-1)

    maxnumbers = 3 ** 9 if masktype == "mask" else 2 ** 9
    allnumbers = np.zeros(maxnumbers)
    imagenumbers = [np.zeros((3, 3))] * maxnumbers

    c = 0
    for m, w in zip(M, W):
        if isinstance(m, list):
            continue
        print(m.shape)

        if len(m.shape) > 2:
            # wm = (w.astype(np.float) * m)#.astype(np.int)
            wm = w * m
            images_w = []
            images_m = []
            images_wm = []
            # numbers = []
            numbers = np.zeros(maxnumbers)

            # print(np.count_nonzero(m == 1) / m.size)

            for i in range(m.shape[-1]):
                for j in range(m.shape[2]):
                    # print(m[:, :, j, i])
                    images_w.append(w[:, :, j, i])
                    images_m.append(m[:, :, j, i])
                    images_wm.append(wm[:, :, j, i])

                    # images.append(a[:, :, j, i])

                    value = tonumber((wm[:, :, j, i] / np.max(wm)).reshape(-1).astype(int), masktype)

                    # numbers.append(value)
                    numbers[int(value)] += 1
                    allnumbers[value] += 1
                    imagenumbers[value] = w[:, :, j, i]
                    # imagenumbers.append(wm[:, :, j, i])

            random.shuffle(images_m)
            random.shuffle(images_w)
            random.shuffle(images_wm)

            axw[c].imshow(utils.MakeGridOfImages(images_w))
            axm[c].imshow(utils.MakeGridOfImages(images_m))
            axwm[c].imshow(utils.MakeGridOfImages(images_wm))
            axn[c].plot(numbers / np.sum(numbers))
            # axn[c].set_yscale("log")

        c += 1
        # print(M[0][:, :, 0, i],bool2int((M[0][:, :, 0, i]>0).reshape(-1)) )

    # input()

    # print(len(numbers))
    # print(numbers[0:100])

    # plt.hist(np.asarray(numbers), bins=512)
    # nx, ny = utils.SplitToCompactGrid(len(images))
    # gridofimages = utils.MakeGridOfImages(images)
    #
    # for i in range(len(imagenumbers)):
    #     imagenumbers[i] = imagenumbers[i] * numbers[i]
    # imagenumbers[i] = imagenumbers[i] * np.log(numbers[i])

    # print(len(imagenumbers))
    #
    # print(imagenumbers[0])
    # print(imagenumbers[1])
    # print(imagenumbers[0:2])
    # input()

    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI())
    ms = ax.imshow(utils.MakeGridOfImages(imagenumbers))
    # # ax.set_zscale("log")
    fig.colorbar(ms)
    #
    # # plt.imshow(utils.MakeGridOfImages(imagenumbers, 64, 8))

    # ax.plot(allnumbers / np.sum(allnumbers))
    # ax.set_yscale("log")
    # plt.grid()
    plt.show()

    # print(m.shape)
    # print(m[:,:,:,0])

    return 0


def MergeJob(paths, function):
    for p in paths:
        function(p)
    return 0


def binomial():
    n = np.random.randint(100, 1000)
    n = 100
    # n=1000000
    p = 0.5
    s = 1000

    print(n, p)

    for s in np.linspace(1, 100, 20):
        x = np.random.binomial(n, p, int(s)).astype(np.int)
        # h = np.zeros(n)
        # for i in x:
        #     h[i] += 1
        h = [np.equal(x, i).sum() for i in range(n)]
        h = h / np.sum(h)

        plt.plot(h)
    # x = np.random.binomial(n, p, int(s)).astype(np.int)
    # h = np.zeros(n)
    #
    # for i in x:
    #     h[i] += 1
    #
    # plt.plot(h/np.sum(h))

    plt.show()

    return


def pigeons():
    npigeons = 20
    nholes = 400
    bins = np.zeros(nholes)

    for i in range(npigeons):
        bins[np.random.randint(0, nholes)] += 1

    plt.plot(bins)
    plt.show()
    return 0


def probs():
    # Dynamic and Logarithm approach find probability of
    # at least k heads

    from math import log2
    MAX = 100001

    # dp[i] is going to store Log ( i !) in base 2
    dp = [0] * MAX

    def probability(k, n):

        ans = 0  # Initialize result

        # Iterate from k heads to n heads
        for i in range(k, n + 1):
            res = dp[n] - dp[i] - dp[n - i] - n
            ans = ans + pow(2.0, res)

        return ans

    def precompute():

        # Preprocess all the logarithm value on base 2
        for i in range(2, MAX):
            dp[i] = log2(i) + dp[i - 1]

            # Driver code

    precompute()

    # Probability of getting 2 head out of 3 coins
    print(probability(2, 3))

    # Probability of getting 3 head out of 6 coins
    print(probability(3, 6))

    # Probability of getting 500 head out of 10000 coins
    print(probability(500, 10000))

    # this code is contributed by ash264


def ChernoffBound():
    N = 30000
    p = .5
    for d in np.linspace(.01, .1, 10):
        # d = .1
        bounds = []
        for n in range(1, N):
            mu = n * p
            # tight bound
            # cb = np.power(np.exp(d) / np.power(1 + d, 1 + d), mu)

            # looser bound
            cb = np.exp(-d * d * mu / (2))
            bounds.append(cb)

        plt.plot(np.arange(1, N) / N, bounds, label=str(d))

    plt.gca().set_yscale('log')
    plt.gca().legend()
    plt.grid()
    plt.show()

    return 0


def PathSigns():
    Na = 130
    Nb = 170
    Nc = 150

    A = np.random.choice([-1., 1.], Na)
    B = np.random.choice([-1., 1.], Nb)
    C = np.random.choice([-1., 1.], Nc)
    # C=np.asanyarray([1])

    print(A[:10], np.count_nonzero(A > 0))
    print(B[:10], np.count_nonzero(B > 0))
    print(C[:10], np.count_nonzero(C > 0))

    prod = [0, 0]
    sump = 0
    for a in A:
        for b in B:
            for c in C:
                p = a * b * c
                sump += p
                # print(p)
                prod[int(p + 1) // 2] += 1

    print(prod, prod[0] / prod[1], sump)

    return 0


def netsearch(nL, nl):
    # nL = 100
    # nl = 50
    if nL % 2 != 0:
        return

    L = np.random.choice([-1, 1], nL)
    l = np.random.choice([-1, 1], nl)

    L[:nL // 2] = -1
    L[nL // 2:] = 1

    np.random.shuffle(L)

    t = np.zeros_like(l)
    T = np.copy(L)
    j = 0
    i = 0
    while i < len(l) and j < len(L):
        if l[i] == L[j]:
            t[i] = 1
            j += 1
            i += 1
            continue
        else:
            j += 1

    print(L)
    print(l)
    # print(T)
    print(t)
    # print(np.count_nonzero(t > 0), len(l))

    return np.count_nonzero(t > 0)


def TestLevenshtein():
    def levenshtein(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        # print(matrix)
        return (matrix[size_x - 1, size_y - 1])

    N = 12
    r = np.zeros(N)
    for i in range(N):
        s1 = np.random.choice([-1, 1], 21)
        s2 = np.random.choice([-1, 1], 21)

        r[i] = levenshtein(s1, s2)
        print(s1)
        print(s2)
        print(r[i])

    # plt.hist(r)
    # plt.show()


def stringdistances():
    def dist(a1, a2):
        d = 0
        for i, j in zip(a1, a2):
            d += i != j

        return d

    def extend(a1, a2):

        a3 = np.zeros_like(a2)
        for i in range(len(a1)):
            a3[2 * i] = a1[i]
            a3[2 * i + 1] = a1[i]

        return a3

    def dist2(a1, a2):
        d = 0
        c = np.abs(a1 - a2)
        d = []
        for i in range(0, len(c), 2):
            d.append(min(c[i], c[i + 1]))

        return d

    a1 = np.random.choice([-1, 1], 5)
    a2 = np.random.choice([-1, 1], 10)
    a3 = extend(a1, a2)

    print(a1)
    print(a2)
    print(a3)
    print(dist2(a3, a2))

    input()

    def scan(a1, a2):
        d = np.zeros(len(a2) - len(a1))

        for i in range(len(a2) - len(a1)):
            d[i] = dist(a1, a2[i:])

        return d

    d = dist([1, 0, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0, 0, 1, 1])
    a1 = np.random.choice([-1, 1], 5000)
    a2 = np.random.choice([-1, 1], 10000)
    # d = dist(a1, a2)
    # print(a1)
    # print(a2)

    d = scan(a1, a2)
    # print(d/len(a1))

    plt.hist(d)
    plt.show()

    return


def scattertest():
    lins = 100
    cols = 12
    a = np.random.randint(0, 100, (lins, cols))
    for i in range(cols):
        plt.scatter(i * np.ones(a[i].shape[0]), a[i], alpha=0.2)

    plt.show()

    return 0


def MergeData():
    FreePruning_MLP = [
        "Outputs/FreePruning/LeNet/P1_0.5/mask_relu_heconstant_LR0.001/"
    ]

    MergeJob(np.asarray(FreePruning_MLP), MergeFinalTrainLogs)

    return 0


def MakePlotsForPaper():
    # MergeData()
    PlotsAccuracyMLP()
    return


def MakePlotsForSupplementary():
    PlotsSparsities()

    return 0


def oldcode():
    # scattertest()
    # return

    # TestLevenshtein()
    # stringdistances()
    # return
    #
    # N = 2
    # a = np.zeros(N)
    # nL = 10
    # nl = int(nL / 2.)
    # for i in range(N):
    #     r = netsearch(nL, nl)
    #     a[i] = r
    #     # print(r, nl, r == nl, r/nl)
    #
    # # print(a)
    # print(np.count_nonzero(a >= nl * .9), N)
    # print(np.count_nonzero(a == nl), N)
    #
    # return
    # PathSigns()
    # return
    # ChernoffBound()
    # return
    #
    # probs()
    # return
    #
    # pigeons()
    # return 0
    # WriteEquations()
    # FiltersToInts("Outputs/Conv6/flip_relu_binary_LR0.001/f1/",
    #               "Masks_Wj2261184_BS25_ID0ee1fda_PTOnTheFly_SD116750187_AR300_PP1.00000000_PS99.pkl",
    #               "NetworkWeights_ID0ee1fda_SD116750187.pkl", "flip")

    # FiltersToInts("Outputs/Conv6/mask_relu_heconstant_LR0.003/f1/",
    #               "Masks_Wj1279860_BS25_IDa66177c_PTOnTheFly_SD61113131_AR300_PP0.5660_PS99.pkl",
    #               "NetworkWeights_IDa66177c_SD61113131.pkl", "mask")
    #
    # return 0
    mypath = SetMypath()

    if len(sys.argv) > 1:
        mypath = sys.argv[1]

    # CompareLRScans()
    # return 0

    # CompareOneWeightMaskerPermuter()
    # return 0

    # mypaths = [
    #     "Outputs/FullNetResults/Conv6_FullNetResults_relu_glorot_LR0.0005/f1/",
    #     "Outputs/FullNetResults/Conv6_FullNetResults_relu_he_LR0.0005/f1/",
    #     "Outputs/FullNetResults/Conv6_FullNetResults_relu_heconstant_LR0.0005/f1/",
    #     "Outputs/FullNetResults/Conv6_FullNetResults_swish_glorot_LR0.0005/f1/",
    #     "Outputs/FullNetResults/Conv6_FullNetResults_swish_he_LR0.0005/f1/",
    #     "Outputs/FullNetResults/Conv6_FullNetResults_swish_heconstant_LR0.0005/f1/"
    # ]
    # # GetBaselinePerformances(mypaths)
    # # input()

    # MergeJob("Outputs/Conv6/1000Epochs/flip_relu_binary_LR0.001/f1/")
    # return 0
    # input()

    mypaths = [
        # "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
        "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f0.5/",
        "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/"
        # "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f0.5/"
    ]
    fnames = ["MergedTestAcc.npy", "MergedTrainAcc.npy", "MergedValAcc.npy", "MergedRemainingWeights.npy"]  # ,"MergedTestLoss.npy"]
    fnames = ["MergedTestAcc.npy", "MergedRemainingWeights.npy"]

    mypaths = [
        # "Outputs/ZMI/Conv6_ZMI__he_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/"
        # "Outputs/Test_IntervalMean0.1Std0.05_swish_heconstant_LR0.005/f1/"
        # "Conv6_ZMI_Disabled.5_swish_heconstant_LR0.005/f1/",
        # "Conv6_ZMI_Disabled.85_swish_heconstant_LR0.005/f1/",
        # "Conv6_ZMI_Disabled.95_swish_heconstant_LR0.005/f1/"

        "Outputs/Disabled/Conv6_ZMI_Disabled.00_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.10_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.20_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.30_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.40_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.50_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.65_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.75_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.85_swish_heconstant_LR0.005/f1/",
        "Outputs/Disabled/Conv6_ZMI_Disabled.95_swish_heconstant_LR0.005/f1/"
        # "Outputs/Disabled/Conv6_ZMI_Disabled.99_swish_heconstant_LR0.005/f1/",
        # "Outputs/Disabled/Conv6_ZMI_Disabled.99_swish_heconstant_LR0.05/f1/"

        # "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/"
        # "Outputs/ZMI/Conv6_ZMI__glorot_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
        # "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/"
        # "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f0.5/",
        # "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f0.5/"
    ]

    plotAccFscan = False
    if plotAccFscan:
        mypaths = [
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f0.25/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f0.5/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f0.75/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f1/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f1.25/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f1.5/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f1.75/",
            # "Outputs/ZMI/Conv6_ZMI_Fscan_swish_heconstant_LR0.005/f2/",
            "Outputs/Conv6/InitActScan100EpochsNoRescaling_NLSInitHe/relu_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100EpochsNoRescaling_NLSInitHe/swish_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs_NoRescaling/relu_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs_NoRescaling/swish_heconstant_LR0.005/f1/",

            # "Outputs/LeNet/InitActScan100Epochs/relu_glorot_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs/relu_he_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs/relu_heconstant_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs/swish_glorot_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs/swish_he_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs/swish_heconstant_LR0.005/f1/"

            # "Outputs/LeNet/InitActScan100Epochs_Norescling/relu_glorot_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs_Norescling/relu_he_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs_Norescling/relu_heconstant_LR0.005/f1/",
            # # "Outputs/LeNet/InitActScan100Epochs_Norescling/swish_glorot_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs_Norescling/swish_he_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs_Norescling/swish_heconstant_LR0.005/f1/",
            #
            # "Outputs/LeNet/InitActScan100Epochs_Norescaling_NLSInitHe/relu_heconstant_LR0.005/f1/",
            # "Outputs/LeNet/InitActScan100Epochs_Norescaling_NLSInitHe/swish_heconstant_LR0.005/f1/"

            "Outputs/Conv6/InitActScan100Epochs/relu_glorot_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs/relu_he_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs/relu_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs/swish_glorot_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs/swish_he_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs/swish_heconstant_LR0.005/f1/"
        ]

        plotlabels = [
            "Conv6 x 0.25",
            "Conv6 x 0.50",
            "Conv6 x 0.75",
            "Conv6 x 1.0x",
            "Conv6 x 1.25",
            "Conv6 x 1.50",
            "Conv6 x 1.75",
            "Conv6 x 2.00"
        ]

        paths = [
            "Outputs/Conv6/InitActScan100EpochsNoRescaling_NLSInitHe/relu_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100EpochsNoRescaling_NLSInitHe/swish_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs_NoRescaling/relu_heconstant_LR0.005/f1/",
            "Outputs/Conv6/InitActScan100Epochs_NoRescaling/swish_heconstant_LR0.005/f1/"
        ]

        PlotAccFscan(mypaths, mypaths)
        return 0

    plotLRScan = False
    if plotLRScan:
        mypaths = [
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.025/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.025/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.025/f1/"
        ]

        plotlabels = [
            "relu_glorot_LR0.0001/f1/",
            "relu_glorot_LR0.00025/f1/",
            "relu_glorot_LR0.0005/f1/",
            "relu_glorot_LR0.00075/f1/",
            "relu_glorot_LR0.001/f1/",
            "relu_glorot_LR0.0025/f1/",
            "relu_glorot_LR0.005/f1/",
            "relu_glorot_LR0.0075/f1/",
            "relu_glorot_LR0.01/f1/",
            "relu_glorot_LR0.025/f1/",
            "relu_he_LR0.0001/f1/",
            "relu_he_LR0.00025/f1/",
            "relu_he_LR0.0005/f1/",
            "relu_he_LR0.00075/f1/",
            "relu_he_LR0.001/f1/",
            "relu_he_LR0.0025/f1/",
            "relu_he_LR0.005/f1/",
            "relu_he_LR0.0075/f1/",
            "relu_he_LR0.01/f1/",
            "relu_he_LR0.025/f1/",
            "relu_heconstant_LR0.0001/f1/",
            "relu_heconstant_LR0.00025/f1/",
            "relu_heconstant_LR0.0005/f1/",
            "relu_heconstant_LR0.00075/f1/",
            "relu_heconstant_LR0.001/f1/",
            "relu_heconstant_LR0.0025/f1/",
            "relu_heconstant_LR0.005/f1/",
            "relu_heconstant_LR0.0075/f1/",
            "relu_heconstant_LR0.01/f1/",
            "relu_heconstant_LR0.025/f1/",
            "swish_glorot_LR0.0001/f1/",
            "swish_glorot_LR0.00025/f1/",
            "swish_glorot_LR0.0005/f1/",
            "swish_glorot_LR0.00075/f1/",
            "swish_glorot_LR0.001/f1/",
            "swish_glorot_LR0.0025/f1/",
            "swish_glorot_LR0.005/f1/",
            "swish_glorot_LR0.0075/f1/",
            "swish_glorot_LR0.01/f1/",
            "swish_glorot_LR0.025/f1/",
            "swish_he_LR0.0001/f1/",
            "swish_he_LR0.00025/f1/",
            "swish_he_LR0.0005/f1/",
            "swish_he_LR0.00075/f1/",
            "swish_he_LR0.001/f1/",
            "swish_he_LR0.0025/f1/",
            "swish_he_LR0.005/f1/",
            "swish_he_LR0.0075/f1/",
            "swish_he_LR0.01/f1/",
            "swish_he_LR0.025/f1/",
            "swish_heconstant_LR0.0001/f1/",
            "swish_heconstant_LR0.00025/f1/",
            "swish_heconstant_LR0.0005/f1/",
            "swish_heconstant_LR0.00075/f1/",
            "swish_heconstant_LR0.001/f1/",
            "swish_heconstant_LR0.0025/f1/",
            "swish_heconstant_LR0.005/f1/",
            "swish_heconstant_LR0.0075/f1/",
            "swish_heconstant_LR0.01/f1/",
            "swish_heconstant_LR0.025/f1/"
        ]

        PlotAccFscan(mypaths, plotlabels)

        return

    if False:
        mypaths = [
            "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
            "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
            "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/"
        ]

        axtitles = [
            "Glorot Initialization",
            "He Initialization",
            "He Constant Initialization"
        ]

        PlotAccForDiffInitializations(mypaths, axtitles)
        input()

    mypaths = [
        # ["Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f0.5/"],
        # ["Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f0.5/"],
        # ["Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/"]

        # "Outputs/FullNetResults/Conv6_FullNetResults_relu_glorot_LR0.0005/f1/",
        # "Outputs/FullNetResults/Conv6_FullNetResults_relu_he_LR0.0005/f1/",
        # "Outputs/FullNetResults/Conv6_FullNetResults_relu_heconstant_LR0.0005/f1/",
        # "Outputs/FullNetResults/Conv6_FullNetResults_swish_glorot_LR0.0005/f1/",
        # "Outputs/FullNetResults/Conv6_FullNetResults_swish_he_LR0.0005/f1/",
        # "Outputs/FullNetResults/Conv6_FullNetResults_swish_heconstant_LR0.0005/f1/"
        #
        # ["Outputs/FullNetResults/Conv6_FullNetResults_relu_glorot_LR0.0005/f1/",
        #  "Outputs/FullNetResults/Conv6_FullNetResults_swish_glorot_LR0.0005/f1/"],
        #
        # ["Outputs/FullNetResults/Conv6_FullNetResults_relu_he_LR0.0005/f1/",
        #  "Outputs/FullNetResults/Conv6_FullNetResults_swish_he_LR0.0005/f1/"],
        #
        # ["Outputs/FullNetResults/Conv6_FullNetResults_relu_heconstant_LR0.0005/f1/",
        #  "Outputs/FullNetResults/Conv6_FullNetResults_swish_heconstant_LR0.0005/f1/"]

        ["Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_relu_glorot_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_elu_glorot_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_selu_glorot_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI__glorot_LR0.005/f1/"],

        ["Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_relu_he_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_elu_he_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_selu_he_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI__he_LR0.005/f1/"],

        ["Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_relu_heconstant_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_elu_heconstant_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI_selu_heconstant_LR0.005/f1/",
         "Outputs/ZMI/Conv6_ZMI__heconstant_LR0.005/f1/"]

        # ["Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_relu_glorot_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_elu_glorot_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_selu_glorot_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI__glorot_LR0.005/f1/"
        #  ],
        #
        # ["Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_relu_he_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_elu_he_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_selu_he_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI__he_LR0.005/f1/"
        #  ],
        #
        # ["Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_relu_heconstant_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_elu_heconstant_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI_selu_heconstant_LR0.005/f1/",
        #  "Outputs/ZMI/Conv6_ZMI__heconstant_LR0.005/f1/"
        #  ]

    ]

    axtitles = [
        "Glorot Initialization",
        "He Initialization",
        "He Constant Initialization"
    ]

    plotlabels = [

        # ["relu",
        #  "swish"],
        #
        # ["relu",
        #  "swish"],
        #
        # ["relu",
        #  "swish"]

        [
            #    "Outputs/ZMI/LRScan/relu_glorot_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.01/f1/",
            # "Outputs/ZMI/LRScan/relu_glorot_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.025/f1/"
        ],

        [
            #    "Outputs/ZMI/LRScan/relu_he_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.01/f1/",
            # "Outputs/ZMI/LRScan/relu_he_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.025/f1/"
        ],

        [
            #    "Outputs/ZMI/LRScan/relu_heconstant_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.01/f1/",
            # "Outputs/ZMI/LRScan/relu_heconstant_LR0.025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.001/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.005/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.025/f1/"
        ]
    ]

    pmpaths = [

        ["Outputs/PMMasks/Conv6_PMMask_swish_glorot_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_relu_glorot_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_elu_glorot_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_selu_glorot_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask__glorot_LR0.0005/f1/"
         ],

        ["Outputs/PMMasks/Conv6_PMMask_swish_heconstant_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_relu_heconstant_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_elu_heconstant_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_selu_heconstant_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask__heconstant_LR0.0005/f1/"
         ],

        ["Outputs/PMMasks/Conv6_PMMask_swish_he_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_relu_he_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_elu_he_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask_selu_he_LR0.0005/f1/",
         "Outputs/PMMasks/Conv6_PMMask__he_LR0.0005/f1/"
         ]
    ]

    pmpaths = [

        [
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_glorot_LR0.025/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/swish_glorot_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_glorot_LR0.025/f1/"
        ],

        [
            "Outputs/ZMI/LRScan/relu_he_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_he_LR0.025/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/swish_he_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_he_LR0.025/f1/"
        ],

        [
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0001/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.00025/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0005/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.00075/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.001/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0025/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.005/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.0075/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.01/f1/",
            "Outputs/ZMI/LRScan/relu_heconstant_LR0.025/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.0001/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.00025/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.0005/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.00075/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.001/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.0025/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.005/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.0075/f1/",
            # "Outputs/ZMI/LRScan/swish_heconstant_LR0.01/f1/",
            "Outputs/ZMI/LRScan/swish_heconstant_LR0.025/f1/"
        ]
    ]

    # PlotAccForDiffInitializations(pmpaths, axtitles)
    # PlotAccVsWj(mypaths, axtitles, plotlabels, baseline=[0.793, 0.7955, 0.780])
    # PlotAccVsWj(pmpaths, axtitles, plotlabels, baseline=[0.793, 0.7955, 0.780])

    # PlotAccVsWj(pmpaths, axtitles, pmpaths)

    #
    # return

    # mypaths = [
    #     "Outputs/ZMI/Conv6_ZMI__he_LR0.005/f0.5/",
    #     "Outputs/ZMI/Conv6_ZMI__heconstant_LR0.005/f0.5/",
    #     "Outputs/ZMI/Conv6_ZMI__glorot_LR0.005/f0.5/",
    #     "Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f0.5/",
    #     "Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f0.5/",
    #     "Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f0.5/"
    # ]

    # mypaths = [
    # #     # "Outputs/LeNet/ZMI_swish_he_LR0.0005/f0.1/",
    # #     # "Outputs/LeNet/ZMI_swish_heconstant_LR0.0005/f0.1/",
    # #     # "Outputs/LeNet/ZMI_swish_glorot_LR0.0005/f0.1/",
    # #
    #     "Outputs/LeNet/swish_heconstant_LR0.005/f1/",
    #
    #     "Outputs/LeNet/ZMI_swish_he_LR0.0005/f1/",
    #     "Outputs/LeNet/ZMI_swish_heconstant_LR0.0005/f1/",
    #     "Outputs/LeNet/ZMI_swish_glorot_LR0.0005/f1/",
    #
    #     "Outputs/LeNet/ZMI_swish_he_LR0.0005/f0.5/",
    #     "Outputs/LeNet/ZMI_swish_heconstant_LR0.0005/f0.5/",
    #     "Outputs/LeNet/ZMI_swish_glorot_LR0.0005/f0.5/"
    # #
    # #     # "Outputs/LeNet/ZMI_swish_he_LR0.0005/f2/",
    # #     # "Outputs/LeNet/ZMI_swish_heconstant_LR0.0005/f2/",
    # #     # "Outputs/LeNet/ZMI_swish_glorot_LR0.0005/f2/"
    # #
    # ]

    # LoadNetwork("", "")
    # return 0

    mypaths = ["Outputs/Disabled/Conv6_ZMI_Disabled.00_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.10_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.20_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.30_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.40_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.50_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.65_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.75_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.85_swish_heconstant_LR0.005/f1/",
               "Outputs/Disabled/Conv6_ZMI_Disabled.95_swish_heconstant_LR0.005/f1/"]
    PlotMergedFiles(mypaths, fnames, "Outputs/ZMI/Plots/", "1_Disabled.pdf")
    return 0

    # MergeJob()
    # NetworkWeights_ID21136c7_SD26271745.pkl
    # Masks_Wj1174499_BS25_ID9f0c32a_PTOnTheFly_SD49974553_AR300_PP0.5205_PS24
    # NetworkWeights_ID0ca6070_SD22355874
    # PlotWeightMagnitudes("Outputs/Conv6_swish_heconstant_LR0.005/f1/", "0ca6070")

    # NetworkWeights_ID21136c7_SD26271745
    # PlotWeightMagnitudes2("Outputs/Test_swish_he_LR0.005/f1/a/")
    # PlotWeightMagnitudes2("Outputs/Conv4_6/Conv6_swish_heconstant_LR0.005/f1.25/")
    # PlotWeightMagnitudes2("Outputs/ZMI/Test_ZeroMeanInput_DistribMn0.1Std0.02_swish_he_LR0.005/f1/")
    # PlotWeightMagnitudes2("Outputs/ZMI/Conv6_ZeroMeanInputDividedbyMaxminusMin_DistribMn0.1Std0.02_swish_heconstant_LR0.005/f1/")
    # PlotWeightMagnitudes2("Outputs/Disabled/Conv6_ZMI_Disabled.95_swish_heconstant_LR0.005/f1/")
    # PlotMaskEvolution("Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/")

    FinalPercentageOfWeights = True
    if FinalPercentageOfWeights:
        # PlotFinalPercentageOfWeights("Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/")
        PlotFinalPercentageOfWeights("Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/")
        # PlotFinalPercentageOfWeights("Outputs/ZMI/Conv6_ZMI_swish_heconstant_LR0.005/f1/")
        return 0

    # PlotWeightMagnitudes2("Outputs/ZMI/Conv6_ZMI_swish_glorot_LR0.005/f1/")
    # PlotWeightMagnitudes2("Outputs/ZMI/Conv6_ZMI_swish_he_LR0.005/f1/")
    # return 0

    mypath = mypaths[0]

    PlotPruningPercentagesAll(mypath)

    return 0


def main():
    MakePlotsForPaper()


if __name__ == '__main__':
    main()
