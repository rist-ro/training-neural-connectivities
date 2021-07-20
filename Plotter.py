import glob
import pickle
import sys
import matplotlib

import numpy as np
from pathlib import Path

import socket

if socket.gethostname() != "CLJ-C-000CQ" and socket.gethostname() != "kneon":
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.set_printoptions(threshold=sys.maxsize)


def findfiles(mypath, fname):
    files = []

    for path in Path(mypath).rglob(fname):
        # print(path)
        files.append(path)
        # print(files[-1])

    return files


def PlotAccuracy(mypath, excluded=None):
    if excluded is None:
        # fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=60, sharex=True)
        fig, axes = plt.subplots(2, 3, figsize=(27, 15), dpi=60, sharex=True)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(27, 10), dpi=60, sharex=True)

    Panels = {"glorot": 0, "he": 1, "heconstant": 2}
    Colors = {"Baseline": 'tab:blue', "FreeFlipping": 'tab:orange', "FreePruning": 'tab:green', "MinPruning": 'tab:red',
              "MinFlipping": 'tab:purple'}

    for p in Panels.keys():
        axes[0][Panels[p]].set_title(p, fontsize=25)

    Logs = findfiles(mypath, 'TrainLogs.pkl')
    print("working in ", mypath, len(Logs), "files found")

    if len(Logs) == 0:
        return
    # print(Logs)

    AccuracyCurve = {}
    ChangedWeightsCurves = {}

    included_runs = []

    for l in Logs:
        fname = l.as_posix()

        initializer = fname.split('/')[6]
        traintype = fname.split('/')[2]

        if excluded is not None:
            if traintype in excluded:
                continue

        if traintype not in included_runs:
            included_runs.append(traintype)

        LogFile = pickle.load(open(l, "rb"))
        start = 0
        end = None
        testAccuracy = LogFile['testAccuracy'][start:end]
        total_weights = np.sum(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0])
        # changed_weights = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights
        changed_weights = (total_weights - np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 2]) / total_weights

        # changed_weights = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights

        # allweights = np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0, 2]

        # print(allweights)
        # print(nonpositive)

        # print(np.asarray(LogFile['neg_zero_pos_masks'][start:end]))

        curvename = traintype + "_" + initializer
        if curvename in AccuracyCurve.keys():
            AccuracyCurve[curvename].append(testAccuracy)
        else:
            AccuracyCurve[curvename] = [testAccuracy]

        if curvename in ChangedWeightsCurves.keys():
            ChangedWeightsCurves[curvename].append(changed_weights)
        else:
            ChangedWeightsCurves[curvename] = [changed_weights]

    patches = [[], [], []]
    labels = [[], [], []]

    datapoint_selection = None
    start = 0
    step = 20
    epochs = len(AccuracyCurve[list(AccuracyCurve.keys())[0]][0])
    datapoint_selection = np.append([0, 1, 3, 5], np.arange(10, epochs, step))
    ticks_selection = np.append([0, 5], np.arange(10, epochs, step))
    widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))

    manual_selection = [0, 5, 10, 17, 25, 50, 75, 100]
    datapoint_selection = [0, 10, 30, 50, 80, 100]
    datapoint_selection = [0, 3, 12, 25, 40, 50, 60, 70, 80, 90, 100]
    datapoint_selection = [0, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    datapoint_selection = [0, 3, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    # datapoint_selection = [0, 3, 6, 10, 18, 32, 45, 65, 85, 120, 150, 170, 180,200]
    datapoint_selection = [0, 3, 8, 15, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    datapoint_selection = [0, 3, 8, 15, 30, 50, 80, 120, 140, 180, 200]
    datapoint_selection = np.arange(20, 201, 20)
    datapoint_selection = np.append([0, 3, 8], datapoint_selection)
    datapoint_selection = [0, 3, 8, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # datapoint_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # datapoint_selection = manual_selection
    # selection = np.append([0, 5, 10, 20], np.logspace(5,7,15,base=2).astype(np.int))
    # selection2 = np.append([0, 5, 10, 20], np.logspace(5,7,15,base=2).astype(np.int))
    # datapoint_selection = np.logspace(0, 5, 6, base=2).astype(np.int)
    # datapoint_selection=np.append(datapoint_selection,np.arange(50,201,20))

    # datapoint_selection=[0,1,2,4,6,9,12,18,27,39,56,81,117,169,200]

    # datapoint_selection=np.append(datapoint_selection,200)
    # ticks_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # ticks_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # ticks_selection = datapoint_selection
    ticks_selection = [0, 8, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # ticks_selection = [0, 12, 25, 40, 60, 80, 100]
    # ticks_selection = np.arange(0, epochs, 20)
    # ticks_selection = np.logspace(0, 3, 15, base=2).astype(np.int)

    widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))
    widths = np.ones_like(datapoint_selection) * 5
    widths[3:] = 9
    widths = np.logspace(2, 3, 1 + len(ticks_selection), base=2).astype(np.int)
    widths = [4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    # step=10
    # selection = np.arange(start, epochs, step)
    # selection2 = np.arange(start, epochs, step)
    # widths = 3

    # print(datapoint_selection)
    # print(widths)

    for p in reversed(list(AccuracyCurve.keys())):
        traintype = p.split('_')[0]
        initializer = p.split('_')[1]
        # if traintype == "Baseline":
        #     selection = np.append([7], np.arange(15, epochs, step))
        #     widths = np.append([1], 3 * np.ones(1 + (epochs - 15) // step, dtype=np.int))

        # selection = np.arange(start, epochs, step)
        # selection = np.append([0,5],np.arange(10, epochs, step))
        # csel=curves[:,selection]
        # print(csel.shape)

        violin_acc = np.asarray(AccuracyCurve[p])[:, datapoint_selection]
        violin_nzw = np.asarray(ChangedWeightsCurves[p])[:, datapoint_selection]

        testAccuracy_mean = np.mean(AccuracyCurve[p], axis=0)[datapoint_selection]
        nzw_mean = np.mean(ChangedWeightsCurves[p], axis=0)[datapoint_selection]

        panel = Panels[initializer]
        label = traintype + "_" + str(len(AccuracyCurve[p]))

        if traintype == "Baseline":
            vp = axes[0][panel].violinplot(dataset=violin_acc, positions=datapoint_selection, showmeans=True,
                                           showextrema=True, widths=widths)
            # vp=axes[0][panel].scatter(selection, testAccuracy_mean, color=color, label=label, linewidth=3)
            color = vp["bodies"][0].get_facecolor().flatten()
            # color = 'black'
            # axes[0][panel].fill_between(selection, np.min(AccuracyCurve[p], axis=0)[selection], np.max(AccuracyCurve[p], axis=0)[selection], facecolor=color, alpha=0.1)
            axes[0][panel].plot(datapoint_selection, testAccuracy_mean, color=color, label=label, linewidth=3,
                                alpha=0.6)
        else:
            vp = axes[0][panel].violinplot(dataset=violin_acc, positions=datapoint_selection, showmeans=True,
                                           showextrema=True, widths=widths)
            color = vp["bodies"][0].get_facecolor().flatten()

        patches[panel].append(mpatches.Patch(color=color))

        axes[1][panel].violinplot(dataset=violin_nzw, positions=datapoint_selection, showmeans=True, showextrema=True,
                                  widths=widths)
        axes[1][panel].plot(datapoint_selection, nzw_mean, color=color, label=label)
        labels[panel].append(label)

    for panel in [0, 1, 2]:
        axes[1][panel].legend(patches[panel], labels[panel], ncol=2, fontsize=18)

    networktype = mypath.split('/')[1]

    limits = {"LeNet": [(.95, .985), (-.05, .9), (0.00, 0.5), (100 - 0.006, 100.0006)],
              "Conv2": [(.55, .72), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv4": [(.55, .78), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv6": [(.56, .82), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "ResNet": [(.30, .94), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)]
              }

    axacc = axes[0]
    axacc[0].set_ylabel("Test Accuracy", fontsize=22)

    axspar = axes[1]
    axspar[1].set_xlabel(networktype + " training epoch", fontsize=22)
    axspar[0].set_ylabel("Pruned/Flipped weights", fontsize=22)

    for ax in axacc:
        ax.set_ylim(limits[networktype][0])
        ax.grid(True)
        ax.set_xticks(datapoint_selection)
        # ax.set_xscale('log',base=10)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    for ax in axspar:
        # ax.legend(fontsize=18, ncol=2)

        ax.set_ylim(limits[networktype][1])
        # ax.set_yticks(np.arange(0,101,10))
        ax.grid(True)
        ax.set_xticks(ticks_selection)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    fig.tight_layout(pad=1)

    excl = ""
    if excluded is not None:
        excl = "_excluded"

    # print(sorted(included_runs)[::-1])
    included_runs = sorted(included_runs)

    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".pdf")
    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".png")
    fig.savefig(mypath.split('/')[0] + "/Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".png")
    fig.savefig(mypath.split('/')[0] + "/Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".pdf")

    if socket.gethostname() == "CLJ-C-000CQ" or socket.gethostname() == "kneon":
        plt.show()
    # else:
    #     print("not showing the plot, check data folder for outputs")

    return 0


def PlotAccuracyBinary(mypath, excluded=None):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=60, sharex=True)

    Panels = {"Conv6": 2, "Conv4": 1, "Conv2": 0}
    Colors = {"heconstant": 'tab:blue', "binary": 'tab:orange'}

    for p in Panels.keys():
        axes[Panels[p]].set_title(p, fontsize=25)

    Logs = findfiles(mypath, 'TrainLogs.pkl')
    print("working in ", mypath, len(Logs), "files found")

    D = {}
    for l in Logs:
        fname = l.as_posix()
        # print(fname)
        root_folder = fname.split('/')[0]
        nettype = fname.split('/')[1]
        traintype = fname.split('/')[2]
        initializer = fname.split('/')[6]
        key = root_folder + "_" + nettype + "_" + traintype + "_" + initializer
        LogFile = pickle.load(open(l, "rb"))
        start = 0
        end = None
        testAccuracy = LogFile['testAccuracy'][start:end]
        # print(key)

        if key in D.keys():
            D[key].append(testAccuracy)
        else:
            D[key] = [testAccuracy]

    k0 = list(D.keys())[0]
    # print(k0)
    # print(D[k0])
    # return
    datapoint_selection = [2, 8, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    widths = [4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

    patches = [[], [], []]
    labels = [[], [], []]

    for k in list(D.keys()):
        nettype = k.split('_')[-3]
        traintype = k.split('_')[-2]
        initializer = k.split('_')[-1]
        if initializer not in ["binary", "heconstant"]:
            continue
        # print(k)

        # print(initializer)

        panel = Panels[nettype]
        # print(type(D[k]))
        # print(len(D[k]))

        acc = np.asarray(D[k])
        print(acc.shape)
        violin_acc = acc[:, datapoint_selection]
        # testAccuracy_mean = np.mean(acc[k], axis=0)[datapoint_selection]

        print(violin_acc.shape)
        vp = axes[panel].violinplot(dataset=violin_acc, positions=datapoint_selection, showmeans=True, showextrema=True, widths=widths)

        # axes[0][panel].plot(datapoint_selection, testAccuracy_mean, color=color, label=label, linewidth=3, alpha=0.6)
        label = traintype + " " + initializer

        if "FreePruning_binary" in k:
            label = "FreePruning - binary weights"

        if "FreePruning_heconstant" in k:
            label = "FreePruning - heconstant"

        if "Baseline_heconstant" in k:
            label = "Baseline - heconstant"

        color = vp["bodies"][0].get_facecolor().flatten()
        patches[panel].append(mpatches.Patch(color=color))
        labels[panel].append(label)

    axes[1].legend(patches[panel], labels[panel], ncol=1, fontsize=18, loc=8)

    # print(D[k])

    # plt.show()
    # return
    #
    # if len(Logs) == 0:
    #     return
    # # print(Logs)
    #
    # AccuracyCurve = {}
    # ChangedWeightsCurves = {}
    #
    # included_runs = []
    #
    # for l in Logs:
    #     fname = l.as_posix()
    #
    #     initializer = fname.split('/')[6]
    #     traintype = fname.split('/')[2]
    #
    #     if excluded is not None:
    #         if traintype in excluded:
    #             continue
    #
    #     if traintype not in included_runs:
    #         included_runs.append(traintype)
    #
    #     LogFile = pickle.load(open(l, "rb"))
    #     start = 0
    #     end = None
    #     testAccuracy = LogFile['testAccuracy'][start:end]
    #     total_weights = np.sum(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0])
    #     # changed_weights = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights
    #     changed_weights = (total_weights - np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 2]) / total_weights
    #
    #     # changed_weights = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights
    #
    #     # allweights = np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0, 2]
    #
    #     # print(allweights)
    #     # print(nonpositive)
    #
    #     # print(np.asarray(LogFile['neg_zero_pos_masks'][start:end]))
    #
    #     curvename = traintype + "_" + initializer
    #     if curvename in AccuracyCurve.keys():
    #         AccuracyCurve[curvename].append(testAccuracy)
    #     else:
    #         AccuracyCurve[curvename] = [testAccuracy]
    #
    #     if curvename in ChangedWeightsCurves.keys():
    #         ChangedWeightsCurves[curvename].append(changed_weights)
    #     else:
    #         ChangedWeightsCurves[curvename] = [changed_weights]
    #
    # patches = [[], [], []]
    # labels = [[], [], []]
    #
    # datapoint_selection = None
    # start = 0
    # step = 20
    # epochs = len(AccuracyCurve[list(AccuracyCurve.keys())[0]][0])
    # datapoint_selection = np.append([0, 1, 3, 5], np.arange(10, epochs, step))
    # ticks_selection = np.append([0, 5], np.arange(10, epochs, step))
    # widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))
    #
    # manual_selection = [0, 5, 10, 17, 25, 50, 75, 100]
    # datapoint_selection = [0, 10, 30, 50, 80, 100]
    # datapoint_selection = [0, 3, 12, 25, 40, 50, 60, 70, 80, 90, 100]
    # datapoint_selection = [0, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # datapoint_selection = [0, 3, 6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    # # datapoint_selection = [0, 3, 6, 10, 18, 32, 45, 65, 85, 120, 150, 170, 180,200]
    # datapoint_selection = [0, 3, 8, 15, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # datapoint_selection = [0, 3, 8, 15, 30, 50, 80, 120, 140, 180, 200]
    # datapoint_selection = np.arange(20, 201, 20)
    # datapoint_selection = np.append([0, 3, 8], datapoint_selection)
    # datapoint_selection = [0, 3, 8, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    #
    # # datapoint_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # # datapoint_selection = manual_selection
    # # selection = np.append([0, 5, 10, 20], np.logspace(5,7,15,base=2).astype(np.int))
    # # selection2 = np.append([0, 5, 10, 20], np.logspace(5,7,15,base=2).astype(np.int))
    # # datapoint_selection = np.logspace(0, 5, 6, base=2).astype(np.int)
    # # datapoint_selection=np.append(datapoint_selection,np.arange(50,201,20))
    #
    # # datapoint_selection=[0,1,2,4,6,9,12,18,27,39,56,81,117,169,200]
    #
    # # datapoint_selection=np.append(datapoint_selection,200)
    # # ticks_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # # ticks_selection = np.append([0, 5, 10], np.arange(20, epochs, step))
    # # ticks_selection = datapoint_selection
    # ticks_selection = [0, 8, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # # ticks_selection = [0, 12, 25, 40, 60, 80, 100]
    # # ticks_selection = np.arange(0, epochs, 20)
    # # ticks_selection = np.logspace(0, 3, 15, base=2).astype(np.int)
    #
    # widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))
    # widths = np.ones_like(datapoint_selection) * 5
    # widths[3:] = 9
    # widths = np.logspace(2, 3, 1 + len(ticks_selection), base=2).astype(np.int)
    # widths = [4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    # # step=10
    # # selection = np.arange(start, epochs, step)
    # # selection2 = np.arange(start, epochs, step)
    # # widths = 3
    #
    # # print(datapoint_selection)
    # # print(widths)
    #
    # print(AccuracyCurve.keys())
    # print(AccuracyCurve)
    # input()
    # for p in reversed(list(AccuracyCurve.keys())):
    #     traintype = p.split('_')[0]
    #     initializer = p.split('_')[1]
    #     # nettype=
    #     # if traintype == "Baseline":
    #     #     selection = np.append([7], np.arange(15, epochs, step))
    #     #     widths = np.append([1], 3 * np.ones(1 + (epochs - 15) // step, dtype=np.int))
    #
    #     # selection = np.arange(start, epochs, step)
    #     # selection = np.append([0,5],np.arange(10, epochs, step))
    #     # csel=curves[:,selection]
    #     # print(csel.shape)
    #
    #     violin_acc = np.asarray(AccuracyCurve[p])[:, datapoint_selection]
    #     violin_nzw = np.asarray(ChangedWeightsCurves[p])[:, datapoint_selection]
    #
    #     testAccuracy_mean = np.mean(AccuracyCurve[p], axis=0)[datapoint_selection]
    #     nzw_mean = np.mean(ChangedWeightsCurves[p], axis=0)[datapoint_selection]
    #
    #     panel = Panels[initializer]
    #     label = traintype + "_" + str(len(AccuracyCurve[p]))
    #
    #     if traintype == "Baseline":
    #         vp = axes[0][panel].violinplot(dataset=violin_acc, positions=datapoint_selection, showmeans=True, showextrema=True, widths=widths)
    #         # vp=axes[0][panel].scatter(selection, testAccuracy_mean, color=color, label=label, linewidth=3)
    #         color = vp["bodies"][0].get_facecolor().flatten()
    #         # color = 'black'
    #         # axes[0][panel].fill_between(selection, np.min(AccuracyCurve[p], axis=0)[selection], np.max(AccuracyCurve[p], axis=0)[selection], facecolor=color, alpha=0.1)
    #         axes[0][panel].plot(datapoint_selection, testAccuracy_mean, color=color, label=label, linewidth=3, alpha=0.6)
    #     else:
    #         vp = axes[0][panel].violinplot(dataset=violin_acc, positions=datapoint_selection, showmeans=True, showextrema=True, widths=widths)
    #         color = vp["bodies"][0].get_facecolor().flatten()
    #
    #     patches[panel].append(mpatches.Patch(color=color))
    #
    #     axes[1][panel].violinplot(dataset=violin_nzw, positions=datapoint_selection, showmeans=True, showextrema=True, widths=widths)
    #     axes[1][panel].plot(datapoint_selection, nzw_mean, color=color, label=label)
    #     labels[panel].append(label)
    #
    # for panel in [0, 1, 2]:
    #     axes[1][panel].legend(patches[panel], labels[panel], ncol=2, fontsize=18)

    networktype = mypath.split('/')[1]

    limits = {"LeNet": [(.95, .985), (-.05, .9), (0.00, 0.5), (100 - 0.006, 100.0006)],
              "Conv2": [(.55, .72), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv4": [(.55, .78), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv6": [(.56, .82), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "ResNet": [(.30, .94), (-.05, .9), (-0.05, 1.05), (100 - 0.012, 100.0012)]
              }

    axes[0].set_ylabel("Test Accuracy", fontsize=22)
    axes[1].set_xlabel("Training epoch", fontsize=22)

    for ax in axes:
        ax.set_ylim((0.6, .82))
        ax.grid(True)
        ax.set_xticks(datapoint_selection)
        # ax.set_xscale('log',base=10)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    fig.tight_layout(pad=1)
    #
    # excl = ""
    # if excluded is not None:
    #     excl = "_excluded"
    #
    # # print(sorted(included_runs)[::-1])
    # included_runs = sorted(included_runs)
    #
    fig.savefig(mypath + "Accuracy_Binary" ".png")
    fig.savefig(mypath + "Accuracy_Binary" ".pdf")
    # fig.savefig(mypath.split('/')[0] + "/Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".png")
    # fig.savefig(mypath.split('/')[0] + "/Accuracy_Sparsity" + networktype + "_" + "_".join(included_runs) + ".pdf")

    # if socket.gethostname() == "CLJ-C-000CQ":
    plt.show()
    # else:
    #     print("not showing the plot, check data folder for outputs")

    return 0


def PlotAccuracyP1(mypath, included_nets=None, included_p1=None, tickallpoints=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=60, sharex=True)

    Logs = findfiles(mypath, 'TrainLogs.pkl')
    print("working in ", mypath, len(Logs), "files found")

    net_dict = {}
    datapoint_selection = []
    for l in Logs:
        fname = l.as_posix()
        nettype = fname.split('/')[1]
        p1 = float(fname.split('/')[-7].split('_')[-1])
        if nettype not in included_nets:
            continue

        if included_p1 is not None and p1 not in included_p1:
            continue

        datapoint_selection.append(p1)

        # key = nettype + "_" + str(p1)
        LogFile = pickle.load(open(l, "rb"))
        start = -1
        end = None
        testAccuracy = np.max(LogFile['testAccuracy'])
        total_weights = np.sum(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0])
        changed_weights = (total_weights - np.min(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 2])) / total_weights

        if nettype in net_dict.keys():
            if p1 in net_dict[nettype].keys():
                net_dict[nettype][p1].append([testAccuracy, changed_weights])
            else:
                net_dict[nettype][p1] = [[testAccuracy, changed_weights]]
        else:
            net_dict[nettype] = {}

    print(net_dict)
    patches = []
    labels = []
    for nt in sorted(list(net_dict.keys())):
        positions = sorted(list(net_dict[nt].keys()))
        # positions = np.asarray(list(net_dict[nt].keys()))
        # positions = np.log10(positions)
        accuracies = []
        chweights = []
        for p in positions:
            # print(p, net_dict[nt][p])
            acc_nzw = np.asarray(net_dict[nt][p])
            acc = acc_nzw[:, 0]
            nzw = acc_nzw[:, 1]
            accuracies.append(acc)
            chweights.append(nzw)

        positions = np.asarray(sorted(list(net_dict[nt].keys())))
        log_positions = np.log10(positions)
        vp = axes[0].violinplot(dataset=accuracies, positions=log_positions, showmeans=True, showextrema=True, widths=.1)
        color = vp["bodies"][0].get_facecolor().flatten()
        patches.append(mpatches.Patch(color=color))
        labels.append(nt)

        axes[1].violinplot(dataset=chweights, positions=log_positions, showmeans=True, showextrema=True, widths=.1)
        # print(acc)
        # print(nzw)
        # print(np.asarray(accuracies).shape)

    axes[0].set_ylabel("Test Accuracy", fontsize=22)
    axes[1].legend(patches, labels, ncol=1, fontsize=18, loc=2)

    # axes[0].set_xticks([-2, -1, -0.3, 0])
    # axes[0].set_xticklabels([0.01, 0.1, 0.5, 1])

    axes[1].set_xlabel("Probability of chosing a positive value", fontsize=22)
    axes[1].set_ylabel("Pruned weights", fontsize=22)
    axes[1].grid(True)

    # print(np.where(positions == 0.5)[0][0])
    # input()
    tickpos = [0, 1, np.where(positions == 0.5)[0][0], -1]
    if tickallpoints:
        tickpos=np.arange(0,len(positions))

    print(positions)
    for ax in axes:
        ax.grid(True)
        ax.set_xticks((log_positions[tickpos] * 1000).astype(int) / 1000)
        ax.set_xticklabels((positions[tickpos] * 1000).astype(int) / 1000)
    # axes[1].set_xticks([-2, -1,-0.6989, -0.3, 0])
    # axes[1].set_xticklabels(positions)
    # axes[1].set_xticks([-2, -1, -0.3, 0])
    # axes[1].set_xticklabels([0.01, 0.1, 0.5, 1])

    fig.tight_layout(pad=1)
    plt.show()
    return


def main():
    mypath = "Outputs/LeNet/"
    mypath = "TestRun/LeNet/"
    # mypath = "TestRun/Conv2/"
    # mypath = "TestRun/Conv4/"
    # mypath = "TestRun/Conv6/"
    # mypath = "TestRun/ResNet/"
    # mypath = "Outputs/ResNet/"
    # PlotAccuracy("Run_07_07/LeNet/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_07/Conv2/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_07/Conv4/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_07/Conv6/", excluded=["FreeFlipping", "MinFlipping"])
    #
    # PlotAccuracy("Run_07_07/LeNet/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_07/Conv2/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_07/Conv4/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_07/Conv6/", excluded=["FreePruning", "MinPruning"])

    # PlotAccuracy("Run_07_07/LeNet/")
    # PlotAccuracy("Run_07_07/Conv2/")
    # PlotAccuracy("Run_07_07/Conv4/")
    # PlotAccuracy("Run_07_07/Conv6/")
    # PlotAccuracy("Run_07_07/ResNet/")
    #
    # PlotAccuracy("Run_07_08/LeNet/")
    # PlotAccuracy("Run_07_08/Conv2/")
    # PlotAccuracy("Run_07_08/Conv4/")
    # PlotAccuracy("Run_07_08/Conv6/")
    # PlotAccuracy("Run_07_08/ResNet/")

    # PlotAccuracy("Run_07_09/LeNet/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_09/Conv2/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_09/Conv4/", excluded=["FreeFlipping", "MinFlipping"])
    # PlotAccuracy("Run_07_09/Conv6/", excluded=["FreeFlipping", "MinFlipping"])
    #
    # PlotAccuracy("Run_07_09/LeNet/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_09/Conv2/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_09/Conv4/", excluded=["FreePruning", "MinPruning"])
    # PlotAccuracy("Run_07_09/Conv6/", excluded=["FreePruning", "MinPruning"])

    # PlotAccuracy("Run_07_09/LeNet/")
    # PlotAccuracy("Run_07_09/Conv2/")
    # PlotAccuracy("Run_07_09/Conv4/")
    # PlotAccuracy("Run_07_09/Conv6/")

    # PlotAccuracyBinary("test_binary/")

    PlotAccuracyP1("Run_07_19_p1scan/", included_nets=["Conv2", "Conv4", "Conv6"], included_p1=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9], tickallpoints=True)
    PlotAccuracyP1("Run_07_19_p1scan/", included_nets=["LeNet"])

    return 0


if __name__ == '__main__':
    main()
