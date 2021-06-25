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


def PlotAccuracy(mypath):
    fig, axes = plt.subplots(2, 3, figsize=(27, 10), dpi=60, sharex=True)

    Panels = {"glorot": 0, "he": 1, "heconstant": 2}
    Colors = {"Baseline": 'tab:blue', "FreeFlipping": 'tab:orange', "FreePruning": 'tab:green', "MinPruning": 'tab:red', "MinFlipping": 'tab:purple'}

    for p in Panels.keys():
        axes[0][Panels[p]].set_title(p, fontsize=25)

    Logs = findfiles(mypath, 'TrainLogs.pkl')
    print("working in ", mypath, len(Logs), "files found")

    if len(Logs) == 0:
        return
    # print(Logs)

    AccuracyCurve = {}
    ChangedWeightsCurves = {}

    for l in Logs:
        fname = l.as_posix()

        initializer = fname.split('/')[6]
        traintype = fname.split('/')[2]

        LogFile = pickle.load(open(l, "rb"))
        start = 0
        end = None
        testAccuracy = LogFile['testAccuracy'][start:end]
        total_weights = np.sum(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0])
        # changed_weights = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights
        changed_weights = 100 * (total_weights - np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 2]) / total_weights

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

    selection = None
    start = 0
    step = 20
    epochs = len(AccuracyCurve[list(AccuracyCurve.keys())[0]][0])
    selection = np.append([0, 1, 3, 5], np.arange(10, epochs, step))
    selection2 = np.append([0, 5], np.arange(10, epochs, step))
    widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))

    selection = np.append([0, 5, 10, 20], np.arange(20, epochs, step))
    selection2 = np.append([0,  10], np.arange(20, epochs, step))
    widths = np.append([1, 1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))
    # step=10
    # selection = np.arange(start, epochs, step)
    # selection2 = np.arange(start, epochs, step)
    # widths = 1

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

        violin_acc = np.asarray(AccuracyCurve[p])[:, selection]
        violin_nzw = np.asarray(ChangedWeightsCurves[p])[:, selection]

        testAccuracy_mean = np.mean(AccuracyCurve[p], axis=0)[selection]
        nzw_mean = np.mean(ChangedWeightsCurves[p], axis=0)[selection]

        panel = Panels[initializer]
        label = traintype + "_" + str(len(AccuracyCurve[p]))

        if traintype == "Baseline":
            vp = axes[0][panel].violinplot(dataset=violin_acc, positions=selection, showmeans=True, showextrema=True, widths=widths)
            # vp=axes[0][panel].scatter(selection, testAccuracy_mean, color=color, label=label, linewidth=3)
            color = vp["bodies"][0].get_facecolor().flatten()
            color = 'black'
            # axes[0][panel].fill_between(selection, np.min(AccuracyCurve[p], axis=0)[selection], np.max(AccuracyCurve[p], axis=0)[selection], facecolor=color, alpha=0.1)
            axes[0][panel].plot(selection, testAccuracy_mean, color='black', label=label, linewidth=3)
        else:
            vp = axes[0][panel].violinplot(dataset=violin_acc, positions=selection, showmeans=True, showextrema=True, widths=widths)
            color = vp["bodies"][0].get_facecolor().flatten()

        patches[panel].append(mpatches.Patch(color=color))

        axes[1][panel].violinplot(dataset=violin_nzw, positions=selection, showmeans=True, showextrema=True, widths=widths)
        axes[1][panel].plot(selection, nzw_mean, color=color, label=label)
        labels[panel].append(label)

    for panel in [0, 1, 2]:
        axes[1][panel].legend(patches[panel], labels[panel], ncol=2, fontsize=18)

    networktype = mypath.split('/')[1]
    limits = {"LeNet": [(.95, .985), (-5, 75), (0.00, 0.5), (100 - 0.006, 100.0006)],
              "ResNet": [(.50, .91), (-5, 75), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv2": [(.60, .73), (-5, 75), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv4": [(.60, .81), (-5, 75), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv6": [(.60, .81), (-5, 75), (-0.05, 1.05), (100 - 0.012, 100.0012)]
              }

    axacc = axes[0]
    axacc[0].set_ylabel("Test Accuracy", fontsize=22)

    axspar = axes[1]
    axspar[1].set_xlabel("Epoch", fontsize=22)
    axspar[0].set_ylabel("Pruned/Flipped weights", fontsize=22)

    for ax in axacc:
        ax.set_ylim(limits[networktype][0])
        ax.grid(True)
        ax.set_xticks(selection)
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
        ax.set_xticks(selection2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    fig.tight_layout(pad=1)

    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + ".pdf")
    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + ".png")
    # if socket.gethostname() == "CLJ-C-000CQ":
    #     plt.show()
    # else:
    #     print("not showing the plot, check data folder for outputs")

    return 0


def main():
    mypath = "Outputs/LeNet/"
    mypath = "TestRun/LeNet/"
    # mypath = "TestRun/Conv2/"
    # mypath = "TestRun/Conv4/"
    # mypath = "TestRun/Conv6/"
    # mypath = "TestRun/ResNet/"
    # mypath = "Outputs/ResNet/"
    PlotAccuracy("TestRun/LeNet/")
    PlotAccuracy("TestRun/Conv2/")
    PlotAccuracy("TestRun/Conv4/")
    PlotAccuracy("TestRun/Conv6/")
    PlotAccuracy("TestRun/ResNet/")

    return 0


if __name__ == '__main__':
    main()
