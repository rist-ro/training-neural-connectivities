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

    print("working in ", mypath)
    Logs = findfiles(mypath, 'TrainLogs.pkl')
    if len(Logs) == 0:
        return
    # print(Logs)

    AccCurve = {}
    NZWCurves = {}

    for l in Logs:
        fname = l.as_posix()

        initializer = fname.split('/')[6]
        traintype = fname.split('/')[2]

        LogFile = pickle.load(open(l, "rb"))
        start = 0
        end = None
        testAccuracy = LogFile['testAccuracy'][start:end]
        total_weights = np.sum(np.asarray(LogFile['neg_zero_pos_masks'][start:end])[0])
        nzw = 100 * np.asarray(LogFile['neg_zero_pos_masks'][start:end])[:, 1] / total_weights

        curvename = traintype + "_" + initializer
        if curvename in AccCurve.keys():
            AccCurve[curvename].append(testAccuracy)
        else:
            AccCurve[curvename] = [testAccuracy]

        if curvename in NZWCurves.keys():
            NZWCurves[curvename].append(nzw)
        else:
            NZWCurves[curvename] = [nzw]

    patches = [[], [], []]
    labels = [[], [], []]

    selection = None
    start = 0
    step = 10
    widths = 3
    epochs = len(AccCurve[list(AccCurve.keys())[0]][0])
    selection = np.append([1, 3, 5], np.arange(10, epochs, step))
    # selection = np.arange(start,epochs,1)
    selection2 = np.append([1], np.arange(10, epochs, step))
    # selection2 = np.arange(start,epochs,1)
    widths = np.append([1, 1, 1], 5 * np.ones(1 + (epochs - 10) // step, dtype=np.int))
    print(selection)
    print(widths)
    print(AccCurve)
    for p in AccCurve.keys():
        traintype = p.split('_')[0]
        initializer = p.split('_')[1]

        # epochs = len(AccCurve[p][0])
        # selection = np.arange(start, epochs, step)
        # selection = np.append([0,5],np.arange(10, epochs, step))
        # print(AccCurve[p][0])
        print(selection)
        # curves = np.asarray(AccCurve[p])
        # csel=curves[:,selection]
        # print(csel.shape)

        violin_acc = np.asarray(AccCurve[p])[:, selection]
        violin_nzw = np.asarray(NZWCurves[p])[:, selection]

        testAccuracy_mean = np.mean(AccCurve[p], axis=0)[selection]
        nzw_mean = np.mean(NZWCurves[p], axis=0)[selection]

        panel = Panels[initializer]
        label = traintype + "_" + str(len(AccCurve[p]))

        p = axes[0][panel].violinplot(dataset=violin_acc, positions=selection, showmeans=True, showextrema=True, widths=widths)
        color = p["bodies"][0].get_facecolor().flatten()

        patches[panel].append(mpatches.Patch(color=color))
        # axes[0][panel].plot(selection, testAccuracy_mean, color=color, label=label)

        axes[1][panel].violinplot(dataset=violin_nzw, positions=selection, showmeans=True, showextrema=True, widths=widths)
        axes[1][panel].plot(selection, nzw_mean, color=color, label=label)
        labels[panel].append(label)

    for panel in [0, 1, 2]:
        axes[1][panel].legend(patches[panel], labels[panel], ncol=2, fontsize=18)

    networktype = mypath.split('/')[1]
    limits = {"LeNet": [(.945, .985), (-5, 105), (0.00, 0.5), (100 - 0.006, 100.0006)],
              "ResNet": [(.70, .91), (-5, 105), (-0.05, 1.05), (100 - 0.012, 100.0012)],
              "Conv2": [(.70, .91), (-5, 105), (-0.05, 1.05), (100 - 0.012, 100.0012)]
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
        ax.grid(True)
        ax.set_xticks(selection2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)

    fig.tight_layout(pad=1)

    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + ".pdf")
    fig.savefig(mypath + "Accuracy_Sparsity" + networktype + ".png")
    if socket.gethostname() == "CLJ-C-000CQ":
        plt.show()
    else:
        print("not showing the plot, check data folder for outputs")

    return 0


def main():
    mypath = "Outputs/LeNet/"
    # mypath = "Outputs/ResNet/"
    PlotAccuracy(mypath)

    return 0


if __name__ == '__main__':
    main()
