"""
plotting functions called in /notebooks/make_plots.ipynb
"""

from collections import OrderedDict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize


def batch_size_accuracy_bar(val_csv, save_name, save_fig=False):
    """
    plot batch size vs max accuracy (20 epochs)
    val_csv (str): filename holding validation accuracies
        - all accuracies for each batch size lives in one csv
    """
    sns.set_style("whitegrid")
    df_val = pd.read_csv(
        val_csv, names=["Model", "Epoch", "Batch Size", "Accuracy", "Loss"]
    )
    df_val["Accuracy"] = df_val["Accuracy"] * 100
    groupedvalues = df_val.groupby("Batch Size")["Accuracy"].max()
    print(groupedvalues)
    my_cmap = cm.get_cmap("icefire")
    my_norm = Normalize(vmin=95.0, vmax=97)
    g = groupedvalues.plot.bar(color=my_cmap(my_norm(groupedvalues)))
    g.set_xlabel("Batch Size", fontsize=14, fontfamily="serif")
    g.set_ylabel("Accuracy [%]", fontsize=14, fontfamily="serif")
    g.set_ylim(90, 100)
    g.set_xticks([0, 1, 2, 3, 4, 5])
    g.set_yticks([90, 92, 94, 96, 98, 100])
    g.set_xticklabels(
        [32, 64, 128, 256, 512, 1024], rotation=90, fontsize=14, fontfamily="serif"
    )
    g.set_yticklabels(g.get_yticks(), size=14, fontfamily="serif")
    g.grid(False)
    # plot text above bar

    for p in g.patches:
        g.annotate(
            "%.2f" % p.get_height(),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=12,
            fontfamily="serif",
            color="k",
            xytext=(0, 8),
            textcoords="offset points",
        )

    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def train_val_acc_loss(
    model_names,
    num_models,
    num_epochs,
    train_accs,
    colors,
    new_names,
    val_accs,
    train_losses,
    val_losses,
    save_name,
    save_fig=False,
):

    """
    training and validation accuracy and loss for each model
    4 scatter subplots
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=(13, 8), sharex=True, sharey=True
    )

    # fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot(2, 2, 1)

    for i in range(num_models):
        ax1.scatter(
            np.arange(1, (num_epochs + 1)),
            [i * 100 for i in train_accs[i, :]],
            c=colors[new_names[i]],
            marker="o",
            s=35,
            label=new_names[i],
        )
    plt.ylabel("Accuracy [%]")
    plt.ylim(40, 100)
    plt.xlim(0, num_epochs + 1)
    ax1.legend(title="Model type:", loc="lower right", prop={"size": 14}, ncol=2)
    # ax1.axes.xaxis.set_ticks([])
    ax1.yaxis.set_ticks_position("both")
    ax1.minorticks_on()
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    ax1.tick_params(axis="y", which="minor", direction="out")
    # ax1.xaxis.set_tick_params(which='minor', bottom=False)
    ax1.title.set_text("Training Data")

    # fig = plt.figure(figsize=(20,5))
    ax2 = plt.subplot(2, 2, 2)
    for i in range(num_models):
        ax2.scatter(
            np.arange(1, (num_epochs + 1)),
            [i * 100 for i in val_accs[i, :]],
            c=colors[new_names[i]],
            marker="*",
            s=55,
            label=model_names[i],
        )
    plt.ylim(40, 100)
    plt.xlim(0, num_epochs + 1)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    # ax2.legend(title='Model type:', loc='best', prop={'size': 10})
    #     ax2.axes.yaxis.set_ticks([])
    #     ax2.axes.xaxis.set_ticks([])
    #     ax2.yaxis.set_ticks_position('both')
    # ax2.minorticks_on()
    ax2.tick_params(axis="y", which="minor", direction="out")
    ax2.xaxis.set_tick_params(which="minor", bottom=False)
    ax2.title.set_text("Validation Data")

    ax3 = plt.subplot(2, 2, 3)
    for i in range(num_models):
        ax3.scatter(
            np.arange(1, (num_epochs + 1)),
            [i for i in train_losses[i, :]],
            c=colors[new_names[i]],
            marker="o",
            s=35,
            label=model_names[i],
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # ax3.legend(title='Model type:', loc='best', prop={'size': 10})
    plt.ylim(0, 2.4)
    plt.xlim(0, num_epochs + 1)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    plt.tight_layout()
    ax3.yaxis.set_ticks_position("both")
    ax3.minorticks_on()
    ax3.tick_params(axis="y", which="minor", direction="out")
    ax3.xaxis.set_tick_params(which="minor", bottom=False)

    ax4 = plt.subplot(2, 2, 4)
    for i in range(num_models):
        ax4.scatter(
            np.arange(1, (num_epochs + 1)),
            [i for i in val_losses[i, :]],
            c=colors[new_names[i]],
            marker="*",
            s=55,
            label=model_names[i],
        )
    plt.xlabel("Epoch")
    # ax4.legend(title='Model type:', loc='best', prop={'size': 10})
    plt.ylim(0, 2.4)
    plt.xlim(0, num_epochs + 1)
    plt.xticks(np.arange(0, num_epochs + 1, 5))
    # ax4.axes.yaxis.set_ticks([])
    plt.tight_layout()
    # ax4.yaxis.set_ticks_position('both')
    ax4.minorticks_on()
    ax4.tick_params(axis="y", which="minor", direction="out")
    ax4.xaxis.set_tick_params(which="minor", bottom=False)

    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def balance_diff_accuracy(
    num_epochs, df_val, df_val_unbalanced, df_train, df_train_unbalanced, save_fig=False
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3))

    for c, item in enumerate(
        [df_val['Accuracy'].values * 100, df_val_unbalanced['Accuracy'].values * 100]
    ):

        color = 'blue' if c == 0 else 'orange'
        label = 'Unweighted' if c == 1 else 'Weighted'

        ax1.plot(
            np.arange(1, len(df_train_unbalanced['Loss']) + 1),
            item,
            color=color,
            label=label,
        )

    for c, item in enumerate([df_val['Loss'].values, df_val_unbalanced['Loss'].values]):

        color = 'blue' if c == 0 else 'orange'
        label = 'Unweighted' if c == 1 else 'Weighted'

        ax2.plot(
            np.arange(1, len(df_train_unbalanced['Loss']) + 1),
            item,
            color=color,
            label=label,
        )
    ax1.set_ylabel("Accuracy [%]")
    ax2.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax2.set_xlabel("Epoch")
    ax1.set_xlim(1, num_epochs)
    ax2.set_xlim(1, num_epochs)
    ax1.set_ylim(85, 100)
    ax1.legend(loc="best", prop={"size": 14})
    ax2.legend(loc="best", prop={"size": 14})

    ax1.yaxis.set_ticks_position("both")
    ax1.minorticks_on()
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_name)


def sorted_colors(colors, accs):
    return OrderedDict([(el, colors[el]) for el in accs])


def val_acc_fold_bar(colors, kfold, num_models, new_names, val_accs_avg_sort, val_accs):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)
    fig.tight_layout(pad=3.0)
    # fig = plt.figure(figsize=(20,20))
    ax1 = plt.subplot(2, 1, 1)

    for i in range(num_models):
        ax1.plot(
            np.arange(1, (kfold + 1)),
            [i * 100 for i in val_accs[i, :, -1]],
            c=colors[new_names[i]],
            marker="o",
            label=new_names[i],
        )
        plt.ylabel("Accuracy [%]")
        plt.xlabel("Fold")
        plt.ylim(70, 100)
        # plt.xlim(1,num_epochs)
        # ax1.legend(title='Model type:', loc='best', prop={'size': 12})
        # Shrink current axis by 20%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14)
        ax1.axes.xaxis.set_ticks(np.arange(1, 6, 1))
        ax1.yaxis.set_ticks_position("both")
        ax1.minorticks_on()
        ax1.tick_params(axis="y", which="minor", direction="out")
        # ax1.xaxis.set_tick_params(which='minor', bottom=False)
        ax1.title.set_text("Validation Data Accuracies")

        colors = sorted_colors(colors, val_accs_avg_sort)
        ax2 = plt.subplot(2, 1, 2)
        plt.bar(
            np.arange(1, num_models + 1),
            [i * 100 for i in val_accs_avg_sort.values()],
            color=colors.values(),
        )
        plt.ylabel("Average Accuracy [%]")
        plt.xlabel("Model Name")
        plt.ylim(85, 100)
        # plt.xlim(1,num_epochs)
        # Shrink current axis by 20%
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        # Set number of ticks for x-axis
        ax2.set_xticks(np.arange(1, 10))
        # Set ticks labels for x-axis
        ax2.set_xticklabels(colors.keys(), rotation="vertical")
        ax2.yaxis.set_ticks_position("both")
