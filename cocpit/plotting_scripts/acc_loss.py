"""
plots called in /notebooks/accuracy_plots.ipynb
"""

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

plt_params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


class AccLossPlot:
    def __init__(self, model_names, num_epochs, savename, colors, new_names):
        self.model_names = model_names
        self.num_models = len(model_names)
        self.num_epochs = num_epochs
        self.savename = savename
        self.colors = colors
        self.new_names = new_names
        self.savefig = False

    def plot_var(self, ax, var, marker, size, acc=True):

        for i in range(self.num_models):
            factor = 100 if acc else 1
            ax.scatter(
                np.arange(1, (self.num_epochs + 1)),
                [i * factor for i in var[i, :]],
                c=self.colors[self.new_names[i]],
                marker=marker,
                s=size,
                label=self.model_names[i],
            )

    def acc_layout(self, ax, label):
        plt.xticks(np.arange(0, self.num_epochs + 1, 5))
        ax.tick_params(axis="y", which="minor", direction="out")
        ax.xaxis.set_tick_params(which="minor", bottom=False)
        ax.title.set_text(f"{label} Data")
        ax.yaxis.set_ticks_position("both")
        ax.minorticks_on()
        if label == "Training":
            ax.set_ylabel("Accuracy [%]")

    def loss_layout(self, ax, label):
        ax.yaxis.set_ticks_position("both")
        ax.minorticks_on()
        ax.tick_params(axis="y", which="minor", direction="out")
        ax.xaxis.set_tick_params(which="minor", bottom=False)
        plt.xlabel("Epoch")
        if label == "Training":
            ax.set_ylabel("Loss")

    def training_acc(self, ax, train_acc):
        self.plot_var(ax, var=train_acc, marker="o", size=35)
        self.acc_layout(ax, label="Training")
        ax.set_ylim([40, 100])
        ax.set_xlim(0, self.num_epochs + 1)
        ax.legend(
            title="Model Architecture:", loc="lower right", prop={"size": 14}, ncol=2
        )

    def validation_acc(self, ax, val_acc):
        self.plot_var(ax, var=val_acc, marker="*", size=55)
        self.acc_layout(ax, label="Validation")
        ax.set_ylim([40, 100])
        ax.set_xlim(0, self.num_epochs + 1)

    def training_loss(self, ax, train_loss):
        self.plot_var(ax, var=train_loss, marker="o", size=35, acc=False)
        self.loss_layout(ax, label="Training")
        ax.set_ylim([0, 2.0])
        ax.set_xlim(0, self.num_epochs + 1)

    def validation_loss(self, ax, val_loss):
        self.plot_var(ax, var=val_loss, marker="*", size=55, acc=False)
        self.loss_layout(ax, label="Validation")
        ax.set_ylim([0, 2.0])
        ax.set_xlim(0, self.num_epochs + 1)

    def savefig(self):
        plt.savefig(self.save_name, dpi=300, bbox_inches="tight")

    def create_plot(self, train_acc, val_acc, train_loss, val_loss):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2,
            2,
            figsize=(13, 8),
        )
        self.training_acc(ax1, train_acc)
        self.validation_acc(ax2, val_acc)
        self.training_loss(ax3, train_loss)
        self.validation_loss(ax4, val_loss)
        plt.tight_layout()
        if self.savefig:
            self.savefig()


def balance_diff_accuracy(
    num_epochs, df_val, df_val_unbalanced, df_train, df_train_unbalanced, save_fig=False
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3))

    for c, item in enumerate(
        [df_val["Accuracy"].values * 100, df_val_unbalanced["Accuracy"].values * 100]
    ):

        color = "blue" if c == 0 else "orange"
        label = "Unweighted" if c == 1 else "Weighted"

        ax1.plot(
            np.arange(1, len(df_train_unbalanced["Loss"]) + 1),
            item,
            color=color,
            label=label,
        )

    for c, item in enumerate([df_val["Loss"].values, df_val_unbalanced["Loss"].values]):

        color = "blue" if c == 0 else "orange"
        label = "Unweighted" if c == 1 else "Weighted"

        ax2.plot(
            np.arange(1, len(df_train_unbalanced["Loss"]) + 1),
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
        plt.savefig(self.save_name)


def sorted_colors(colors, accs):
    return OrderedDict([(el, colors[el]) for el in accs])


def val_acc_fold_bar(colors, kfold, num_models, new_names, val_accs_avg_sort, val_accs):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)
    # fig = plt.figure(figsize=(20,20))
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
        #        box = ax1.get_position()
        #        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax1.legend(fontsize=14)
        ax1.axes.xaxis.set_ticks(np.arange(1, 6, 1))
        ax1.yaxis.set_ticks_position("both")
        ax1.minorticks_on()
        ax1.tick_params(axis="y", which="minor", direction="out")
        # ax1.xaxis.set_tick_params(which='minor', bottom=False)
        ax1.title.set_text("Validation Data Accuracies")

        colors = sorted_colors(colors, val_accs_avg_sort)
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
        # box = ax2.get_position()
        # ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Set number of ticks for x-axis
        ax2.set_xticks(np.arange(1, 10))
        # Set ticks labels for x-axis
        ax2.set_xticklabels(colors.keys(), rotation="vertical")
        ax2.yaxis.set_ticks_position("both")
