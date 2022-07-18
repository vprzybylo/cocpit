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
    """
    Create acc and loss training and validation curves per epoch
    """

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
    num_epochs,
    df_val,
    df_val_unbalanced,
    df_train,
    df_train_unbalanced,
    save_name,
    save_fig=False,
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
    plt.show()
    if save_fig:
        plt.savefig(save_name)
