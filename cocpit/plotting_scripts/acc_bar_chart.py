"""
plots called in /notebooks/accuracy_plots.ipynb
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt_params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


class BarChart:
    """
    Create a bar chart for batch size vs accuracy.

    Args:
        val_csv (str): CSV file containing batch size and accuracy
        savename (str): absolute path to save figure to
        g (pd.core.groubpy.Groupby): bar chart from grouped df by batch size
    """

    def __init__(self, val_csv, savename):
        self.val_csv: str = val_csv
        self.savename: str = savename
        self.g: pd.core.groupby.GroupBy = None

    def text_above_bar(self) -> None:
        """
        Plot text above bar
        """
        for p in self.g.patches:
            self.g.annotate(
                "%.2f" % p.get_height(),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=12,
                fontfamily=plt.rcParams["font.family"],
                color="k",
                xytext=(0, 8),
                textcoords="offset points",
            )

    def layout(self) -> None:
        """
        Create chart with limits and ticks labels
        """
        self.g = self.groupedvalues.plot.bar(color=self.cmap())
        self.g.set_xlabel(
            "Batch Size",
            fontfamily=plt.rcParams["font.family"],
            fontsize=plt_params["xtick.labelsize"],
        )
        self.g.set_ylabel(
            "Accuracy [%]",
            fontfamily=plt.rcParams["font.family"],
            fontsize=plt_params["xtick.labelsize"],
        )
        self.g.set_ylim(90, 100)
        self.g.set_yticks([90, 92, 94, 96, 98, 100])
        self.g.set_xticks([0, 1, 2, 3, 4, 5])
        self.g.set_xticklabels(
            [32, 64, 128, 256, 512, 1024],
            rotation=90,
            fontfamily=plt.rcParams["font.family"],
            fontsize=plt_params["xtick.labelsize"],
        )
        self.g.set_yticklabels(
            self.g.get_yticks(),
            fontfamily=plt.rcParams["font.family"],
            fontsize=plt_params["xtick.labelsize"],
        )
        self.g.grid(False)

    def cmap(self) -> cm:
        """Get color map for chart

        Returns:
            cmap (matplotlib.cm): normalized color map between vmin and vmax
        """
        cmap = cm.get_cmap("Oranges")
        norm = Normalize(vmin=95.8, vmax=96.9)
        cmap = cmap(norm(self.groupedvalues))
        return cmap

    def savefig(self):
        """Save the figure with resolution and absolute path"""
        plt.savefig(self.save_name, dpi=300, bbox_inches="tight")

    def read_csv(self) -> pd.DataFrame:
        """
        Read CSV columns seen below

        Returns:
            df_val (pd.DataFrame): DataFrame of ["Model", "Epoch", "Batch Size", "Accuracy", "Loss"]
        """
        df_val = pd.read_csv(
            self.val_csv, names=["Model", "Epoch", "Batch Size", "Accuracy", "Loss"]
        )
        df_val["Accuracy"] = df_val["Accuracy"] * 100
        return df_val

    def batch_size_accuracy_bar(self, save_fig=False) -> None:
        """
        Call methods for bar chart of batch size vs max accuracy

        val_csv (str): filename holding validation accuracies
            - all accuracies for each batch size lives in one csv
        """
        df_val = self.read_csv()
        self.groupedvalues = df_val.groupby("Batch Size")["Accuracy"].max()
        self.layout()
        self.text_above_bar()
        if save_fig:
            self.savefig()
