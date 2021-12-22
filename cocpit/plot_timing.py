import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def time_samples(time_csv, save_name, convert_names, save_fig=False):
    """
    model vs time it takes to process 100, 1,000, and 10,000 samples
    """

    df = pd.read_csv(time_csv, names=["Model", "Samples", "Time"])
    df["Model"].astype(str)
    df["Samples"].astype(int)
    df["Time"].astype(float)
    df.replace(convert_names, inplace=True)
    df = df.set_index("Model")
    df = df.loc[
        [
            "ResNet-18",
            "AlexNet",
            "ResNet-34",
            "Efficient-b0",
            "VGG-16",
            "DenseNet-169",
            "VGG-19",
            "DenseNet-201",
            "ResNet-152",
        ]
    ]
    df.reset_index(inplace=True)

    # Model Samples vs. Time
    ax = sns.catplot(
        data=df,
        kind="bar",
        x="Model",
        y="Time",
        hue="Samples",
        legend=True,
        ci=False,
        palette="icefire",
    )
    ax.set_xticklabels(rotation=90, fontsize=14)
    ax.set(xlabel="Model", ylabel="Time [s]")
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def efficiency_samples(time_csv, save_name, convert_names, save_fig=False):
    """
    model vs efficiency per sample
    """

    df = pd.read_csv(time_csv, names=["Model", "Samples", "Time"])
    df["Model"].astype(str)
    df["Samples"].astype(int)
    df["Time"].astype(float)
    df["Efficiency"] = df["Samples"] / df["Time"]
    df.replace(convert_names, inplace=True)
    df = df.set_index("Model")
    df = df.loc[
        [
            "ResNet-18",
            "AlexNet",
            "ResNet-34",
            "Efficient-b0",
            "VGG-16",
            "DenseNet-169",
            "VGG-19",
            "DenseNet-201",
            "ResNet-152",
        ]
    ]
    df.reset_index(inplace=True)

    # Model Efficiency vs. Time
    ax = sns.catplot(
        data=df,
        kind="bar",
        x="Model",
        y="Efficiency",
        hue="Samples",
        legend=True,
        ci=False,
        palette="icefire",
    )
    ax.set_xticklabels(rotation=90, fontsize=14)
    ax.set(xlabel="Model", ylabel="Efficiency [samples/s]")

    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def model_timing(time_csv, convert_names, colors, save_name, save_fig=False):
    """
    model vs time it took to train
    """
    df = pd.read_csv(time_csv)
    df.replace(convert_names, inplace=True)
    df["Model"].astype(str)
    df["Time"].astype(float)
    df = df.sort_values(by=["Time"])
    sorted_colors = {k: colors[k] for k in df["Model"]}

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    time = df["Time"] / 60
    # ax = time.plot(kind='bar')
    g = sns.barplot(x="Model", y=time, data=df, ci=None, palette=sorted_colors.values())
    g.set_xlabel("Model")
    g.set_ylabel("Training Time [minutes]")
    g.set_xticklabels(df["Model"], rotation=90, fontsize=14)

    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
