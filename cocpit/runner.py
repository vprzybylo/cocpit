"""
train the CNN model(s)
"""
import time
import cocpit
import cocpit.config as config  # isort:split
from typing import List
from cocpit.plotting_scripts import report as report
import matplotlib.pyplot as plt
import pickle

plt_params = {
    "axes.labelsize": "x-large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


def determine_phases() -> List[str]:
    """determine if there is both a training and validation phase

    Returns:
        List[str]: either training or training and validation phase"""
    return ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]


def main(
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
    epochs: int,
    kfold: int,
) -> None:
    """
    Trains and validates a model across epochs, phases, and batches

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        c (model_config.ModelConfig): instance of ModelConfig class
        model_name (str): name of model architecture
        epochs (int): total epochs for training loop
        kfold (int): number of folds use in k-fold cross validation
    """

    since_total = time.time()
    val_best_acc = 0.0
    val_labels = []
    val_preds = []
    val_uncertainties = []
    val_probs = []
    for epoch in range(epochs):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            c.model.train() if phase == "train" else c.model.eval()
            if phase == "train":
                train = cocpit.train.Train(
                    f, epoch, epochs, model_name, kfold, c
                )
                train.run()
            else:
                val = cocpit.validate.Validation(
                    f, epoch, epochs, model_name, kfold, val_best_acc, c
                )
                val_best_acc = val.run()
                if epoch == epochs - 1:
                    val_labels.append(val.epoch_labels)
                    val_preds.append(val.epoch_preds)
                    val_uncertainties.append(val.epoch_uncertainties)
                    val_probs.append(val.epoch_probs)
            t.print_time_one_epoch()

    # flatten across batches
    val_uncertainties = report.flatten(val_uncertainties)
    val_probs = report.flatten(val_probs)
    val_labels = report.flatten(val_labels)
    val_preds = report.flatten(val_preds)

    # plots
    # report.conf_matrix(val_labels, val_preds)
    # report.class_report(model_name, val_labels, val_preds, kfold)
    with open("val_probs.pickle", "wb") as handle:
        pickle.dump(val_probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("val_uncertainties.pickle", "wb") as handle:
        pickle.dump(
            val_uncertainties, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
    with open("val_labels.pickle", "wb") as handle:
        pickle.dump(val_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("val_preds.pickle", "wb") as handle:
        pickle.dump(val_preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # report.uncertainty_prob_scatter(val_probs, val_uncertainties)
    # report.hist(val_probs)
    # report.hist(val_uncertainties)

    try:
        t.print_time_all_epochs()
        t.write_times(model_name, kfold)
    except NameError:
        print("Number of epochs needs to increase")
