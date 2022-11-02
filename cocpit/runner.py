"""
train the CNN model(s) and record performance
"""
import time
import cocpit
import cocpit.config as config  # isort:split
from typing import List, Any
from cocpit.plotting_scripts import report as report
import matplotlib.pyplot as plt
import itertools
from tune_sklearn import TuneGridSearchCV
import numpy as np


def set_plt_params() -> None:
    """set matplotlib params"""
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


def flatten(var: np.ndarray[Any, Any]) -> List[Any]:
    """flatten vars from batches"""
    return (
        [item for sublist in list(itertools.chain(*var)) for item in sublist],
    )


def tune_grid_search_cv(
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
) -> TuneGridSearchCV.best_params_:
    """
    inner loop for nested cross validation

    https://www.google.com/search?q=nested+cross+validation&sxsrf=ALiCzsaNa6dvY2-5dB56tcxlNCekUJ2P5Q:1666988629129&tbm=isch&source=iu&ictx=1&vet=1&sa=X&ved=2ahUKEwi3vNXI4IP7AhVfMlkFHbpoB5oQ_h16BAgTEAc&biw=1571&bih=880&dpr=2.2#imgrc=sPazlEcy0--WvM
    """

    tune_search = TuneGridSearchCV(
        c.model,
        config.CONFIG_RAY,
        early_stopping=True,
        max_iters=10,
        use_gpu=True,
        scoring="balanced_accuracy",
        verbose=1,
        n_jobs=config.NUM_WORKERS,
    )  # cv =5 by default

    start = time.time()
    tune_search.fit(f.train_data, f.train_labels)
    end = time.time()
    print("Tune Fit Time:", end - start)
    pred = tune_search.predict(f.val_data)
    accuracy = np.count_nonzero(
        np.array(pred) == np.array(f.val_labels)
    ) / len(pred)
    print("Tune Accuracy:", accuracy)
    return tune_search.best_params_


def record_performance(model_name, kfold, uncertainties, probs, labels, preds):
    """record performance plots and uncertainties"""
    r = report.Report(uncertainties, probs, labels, preds)
    r.conf_matrix(labels, preds)
    r.class_report(model_name, labels, preds, kfold)
    r.uncertainty_prob_scatter(probs, uncertainties)
    r.hist(probs, f"{config.PLOT_DIR}/histogram_probs.png")
    r.hist(uncertainties, f"{config.PLOT_DIR}/histogram_uncertainties.png")


def main(
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
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

    best_params = tune_grid_search_cv(f, c)
    set_plt_params()

    # BEST EPOCH FROM ABOVE
    for epoch in range(best_params["MAX_EPOCHS"]):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            c.model.train() if phase == "train" else c.model.eval()

            if phase == "train":
                train = cocpit.train.Train(
                    f, epoch, best_params["MAX_EPOCHS"], model_name, kfold, c
                )
                train.run(best_params["BATCH_SIZE"])

            else:
                val = cocpit.validate.Validation(
                    f,
                    epoch,
                    best_params["MAX_EPOCHS"],
                    model_name,
                    kfold,
                    val_best_acc,
                    c,
                )
                val_best_acc = val.run(best_params["BATCH_SIZE"])
                if epoch == best_params["MAX_EPOCHS"] - 1:
                    val_labels.append(val.epoch_labels)
                    val_preds.append(val.epoch_preds)
                    val_uncertainties.append(val.epoch_uncertainties)
                    val_probs.append(val.epoch_probs)
            t.print_time_one_epoch()

    # flatten across batches
    val_uncertainties = flatten(val_uncertainties)
    val_probs = flatten(val_probs)
    val_labels = flatten(val_labels)
    val_preds = flatten(val_preds)
    report.pickle_uncertainties()

    record_performance(
        model_name, kfold, val_uncertainties, val_probs, val_labels, val_preds
    )
    try:
        t.print_time_all_epochs()
        t.write_times(model_name, kfold)
    except NameError:
        print("Number of epochs needs to increase")
