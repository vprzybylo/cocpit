"""
train the CNN model(s) and record performance
"""
import time
import cocpit
import cocpit.config as config  # isort:split
from typing import List, Any
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.model_selection import StratifiedKFold
import optuna


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


def flatten(var: np.ndarray) -> List[Any]:
    """flatten vars from batches"""
    return (
        [item for sublist in list(itertools.chain(*var)) for item in sublist],
    )


def model_setup(
    model_name: str, trial=False
) -> cocpit.model_config.ModelConfig:
    """
    Create instances for model configurations and training/validation.
    Runs model.

    Args:
        model_name (str): name of model architecture
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, model_name)
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.create_optimizer(trial)
    c.set_dropout(trial)
    c.to_device()
    return c


def train_val_kfold_split(
    trial,
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
    kfold: int,
):
    """
    inner loop for nested cross validation
    returning best hyperparams
    """
    val_best_accs = []
    skf = StratifiedKFold(
        n_splits=config.KFOLD_INNER, shuffle=True, random_state=42
    )
    data = cocpit.data_loaders.get_data("val")
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        f = cocpit.fold_setup.FoldSetup(
            model_name, kfold, train_indices, val_indices
        )
        f.split_data()
        f.update_save_names()

        c = cocpit.model_setup(model_name, trial)
        (val_best_acc, _, _, _, _,) = train_val(
            f,
            c,
            model_name,
            kfold=kfold,
        )
        val_best_accs.append(val_best_acc)
    return np.mean(val_best_accs)


def train_val(f, c, epochs, batch_size, model_name, kfold):
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
                train.run(batch_size)

            else:
                val = cocpit.validate.Validation(
                    f,
                    epoch,
                    epochs,
                    model_name,
                    kfold,
                    val_best_acc,
                    c,
                )
                val_best_acc = val.run(batch_size)
                if epoch == batch_size - 1:
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
    cocpit.report.pickle_uncertainties()

    try:
        t.print_time_all_epochs()
        t.write_times(model_name, kfold)
    except NameError:
        print("Number of epochs needs to increase")

    return val_best_acc, val_uncertainties, val_probs, val_labels, val_preds


def report_best_trial(study, model_name):
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print(f"Best trial for {model_name}:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)
    return best_trial


def inner_kfold_tune(
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

    set_plt_params()
    study = optuna.create_study(direction="maximize")
    study.optimize(train_val_kfold_split, n_trials=20, timeout=600)
    best_trial = report_best_trial(study, model_name)

    return best_trial
