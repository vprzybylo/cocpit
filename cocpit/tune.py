"""
train the CNN model(s) and record performance
"""
import itertools
import time
from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold

import cocpit
import cocpit.config as config  # isort:split
from cocpit import model_config as model_config


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
    return ([item for sublist in list(itertools.chain(*var)) for item in sublist],)


def model_setup(model_name: str, trial=False):
    """
    Create instances for model configurations and training/validation.

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
    c: model_config.ModelConfig,
    model_name: str,
    k_outer: int,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
):
    """
    inner loop for nested cross validation
    returning best hyperparams
    """
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    epochs = trial.suggest_int("epochs", 25, 55, step=10, log=False)

    val_best_accs = []
    skf = StratifiedKFold(n_splits=config.KFOLD_INNER, shuffle=True, random_state=42)
    data = cocpit.data_loaders.get_data("val")
    for k_inner, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        f = cocpit.fold_setup.FoldSetup(model_name, k_inner, train_indices, val_indices)
        f.split_data()
        f.update_save_names()

        c = model_setup(model_name, trial)
        (val_best_acc, _, _, _, _,) = run_after_split(
            f, c, model_name, k_outer, k_inner, epochs, batch_size, trial
        )
        val_best_accs.append(val_best_acc.cpu().numpy())

    return np.mean(val_best_accs)


def run_after_split(f, c, model_name, k_outer, k_inner, epochs, batch_size, trial=None):
    since_total = time.time()
    best_acc = 0.0
    labels = []
    preds = []
    uncertainties = []
    probs = []

    for epoch in range(epochs):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            c.model.train() if phase == "train" else c.model.eval()

            if phase == "train":
                train = cocpit.train.Train(
                    f, epoch, epochs, model_name, k_outer, k_inner, c
                )
                train.run(batch_size)

            else:
                val = cocpit.validate.Validation(
                    f,
                    epoch,
                    epochs,
                    model_name,
                    k_outer,
                    k_inner,
                    best_acc,
                    c,
                )
                best_acc = val.run(batch_size)
                if epoch == batch_size - 1:
                    labels.append(val.epoch_labels)
                    preds.append(val.epoch_preds)
                    uncertainties.append(val.epoch_uncertainties)
                    probs.append(val.epoch_probs)
                if trial:
                    trial.report(best_acc, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
            t.print_time_one_epoch()

    # flatten across batches
    uncertainties = flatten(uncertainties)
    probs = flatten(probs)
    labels = flatten(labels)
    preds = flatten(preds)

    try:
        t.print_time_all_epochs()
        t.write_times(model_name, k_inner)
    except NameError:
        print("Number of epochs needs to increase")

    return best_acc, uncertainties, probs, labels, preds


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


def inner_kfold_tune(model_name: str, func: Callable) -> None:
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
    sampler = optuna.samplers.TPESampler(
        seed=10
    )  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        sampler=sampler,
    )
    study.optimize(func, n_trials=3, timeout=600)

    best_trial = report_best_trial(study, model_name)
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{config.PLOT_DIR}/hyperparameter_importance.png")
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{config.PLOT_DIR}/optimization_history.png")

    return best_trial
