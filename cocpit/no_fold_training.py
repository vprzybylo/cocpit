"""
train model without folds for cross validation
called in build_model.py
"""

import pandas as pd

import cocpit


def main(
    data,
    params,
    experiment,
    acc_savename_train,
    acc_savename_val,
    metrics_savename,
    save_acc,
    save_model,
    valid_size,
    num_workers,
    num_classes,
):
    """
    create dataloaders
    initialize and train model
    """

    total_size = len(data)
    # randomly split indices for training and validation indices according to valid_size
    if valid_size < 0.01:
        # use all of the data
        train_indices = np.arange(0, total_size)
        random.shuffle(train_indices)
        val_indices = None
    else:
        train_indices, val_indices = train_test_split(
            list(range(total_size)), test_size=valid_size
        )
    # train_val_composition(data, train_indices, val_indices)
    model_savename = (
        f"{params['model_save_dir']}e{max(params['max_epochs'])}_"
        f"bs{max(params['batch_size'])}"
        f"{len(params['model_names'])}model(s)_{tag}.pt"
    )
    val_loader_savename = (
        f"{params['val_loader_save_dir']}e{max(params['max_epochs'])}_"
        f"bs{max(params['batch_size'])}"
        f"{len(params['model_names'])}model(s)_{tag}.pt"
    )

    # DATALOADERS
    train_loader, val_loader = cocpit.data_loaders.create_dataloaders(
        data,
        train_indices,
        val_indices,
        batch_size,
        save_model,
        val_loader_savename,
        class_names=params["class_names"],
        data_dir=params["data_dir"],
        valid_size=valid_size,
        num_workers=num_workers,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    # INITIALIZE MODEL
    model = cocpit.models.initialize_model(model_name, num_classes)

    # TRAIN MODEL
    clf_report = cocpit.train_model.train_model(
        experiment,
        params["log_exp"],
        model,
        i,
        batch_size,
        params["class_names"],
        model_name,
        model_savename,
        acc_savename_train,
        acc_savename_val,
        save_acc,
        save_model,
        dataloaders_dict,
        epochs,
        num_classes,
        valid_size=valid_size,
    )

    if save_acc:
        clf_report.to_csv(metrics_savename, mode="a")


if __name__ == "__main__":

    main(
        data,
        params,
        experiment,
        acc_savename_train,
        acc_savename_val,
        metrics_savename,
        save_acc,
        save_model,
        valid_size,
        num_workers,
        num_classes,
        model_save_dir,
        val_loader_save_dir,
    )
