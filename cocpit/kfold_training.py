"""
train model with k folds
called in build_model.py
"""

import pandas as pd

import cocpit


def train_val_composition(data, train_indices, val_indices):
    """
    confirms length of train and test data based on validation %
    """
    train_y = list(map(data.targets.__getitem__, train_indices))
    test_y = list(map(data.targets.__getitem__, val_indices))
    print(len(train_y), len(test_y), len(train_y) + len(test_y))
    print("train counts")
    print(Counter(train_y))
    print("val counts")
    print(Counter(test_y))


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
    split dataset into folds
    create dataloaders
    initialize and train model
    """

    fold_report = []  # holds classification metric performances per kfold

    # preserve the percentage of samples for each class with stratified
    skf = StratifiedKFold(n_splits=params["kfold"], shuffle=True, random_state=42)
    for i, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", i)
        # train_val_composition(data, train_indices, val_indices)

        # saving in terms of kfold (index i)
        model_savename = (
            f"{params['model_save_dir']}e{max(params['max_epochs'])}_"
            f"bs{max(params['batch_size'])}_k{i}_"
            f"{len(params['model_names'])}model(s)_{tag}.pt"
        )
        val_loader_savename = (
            f"{['paramsval_loader_save_dir']}e{max(params['max_epochs'])}_"
            f"bs{max(params['batch_size'])}_k{i}_"
            f"{len(params['model_names'])}model(s)_{tag}.pt"
        )

        # DATALOADERS based on split from StratifiedKFold
        (train_loader, val_loader,) = cocpit.data_loaders.create_dataloaders(
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
        fold_report.append(clf_report)

    # concatenate all metric reports from each fold and model and write
    concat_df = pd.concat(fold_report)
    if save_acc:
        concat_df.to_csv(metrics_savename, mode="a")


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
    )
