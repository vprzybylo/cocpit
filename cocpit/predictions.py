"""
Predict on a batch of images.
Find incorrect predictions from validation dataloader.
"""

import itertools
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader

import cocpit.config as config
from cocpit import data_loaders
from cocpit.auto_str import auto_str


@auto_str
class BatchPredictions:
    """
    Makes predictions on a given batch of data

    Args:
        imgs (torch.Tensor): loaded imgs on device
        model (torch.nn.parallel.data_parallel.DataParallel): model on device for prediction
        max_preds (List[float]): class with the highest probability across all images in the batch
        probs (List[List[float]]): list of top k probabilities across all images in the batch
        classes (List[List[int]]]): list of top k classes across all images in the batch
    """

    def __init__(
        self,
        imgs: torch.Tensor,
        model,
    ):

        self.imgs = imgs.to(config.DEVICE)
        self.model = model.to(config.DEVICE)
        self.max_preds: List[float] = []
        self.probs: List[float] = []
        self.classes: List[int] = []

    def find_logits(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Vector of raw (non-normalized) predictions
        """
        return self.model(self.imgs)

    def find_max_preds(self) -> None:
        """
        Finds the class with the highest probability across all images in the batch
        """
        _, max_preds = torch.max(self.find_logits(), dim=1)
        # convert back to lists from being on gpus
        self.max_preds = max_preds.cpu().tolist()

    def preds_softmax(self) -> torch.Tensor:
        """
        Applies a softmax function such that probabilities lie in the range [0, 1] and sum to 1.

        Returns:
            torch.Tensor: normalized probabilities
        """
        return F.softmax(self.find_logits(), dim=1)

    def top_k_preds(self, top_k_preds: int = 3) -> None:
        """
        Get top k probabilities and respective classes for each index in the batch for bar chart

        Args:
            top_k_preds (int): the top k probabilities/classes predicted
        """
        topk = self.preds_softmax().cpu().topk(top_k_preds)
        self.probs, self.classes = [e.data.numpy().squeeze().tolist() for e in topk]


class LoaderPredictions:
    """
    Concats all validation loader predictions across batches.
    Finds incorrect predictions.

    Args:
        self.labels (List[int]): list of labels across all batches
        self.paths (List[str]): list of paths across all batches
        self.topk_probs (List[float]): list of topk probabilities across all batches
        self.topk_classes (List[int]): list of topk classes across all batches
        self.max_preds (List[float]): list of max predictions across all batches
        self.wrong_trunc (List[int]): indices of wrong predictions
        self.wrong_idx (List[int]): all indices where the model incorrectly predicted
    """

    def __init__(
        self,
        labels: List[int] = [],
        paths: List[str] = [],
        topk_probs: List[List[float]] = [],
        topk_classes: List[List[int]] = [],
        max_preds: List[List[float]] = [],
        wrong_trunc: List[int] = [],
        wrong_idx: List[int] = [],
    ):
        self.labels = labels
        self.paths = paths
        self.topk_probs = topk_probs
        self.topk_classes = topk_classes
        self.max_preds = max_preds
        self.wrong_trunc = wrong_trunc
        self.wrong_idx = wrong_idx

    def load_model(self) -> torch.nn.DataParallel:
        MODEL_SAVE_DIR = f"{config.BASE_DIR}/saved_models/{config.TAG}/"
        MODEL_SAVENAME = (
            f"{MODEL_SAVE_DIR}e{config.MAX_EPOCHS}_"
            f"bs{config.BATCH_SIZE}_"
            f"k{config.KFOLD}_"
            f"{len(config.MODEL_NAMES)}model(s).pt"
        )
        model = torch.load(MODEL_SAVENAME)
        model.eval()
        return model

    def load_val_loader(self) -> torch.utils.data.dataloader.DataLoader:
        VAL_LOADER_SAVE_DIR = f"{config.BASE_DIR}/saved_val_loaders/{config.TAG}/"

        VAL_LOADER_SAVENAME = (
            f"{VAL_LOADER_SAVE_DIR}e{config.MAX_EPOCHS}_"
            f"bs{config.BATCH_SIZE}_"
            f"k{config.KFOLD}_"
            f"{len(config.MODEL_NAMES)}model(s).pt"
        )
        val_data = torch.load(VAL_LOADER_SAVENAME)

        return data_loaders.create_loader(val_data, batch_size=100, sampler=None)

    def concat(self, var: Any) -> List[Any]:
        """
        Flatten a variable across batches
        """
        return list(itertools.chain(*var))

    def concatenate_loader_vars(self) -> None:
        """
        Flatten arrays from appending in batches
        """
        if self.labels is not None:
            pred_list = [
                self.labels,
                self.paths,
                self.topk_probs,
                self.topk_classes,
                self.max_preds,
            ]
            (
                self.labels,
                self.paths,
                self.topk_probs,
                self.topk_classes,
                self.max_preds,
            ) = map(self.concat, pred_list)

        else:
            pred_list = [
                self.paths,
                self.topk_probs,
                self.topk_classes,
                self.max_preds,
            ]
            (
                self.paths,
                self.topk_probs,
                self.topk_classes,
                self.max_preds,
            ) = map(self.concat, pred_list)

    def append_batch(
        self,
        b: BatchPredictions,
        paths: List[str],
        labels: torch.tensor = None,
    ) -> None:
        """
        Append batch predictions across all possible validation datasets (including k-folds)

        Args:
            b (BatchPredictions): instance of batch predictions class
            paths (List[str]): batch paths
            labels (List[int]): batch labels

        """
        self.topk_probs.append(b.probs)
        self.topk_classes.append(b.classes)
        self.max_preds.append(b.max_preds)
        self.paths.append(paths)
        if labels is not None:
            self.labels.append(labels)

    def find_wrong_indices(self) -> None:
        """
        Find all indices where the model incorrectly predicted
        """
        self.wrong_idx = [
            index
            for index, elem in enumerate(self.max_preds)
            if elem != self.labels[index]
        ]

    def hone_incorrect_predictions(
        self, label_list: dict[str, int], human_label: int, model_label: int
    ) -> None:
        """
        Find indices where human labeled as one thing and model labeled as another

        Args:
            label_list (dict[str, int]): dictionary of class labels mapped to integers
            human_label (int): category of the human label
            model_label (int): category of the model label
        """
        idx_human = np.where(self.labels == human_label)

        [
            self.wrong_trunc.append(i)
            for i in idx_human[0]
            if self.max_preds[i] == model_label
        ]
        cat1 = list(label_list.keys())[list(label_list.values()).index(human_label)]
        cat2 = list(label_list.keys())[list(label_list.values()).index(model_label)]
        if self.wrong_trunc:
            print(
                f"{len(self.wrong_trunc)} wrong predictions between {cat1} and"
                f" {cat2}"
            )
