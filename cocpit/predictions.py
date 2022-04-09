"""
Predict on a batch of images.
Find incorrect predictions from validation dataloader.
To be used with gui.py in notebooks/gui_move_wrong_predictions.ipynb
"""

import numpy as np
import torch
import cocpit
import cocpit.config as config
from cocpit.auto_str import auto_str

import torch.nn.functional as F
import itertools


@auto_str
class BatchPredictions:
    """finds predictions for a given batch of test data (no label)"""

    def __init__(
        self,
        imgs: torch.Tensor,
        model,  #: torch.nn.parallel.data_parallel.DataParallel
    ):

        self.imgs = imgs.to(config.DEVICE)
        self.model = model.to(config.DEVICE)
        self.max_preds = None
        self.probs = None
        self.classes = None

    def find_logits(self) -> torch.Tensor:
        return self.model(self.imgs)

    def find_max_preds(self) -> None:
        """find the class with the highest probability
        across all images in the batch"""
        _, max_preds = torch.max(self.find_logits(), dim=1)
        # convert back to lists from being on gpus
        self.max_preds = max_preds.cpu().tolist()

    def preds_softmax(self) -> torch.Tensor:
        return F.softmax(self.find_logits(), dim=1)

    def top_k_preds(self, top_k_preds: int = 3) -> None:
        """get top k predictions for each index in the batch for bar chart"""
        topk = self.preds_softmax().cpu().topk(top_k_preds)
        self.probs, self.classes = [e.data.numpy().squeeze().tolist() for e in topk]


class LoaderPredictions:
    """
    In gui.py, use a model and validation dataloader to check predictions.
    Loops through all validation loader predictions but only saves incorrect predictions.
    The incorrect predictions are loaded into a gui so that a user can decide
    whether the label is wrong upon second look (i.e., the model is right).
    The image is automatically moved upon choosing a class from the dropdown menu.
    """

    def __init__(self):
        self.all_labels = []
        self.all_paths = []
        self.all_topk_probs = []
        self.all_topk_classes = []
        self.all_max_preds = []
        self.wrong_trunc = []
        self.wrong_idx = []
        self.labels = True

    def load_model(self, fold: int):  # -> torch.nn.parallel.data_parallel.DataParallel:
        model = torch.load(
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_bs{config.BATCH_SIZE}"
            f"_k{fold}_vgg16.pt"
        ).cuda()
        model.eval()
        return model

    def load_val_loader(self, fold: int):  # -> torch.utils.data.dataloader.DataLoader:
        val_data = torch.load(
            f"{config.VAL_LOADER_SAVE_DIR}e{config.MAX_EPOCHS}_val_loader{int(config.VALID_SIZE*100)}_bs{config.BATCH_SIZE}_k{fold}_vgg16.pt"
        )
        val_loader = cocpit.data_loaders.create_loader(
            val_data, batch_size=100, sampler=None
        )
        return val_loader

    def concat(self, var) -> np.ndarray:
        return np.asarray(list(itertools.chain(*var)))

    def concatenate_loader_vars(
        self,
    ) -> None:
        """
        flatten arrays from appending in batches
        """
        if self.labels is None:
            pred_list = [
                self.all_paths,
                self.all_topk_probs,
                self.all_topk_classes,
                self.all_max_preds,
            ]
            (
                self.all_paths,
                self.all_topk_probs,
                self.all_topk_classes,
                self.all_max_preds,
            ) = map(self.concat, pred_list)
        else:
            pred_list = [
                self.all_labels,
                self.all_paths,
                self.all_topk_probs,
                self.all_topk_classes,
                self.all_max_preds,
            ]
            (
                self.all_labels,
                self.all_paths,
                self.all_topk_probs,
                self.all_topk_classes,
                self.all_max_preds,
            ) = map(self.concat, pred_list)

    def find_wrong_indices(self) -> None:
        """ """
        self.wrong_idx = [
            index
            for index, elem in enumerate(self.all_max_preds)
            if elem != self.all_labels[index]
        ]

    def predict(self, top_k_preds: int = 3, folds: int = 1) -> None:
        """
        loop over folds and find incorrect predictions across all possible validation
        datasets if k-folds used (i.e., not just 20% at once)

        Args:
            top_k_preds (int):  the top k predictions will be displayed in bar chart
                                must be < number of classes
            folds (int): number of k-folds used in resampling procedure
                         if kfold cross validation was not used, keep as 1
        Returns:
            ndarray: converted image to gray scale
        """

        for fold in range(folds):
            for ((imgs, labels, paths), _) in self.load_val_loader(fold):
                b = BatchPredictions(imgs, self.load_model(fold))
                b.find_max_preds()
                b.top_k_preds(top_k_preds)
                self.all_topk_probs.append(b.probs)
                self.all_topk_classes.append(b.classes)
                self.all_max_preds.append(b.max_preds)
                self.all_labels.append(labels)
                self.all_paths.append(paths)

    def hone_incorrect_predictions(
        self, label_list: dict[str, int], human_label: int, model_label: int
    ):
        """
        find indices where human labeled as one thing and model labeled as another

        Args:
            label_list (dict[str, int]): dictionary of class labels mapped to integers
            human_label (int): category of the human label
            model_label (int): category of the model label
        """
        idx_human = np.where(self.all_labels == human_label)

        [
            self.wrong_trunc.append(i)
            for i in idx_human[0]
            if self.all_max_preds[i] == model_label
        ]
        cat1 = list(label_list.keys())[list(label_list.values()).index(human_label)]
        cat2 = list(label_list.keys())[list(label_list.values()).index(model_label)]
        print(f"{len(self.wrong_trunc)} wrong predictions between {cat1} and {cat2}")
