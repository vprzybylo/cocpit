"""holds methods regarding making model predictions
for confusion matrices and running the model on new data"""

import itertools
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import cocpit.config as config
from cocpit.auto_str import auto_str


@auto_str
class LoaderPredictions:
    """setup methods for predictions across batches"""

    def __init__(self):
        self.all_labels = []
        self.all_paths = []
        self.all_topk_probs = []
        self.all_topk_classes = []
        self.all_max_preds = []
        self.wrong_trunc = []

    def concat(self, var):
        return np.asarray(list(itertools.chain(*var)))

    def concatenate_loader_vars(
        self,
    ):
        """
        flatten arrays from appending in batches
        """
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

    def hone_incorrect_predictions(self, label_list, human_label, model_label):
        """
        find indices where human labeled as one thing and model labeled as another
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

    def find_wrong_indices(self):
        """ """
        self.wrong_idx = [
            index
            for index, elem in enumerate(self.all_max_preds)
            if elem != self.all_labels[index]
        ]  # and
        # labels[index]==actual and elem == model_label]


@auto_str
class TestBatchPredictions(LoaderPredictions):
    """finds predictions for a given batch of test data (no label)"""

    def __init__(
        self,
        imgs,
        paths,
        model,
        labels=None,
    ):
        super().__init__()
        self.imgs = imgs.to(config.DEVICE)
        self.paths = paths
        self.model = model.to(config.DEVICE)
        self.labels = labels
        self.max_preds = None
        self.probs = None
        self.classes = None

    def find_logits(self):
        return self.model(self.imgs)

    def find_max_preds(self):
        """find the class with the highest probability
        across all images in the batch"""
        _, max_preds = torch.max(self.find_logits(), dim=1)
        # convert back to lists from being on gpus
        self.max_preds = max_preds.cpu().tolist()

    def preds_softmax(self):
        return F.softmax(self.find_logits(), dim=1)

    def top_k_preds(self, top_k_preds=9):
        '''get top k predictions for each index in the batch for bar chart'''
        topk = self.preds_softmax().cpu().topk(top_k_preds)
        self.probs, self.classes = [e.data.numpy().squeeze().tolist() for e in topk]

    def append_preds(self):
        """create a list of paths, top k predicted probabilities/classes and
        the class with the highest probability across batches.
        Stored in LoaderPredictions."""
        self.all_paths.append(self.paths)
        self.all_topk_probs.append(self.probs)
        self.all_topk_classes.append(self.classes)
        self.all_max_preds.append(self.max_preds)
        self.all_labels.append(self.labels)


# @auto_str
# class Predict:
#     def __init__(self, model, dataloader):
#         self.model = model
#         self.dataloader = dataloader

#         self.model.to(config.DEVICE)
#         self.model.eval()

#     def predictions_for_confmatrix(self):
#         """
#         get a list of hand labels and predictions from a saved dataloader/model

#         Returns
#         -------
#         - all_preds (list): predictions from a model
#         - all_labels (list): correct/hand labels
#         """

#         all_preds = []
#         all_labels = []
#         for ((imgs, labels, img_paths), index) in self.dataloader:
# #             with torch.no_grad():
# #                 # get the inputs
# #                 imgs = imgs.to(config.DEVICE)
# #                 labels = labels.to(config.DEVICE)

# #                 output = self.model(imgs)
# #                 # class with highest probability
# #                 pred = torch.argmax(output, 1)

# #                 all_preds.append(pred.cpu().numpy())
# #                 all_labels.append(labels.cpu().numpy())

#         all_preds = np.asarray(list(itertools.chain(*all_preds)))
#         all_labels = np.asarray(list(itertools.chain(*all_labels)))

#         return all_preds, all_labels
