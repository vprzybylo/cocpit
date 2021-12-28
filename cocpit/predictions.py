'''holds methods regarding making model predictions
for confusion matrices and running the model on new data'''

import itertools
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import cocpit.config as config
from cocpit.auto_str import auto_str


@auto_str
class Predict:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

        self.model.to(config.DEVICE)
        self.model.eval()

    def predictions_for_confmatrix(self):
        """
        get a list of hand labels and predictions from a saved dataloader/model

        Returns
        -------
        - all_preds (list): predictions from a model
        - all_labels (list): correct/hand labels
        """

        all_preds = []
        all_labels = []
        for ((imgs, labels, img_paths), index) in self.dataloader:
            with torch.no_grad():
                # get the inputs
                imgs = imgs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                output = self.model(imgs)
                pred = torch.argmax(output, 1)

                all_preds.append(pred.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.asarray(list(itertools.chain(*all_preds)))
        all_labels = np.asarray(list(itertools.chain(*all_labels)))

        return all_preds, all_labels

    def batch_probability(self, imgs):
        '''make predictions on a batch of images and convert to probabilities'''

        logits = self.model.forward(imgs)
        ps = F.softmax(logits, dim=1)
        # put predictions back on CPU and turn into % between 0-100
        return ps.cpu().numpy() * 100  # dimension of (batch size, # classes)

    def all_predictions(self):
        """Predict the classes of images from a test_loader
        in batches using a trained CNN.  No labels associated.
        """

        d = defaultdict(list)
        top_class = []
        for batch_idx, (imgs, img_paths) in enumerate(self.dataloader):
            with torch.no_grad():
                imgs = imgs.to(config.DEVICE)

                batch_output = self.batch_probability(imgs)
                for pred in batch_output:
                    for c in range(len(config.CLASS_NAMES)):  # class
                        d[config.CLASS_NAMES[c]].append(pred[c])
                    top_class.append(config.CLASS_NAMES[np.argmax(pred)])

        return d, top_class
