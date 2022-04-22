import operator
import numpy as np
import torch
from cocpit.performance_metrics import Metrics
from cocpit import config as config


class Train(Metrics):
    def label_counts(self, label_cnts, labels):
        """
        Calculate the # of labels per batch to ensure
        weighted random sampler is correct
        """
        for n, _ in enumerate(config.CLASS_NAMES):
            label_cnts[n] += len(np.where(labels.numpy() == n)[0])
        print("LABEL COUNT = ", label_cnts)
        return label_cnts

    def print_label_count(self, label_cnts_total, labels) -> None:
        """print cumulative sum of images per class, per batch to
        ensure weighted sampler is working properly"""
        label_cnts = self.label_counts(label_cnts_total, labels)
        label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))

    def forward(self) -> None:
        """perform forward operator"""
        with torch.set_grad_enabled(True):
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.loss.backward()  # compute updates for each parameter
            self.optimizer.step()  # make the updates for each parameter

    def print_batch_metrics(self) -> None:
        """
        outputs batch iteration, loss, and accuracy
        """
        print(
            f"Training, Batch {self.batch + 1}/{len(self.dataloaders['train'])},\
            Loss: {self.loss.item():.3f}, Accuracy: {self.batch_acc:.3f}"
        )

    def iterate_batches(self, print_label_count: bool = False) -> None:
        """iterate over a batch in a dataloader and train"""

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, _), _) in enumerate(
            self.dataloaders["train"]
        ):
            if print_label_count:
                self.print_label_count(label_cnts_total, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.forward()
            self.batch_metrics()
            self.print_batch_metrics()
