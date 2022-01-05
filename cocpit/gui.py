"""
Holds the class for ipywidget buttons to
label incorrect predictions from a dataloader.
The dataloader, model, and all class variables
are initialized in notebooks/move_wrong_predictions.ipynb
"""

import functools
import itertools
import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image

import cocpit.config as config
from cocpit.auto_str import auto_str


@auto_str
class LoaderPredictions(object):
    """finds all incorrect predictions from a model and dataloader (across batches)"""

    def __init__(self):
        self.all_labels = []
        self.all_paths = []
        self.all_topk_probs = []
        self.all_topk_classes = []
        self.all_max_preds = []
        self.model = None
        self.val_loader = None
        self.wrong_trunc = []

    def get_dataloader(self, fold):
        data = torch.load(
            f"{config.VAL_LOADER_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_val_loader20_bs{config.BATCH_SIZE}"
            f"_k{str(fold)}_vgg16.pt"
        )

        self.val_loader = torch.utils.data.DataLoader(
            data,
            batch_size=int(config.BATCH_SIZE[0]),
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    def get_model(self, fold):
        self.model = torch.load(
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_bs{config.BATCH_SIZE}"
            f"_k{str(fold)}_vgg16.pt"
        ).cuda()
        self.model.eval()

    def concat(self, var):
        return np.asarray(list(itertools.chain(*var)))

    def concatenate_loader_vars(
        self,
    ):
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
        idx_human = np.where(self.all_labels == human_label)

        # find indices where human labeled as one thing and model labeled as another
        # according to human_label and model_label above
        [
            self.wrong_trunc.append(i)
            for i in idx_human[0]
            if self.all_max_preds[i] == model_label
        ]
        cat1 = list(label_list.keys())[list(label_list.values()).index(human_label)]
        cat2 = list(label_list.keys())[list(label_list.values()).index(model_label)]
        print(f"{len(self.wrong_trunc)} wrong predictions between {cat1} and {cat2}")


@auto_str
class BatchPredictions(LoaderPredictions):
    """finds wrong predictions for a given batch"""

    def __init__(
        self,
        lp,  # instance of LoaderPrediction
        imgs,
        labels,
        paths,
    ):
        super().__init__()
        self.model = lp.model
        self.imgs = imgs
        self.labels = labels
        self.paths = paths
        self.wrong_idx = []
        self.logits = None
        self.max_preds = None
        self.probs = None
        self.classes = None

    def find_wrong_indices(self):
        self.imgs = self.imgs.to(config.DEVICE)
        # labels = labels.to(config.DEVICE)

        self.logits = self.model(self.imgs)
        # dimension 1 because taking the prediction
        # with the highest probability
        # from all classes across each index in the batch
        _, max_preds = torch.max(self.logits, dim=1)

        # convert back to lists from being on gpus
        self.max_preds = max_preds.cpu().tolist()
        # labels = labels.cpu().tolist()

        self.wrong_idx = [
            index
            for index, elem in enumerate(self.max_preds)
            if elem != self.labels[index]
        ]  # and
        # labels[index]==actual and elem == model_label]

    def top_k_preds(self, top_k_preds=9):
        # get top k predictions for each index in the batch for bar chart
        predictions = F.softmax(self.logits, dim=1)
        topk = predictions.cpu().topk(top_k_preds)  # top k predictions
        self.probs, self.classes = [e.data.numpy().squeeze().tolist() for e in topk]

    def append_preds(self, lp):

        # human label and image path
        lp.all_labels.append([self.labels[i] for i in self.wrong_idx])
        lp.all_paths.append([self.paths[i] for i in self.wrong_idx])

        # model top k predicted  probability and classes per image
        lp.all_topk_probs.append([self.probs[i] for i in self.wrong_idx])
        lp.all_topk_classes.append([self.classes[i] for i in self.wrong_idx])

        # top predicted class from model
        lp.all_max_preds.append([self.max_preds[i] for i in self.wrong_idx])


@auto_str
class GUI(LoaderPredictions):
    """
    creates buttons in notebooks/move_wrong_predictions.ipynb
    and notebooks/gui_check_dataset_one_class.ipynb
    to label predictions
    """

    def __init__(self, all_labels=None, all_paths=None, lp=None):
        super().__init__()

        self.index = 0
        self.count = 0  # number of moved images
        self.center = None
        self.menu = None
        self.forward = None
        if lp is not None:
            # variables at wrong indices based on user input of specific categories to sift through
            # used in gui_move_wrong_prediction.ipynb
            self.wrong = True
            self.all_labels = lp.all_labels[lp.wrong_trunc]
            self.all_paths = lp.all_paths[lp.wrong_trunc]
            self.all_topk_probs = lp.all_topk_probs[lp.wrong_trunc]
            self.all_topk_classes = lp.all_topk_classes[lp.wrong_trunc]

        else:
            # used in gui_check_dataset_one_class.ipynb
            # all labels and paths of training dataset
            self.wrong = False
            self.all_labels = all_labels
            self.all_paths = all_paths

    def make_buttons(self):
        """
        use ipywidgets to create a box for the image, bar chart, dropdown,
        and next button
        """
        self.label = self.all_labels[self.index]
        self.center = ipywidgets.Output()  # center image with predictions
        self.menu = ipywidgets.Dropdown(
            options=config.CLASS_NAMES,
            description="Category:",
            value=config.CLASS_NAMES[self.label],
        )
        self.bar_chart()

        self.menu.observe(self.on_change, names="value")

        # create button that progresses through incorrect predictions
        self.forward = Button(description="Next")
        self.forward.on_click(self.on_button_next)

    def on_change(self, change):
        """
        when a class in the dropdown is selected, move the image and save
        it to the specified class
        """
        self.save_image(change)

    def on_button_next(self, b):
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1
        """
        self.index = self.index + 1
        self.bar_chart()

        # keep the default dropdown value to agg
        # don't want it to change based on previous selection
        self.menu.value = config.CLASS_NAMES[self.label]

    def bar_chart(self):
        """
        use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """

        # update for new index
        self.label = self.all_labels[self.index]
        self.path = self.all_paths[self.index]
        # add chart to ipywidgets.Output()
        with self.center:
            if self.wrong:
                self.topk_probs = self.all_topk_probs[self.index]
                self.topk_classes = self.all_topk_classes[self.index]

                # puts class names in order based on probabilty of prediction
                crystal_names = [config.CLASS_NAMES[e] for e in self.topk_classes]
                self.view_classifications_wrong(crystal_names)
            else:
                self.view_classifications()

    def view_classifications_wrong(self, crystal_names):
        """
        bar chart code
        outputs top k predictions for a given image
        """
        clear_output()  # so that the next fig doesnt display below
        try:
            fig, (ax1, ax2) = plt.subplots(
                constrained_layout=True, figsize=(5, 7), ncols=1, nrows=2
            )
            image = Image.open(self.path)
            ax1.imshow(image)
            ax1.set_title(
                f"Human Labeled as: {config.CLASS_NAMES[self.label]}\n"
                f"Model Labeled as: {crystal_names[0]}"
            )
            ax1.axis("off")

            y_pos = np.arange(len(self.topk_probs))
            ax2.barh(y_pos, self.topk_probs, align="center")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(crystal_names)
            ax2.tick_params(axis="y", rotation=45)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_title("Class Probability")
            plt.show()

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
            pass

    def view_classifications(self):
        """
        show image
        """
        clear_output()  # so that the next fig doesnt display below
        try:
            fig, ax1 = plt.subplots(
                constrained_layout=True, figsize=(5, 7), ncols=1, nrows=1
            )
            image = Image.open(self.path)
            ax1.imshow(image)
            ax1.set_title(f"Human Labeled as: {config.CLASS_NAMES[self.label]}\n")
            ax1.axis("off")
            plt.show()

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
            pass

    def save_image(self, change):
        """
        move the image based on dropdown selection
        """
        filename = self.path.split("/")[-1]

        data_dir = f"/data/data/cpi_data/training_datasets/hand_labeled_resized_{config.TAG}_sideplanes_copy/"
        # print(f"{data_dir}{config.CLASS_NAMES[self.label]}/{filename}")
        # print(f"{data_dir}{change.new}/{filename}")
        try:
            shutil.move(
                f"{data_dir}{config.CLASS_NAMES[self.label]}/{filename}",
                f"{data_dir}{change.new}/{filename}",
            )
            self.count += 1
            print(f"moved {self.count} images")

        except FileNotFoundError:
            print(self.path)
            print("File Not Found. Not moving.")
            pass
