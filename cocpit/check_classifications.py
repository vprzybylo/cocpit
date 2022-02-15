"""
- check predictions from a saved CNN
- called in check_classifications.ipynb for bar chart plot
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from cocpit.auto_str import auto_str


@auto_str
class Classification:
    def __init__(self):
        """
        make prediction on single image
        """

    def open_image(self, path):
        """ """
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.process_image(img)

        # Convert 2D image to 1D vector
        img = np.expand_dims(img, 0)
        self.img = torch.from_numpy(img)

    def process_image(self, image):
        """
        apply transformation to image

        Args:
            image ([]): one image to transform

        Returns:
            [type]: transformed image
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return preprocess(image)

    def top_k_predictions(self, logits, topk=9):
        """
        top k predictions

        Args:
            topk (int, optional): top k predictions included in bar chart. Defaults to 9.

        Returns:
            [list]: top k probabilities in order of highest predicted class to lowest
        """
        ps = F.softmax(logits, dim=1)
        topk = ps.cpu().topk(topk)
        return (e.data.numpy().squeeze().tolist() for e in topk)

    def predict_probability(
        self,
        path,
        device,
        model,
    ):
        """
        :param path: [description]
        :type path: [type]
        :param device: [description]
        :type device: [type]
        :param model: [description]
        :type model: [type]
        :return: [description]
        :rtype: [type]
        """

        """
        Predict the class (or classes) of an image
        using a trained deep learning model.

        Args:
            path ([str]): path directory to the dataset.
            device ([type]): use cuda if available
            model ([type]): torch.nn.parallel.data_parallel.DataParallel loaded from saved file
        """

        model.eval()
        model.to(device)
        inputs = Variable(img).to(device)
        logits = model.forward(inputs)
        return logits

    def view_classification(self, im, prob, crystal_names, savefig=False):
        """
        Function for viewing an image and it's predicted classes.
        Horizontal bar chart with image above predictions
        """

        image = Image.open(im)
        fig, (ax1, ax2) = plt.subplots(figsize=(7, 10), ncols=1, nrows=2)

        ax1.set_title(crystal_names[0])
        ax1.imshow(image)
        ax1.axis("off")

        y_pos = np.arange(len(prob))
        ax2.barh(y_pos, prob, align="center")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(crystal_names)
        ax2.tick_params(axis="y", rotation=45)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title("Class Probability")
        plt.show()
