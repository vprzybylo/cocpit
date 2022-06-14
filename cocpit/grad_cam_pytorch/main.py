import torch
import os
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from cocpit import config as config
import numpy as np
import matplotlib.cm as cm

from cocpit.grad_cam_pytorch.grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
)


class Process:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = None

    def show_raw_image(self):
        self.raw_image = cv2.imread(self.image_path)
        self.raw_image = cv2.resize(self.raw_image, (224,) * 2)
        fig, (ax) = plt.subplots(
            constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1
        )
        ax.imshow(self.raw_image)

    def transform(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image = transform(self.raw_image)
        # unsqueeze returns a new tensor with a dimension of size one inserted at the specified position.
        self.image = torch.unsqueeze(self.image, 0)


class Runner:
    def __init__(self, model, raw_image, image, topk, output_dir, target_layer):
        self.model = model
        self.raw_image = raw_image
        self.image = image
        self.topk = topk
        self.output_dir = output_dir
        self.target_layer = target_layer
        self.probs = None
        self.ids = None

    def deconv(self):
        """Deconvolution"""
        bp = BackPropagation(model=self.model)
        self.probs, self.ids = bp.forward(self.image)  # sorted

        deconv = Deconvnet(model=self.model)
        _ = deconv.forward(self.image)

        for i in range(self.topk):
            deconv.backward(ids=self.ids[:, [i]])
            gradients = deconv.generate()

            self.save_gradient(
                filename=os.path.join(
                    self.output_dir,
                    f"deconvnet-{config.CLASS_NAMES[self.ids.cpu().numpy()[0][i]]}.png",
                ),
                gradient=gradients,
            )
        deconv.remove_hook()

    def gradcam(self):
        """Grad-CAM"""
        gcam = GradCAM(model=self.model)
        _ = gcam.forward(self.image)

        gbp = GuidedBackPropagation(model=self.model)
        _ = gbp.forward(self.image)

        for i in range(self.topk):
            gbp.backward(ids=self.ids[:, [i]])
            gradients = gbp.generate()

            gcam.backward(ids=self.ids[:, [i]])
            regions = gcam.generate(target_layer=self.target_layer)

            self.save_gradient(
                filename=os.path.join(
                    self.output_dir,
                    f"guided-{config.CLASS_NAMES[self.ids.cpu().numpy()[0][i]]}.png",
                ),
                gradient=gradients,
            )

            self.save_gradcam(
                filename=os.path.join(
                    self.output_dir,
                    f"gradcam-{self.target_layer}-{config.CLASS_NAMES[self.ids.cpu().numpy()[0][i]]}.png",
                ),
                gcam=regions,
                raw_image=self.raw_image,
            )

            # Guided Grad-CAM
            self.save_gradient(
                filename=os.path.join(
                    self.output_dir,
                    f"guided_gradcam-{self.target_layer}-{config.CLASS_NAMES[self.ids[0][i].cpu().numpy()]}.png",
                ),
                gradient=torch.mul(regions.cpu(), gradients),
            )

    def vanilla_bp(self):
        """Vanilla Backpropagation"""
        bp = BackPropagation(model=self.model)
        self.probs, self.ids = bp.forward(self.image)  # sorted

        for i in range(self.topk):
            bp.backward(ids=self.ids[:, [i]])
            gradients = bp.generate()

            self.save_gradient(
                filename=os.path.join(
                    self.output_dir,
                    f"vanilla-{config.CLASS_NAMES[self.ids[0][i].cpu().numpy()]}.png",
                ),
                gradient=gradients,
            )

        # Remove all the hook function in the "model"
        bp.remove_hook()

    def save_gradient(self, filename, gradient):
        gradient = gradient.cpu().numpy()[0].transpose(1, 2, 0)
        gradient -= gradient.min()
        gradient /= gradient.max()
        gradient *= 255.0
        cv2.imwrite(filename, np.uint8(gradient))
        fig, (ax) = plt.subplots(
            constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1
        )
        ax.imshow(np.uint8(gradient))

    def save_gradcam(self, filename, gcam, raw_image, paper_cmap=False):
        gcam = gcam.cpu().numpy()[0]

        cmap = cm.hot(gcam)[..., :3] * 255.0
        if paper_cmap:
            alpha = gcam[..., None]
            gcam = alpha * cmap + (1 - alpha) * raw_image
        else:
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        cv2.imwrite(filename, np.uint8(gcam[0]))
        fig, (ax) = plt.subplots(
            constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1
        )
        ax.imshow(np.uint8(gcam[0]))


def main(image_path, model, target_layer, topk=3, output_dir="./results"):
    """
    Visualize model responses given multiple image
    """
    p = Process(image_path)
    p.show_raw_image()
    p.transform()

    r = Runner(model, p.raw_image, p.image, topk, output_dir, target_layer)
    r.deconv()
    r.gradcam()
    r.vanilla_bp()
