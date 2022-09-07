import cv2
import numpy as np
import torch
from cocpit import config as config
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam import GradCAM, FullGrad
from torchvision.transforms import Compose, Normalize, ToTensor

from cocpit.interpretability.misc_funcs import (
    normalize,
)


def deprocess_image(img):
    """see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65"""
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    use_rgb: bool = False,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
) -> np.ndarray:
    """This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}"
        )

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def preprocess_image(
    img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
) -> torch.Tensor:
    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])
    return preprocessing(img.copy()).unsqueeze(0)


def cam_all_layers(image_path, input_tensor, model, method="gradcam"):

    methods = {
        "gradcam": GradCAM,
        "fullgrad": FullGrad,
    }

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img)
    target_layers = [model.module.features[42]]

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[method]

    with cam_algorithm(model=model, target_layers=target_layers, use_cuda=False) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets,
            aug_smooth=True,
            eigen_smooth=True,
        )

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)
    return cam_gb, gb
