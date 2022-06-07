from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    EigenGradCAM,
    LayerCAM,
    FullGrad,
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
    deprocess_image,
    preprocess_image,
)
import torch
from cocpit import config as config
import cv2
import numpy as np


class GUIGradCAM:
    def __init__(self, input_path, method):
        self.input_path = input_path
        self.method = method
        self.model = torch.load(config.MODEL_PATH)
        self.grayscale_cam = None
        self.gb = None
        self.cam_gb = None
        self.cam_image = None
        self.rgb_img = None
        self.methods = {
            "gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            "fullgrad": FullGrad,
        }

    def input_tensor(self):
        rgb_img = cv2.imread(self.input_path, 1)[:, :, ::-1]
        self.rgb_img = np.float32(rgb_img) / 255
        return preprocess_image(
            rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def cam_method(self):

        cam_algorithm = self.methods[self.method]
        with cam_algorithm(
            model=self.model,
            target_layers=[self.model.module.features[-1]],  # MaxPool2d
            use_cuda=True,
        ) as cam:

            # AblationCAM and ScoreCAM have batched implementations.
            # If targets is None, the highest scoring category (for every member in the batch) will be used.
            cam.batch_size = 1
            self.grayscale_cam = cam(
                input_tensor=self.input_tensor(),
                targets=None,
                aug_smooth=True,
                eigen_smooth=True,
            )

            # Here grayscale_cam has only one image in the batch
            self.grayscale_cam = self.grayscale_cam[0, :]

            cam_image = show_cam_on_image(
                self.rgb_img, self.grayscale_cam, use_rgb=True
            )

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            self.cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    def guided_backpropagation(self):
        """
        Guided Backpropagation combines vanilla backpropagation at ReLUs
        (leveraging which elements are positive in the preceding feature map)
        with DeconvNets (keeping only positive error signals).
        We are only interested in what image features the neuron detects.
        So when propagating the gradient, we set all the negative gradients to 0.
        We don’t care if a pixel “suppresses’’ (negative value) a neuron somewhere along the part to our neuron.
        Value in the filter map greater than zero signifies the pixel importance,
        which is overlapped with the input image to show which pixel from the input image contributed the most.
        """

        gb_model = GuidedBackpropReLUModel(model=self.model, use_cuda=True)
        gb = gb_model(self.input_tensor(), target_category=None)

        # merge takes single channel images and combines them to make a multi-channel image.
        cam_mask = cv2.merge(
            [self.grayscale_cam, self.grayscale_cam, self.grayscale_cam]
        )
        self.cam_gb = deprocess_image(cam_mask * gb)
        self.gb = deprocess_image(gb)

    def write_cam_images(self):
        cv2.imwrite(f"{self.method}_cam.jpg", self.cam_image)
        cv2.imwrite(f"{self.method}_gb.jpg", self.gb)
        cv2.imwrite(f"{self.method}_cam_gb.jpg", self.cam_gb)

    def plot_grad_cam(self, ax4, ax5, ax6):

        ax4.imshow(self.cam_image, aspect="auto")
        ax4.set_title("GRAD-CAM")
        ax4.axes.xaxis.set_ticks([])
        ax4.axes.yaxis.set_ticks([])

        ax5.imshow(self.gb, aspect="auto")
        ax5.set_title("Back-Propagation")
        ax5.axes.xaxis.set_ticks([])
        ax5.axes.yaxis.set_ticks([])

        ax6.imshow(self.cam_gb, aspect="auto")
        ax6.set_title("CAM-BP Combined")
        ax6.axes.xaxis.set_ticks([])
        ax6.axes.yaxis.set_ticks([])
