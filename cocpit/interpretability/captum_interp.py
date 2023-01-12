from cocpit.interpretability.plot import Plot
from cocpit import config as config
import torch
import os

import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from cocpit.interpretability.misc_funcs import preprocess_image
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    LayerGradCam,
    LayerAttribution,
    NoiseTunnel,
    Occlusion
)
from captum.attr import visualization as viz


class Captum:
    def __init__(self, model, img):
        self.model: torch.nn.parallel.data_parallel.DataParallel = model
        self.input = img
        self.transformed_img = None
        self.attributions_ig = None
        self.integrated_gradients = None

    def transform_input(self, transform, transform_normalize):
        self.transformed_img = transform(self.input)
        self.input_norm = transform_normalize(self.transformed_img)
        # unsqueeze returns a new tensor with a dimension of size one inserted at the #specified position.
        self.input_norm = self.input_norm.unsqueeze(0)

    def get_target_class(self):
        output = self.model(self.input_norm)
        ## applied softmax() function
        output = F.softmax(output, dim=1)
        # torch.topk returns the k largest elements of the given input tensor along a given #dimension.K here is 1
        _, self.pred_label_idx = torch.topk(output, 1)
        self.pred_label_idx = self.pred_label_idx.squeeze_()

    def integrated_grads(self):
        self.integrated_gradients = IntegratedGradients(self.model)
        # Request the algorithm to assign our output target to
        self.attributions_ig = self.integrated_gradients.attribute(
            self.input_norm, target=self.pred_label_idx, n_steps=200
        )

    def grad_heatmap(self):
        return viz.visualize_image_attr(
            np.transpose(
                self.attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)
            ),
            original_image=np.transpose(
                self.transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)
            ),
            method="blended_heat_map",
            cmap="Blues",
            show_colorbar=False,
            sign="positive",
            outlier_perc=1,
        )

    def noise_tunnel_smoothing(self):
        noise_tunnel = NoiseTunnel(self.integrated_gradients)

        attributions_ig_nt = noise_tunnel.attribute(
            self.input_norm,
            nt_samples=10,
            nt_type="smoothgrad_sq",
            target=self.pred_label_idx,
        )
        _ = viz.visualize_image_attr_multiple(
            np.transpose(
                attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0)
            ),
            np.transpose(
                self.transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)
            ),
            ["heat_map"],
            ["positive"],
            cmap="Blues",
            show_colorbar=False,
        )

    def gradient_shap(self):
        gradient_shap = GradientShap(self.model)

        # Definition of baseline distribution of images
        rand_img_dist = torch.cat([self.input_norm * 0, self.input_norm * 1])

        attributions_gs = gradient_shap.attribute(
            self.input_norm,
            n_samples=50,
            stdevs=0.0001,
            baselines=rand_img_dist,
            target=self.pred_label_idx,
        )
        _ = viz.visualize_image_attr_multiple(
            np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(
                self.transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)
            ),
            ["heat_map"],
            ["absolute_value"],
            cmap="Blues",
            show_colorbar=False,
        )

    def layer_gradcam(self):
        layer_gradcam = LayerGradCam(
            self.model, list(self.model.module.features._modules.items())[20][1]
        )
        attributions_lgc = layer_gradcam.attribute(
            self.input_norm, target=self.pred_label_idx
        )

        _ = viz.visualize_image_attr(
            attributions_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
            sign="all",
        )

    def layer_attributions(self):

        layer_gradcam = LayerGradCam(
            self.model, list(self.model.module.features._modules.items())[20][1]
        )
        attributions_lgc = layer_gradcam.attribute(
            self.input_norm, target=self.pred_label_idx
        )

        upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, self.input_norm.shape[2:])

        _ = viz.visualize_image_attr_multiple(upsamp_attr_lgc[0].cpu().permute(1,2,0).detach().numpy(),
            self.transformed_img.permute(1,2,0).numpy(),
            ["blended_heat_map","masked_image"],
            ["all","positive"],
            show_colorbar=True,
            #titles=["Attribution", "Masked"],
        )
    
    def occlusion(self):
        occlusion = Occlusion(self.model)

        attributions_occ = occlusion.attribute(self.input_norm,
                                            target=self.pred_label_idx,
                                            strides=(3, 8, 8),
                                            sliding_window_shapes=(3,15, 15),
                                            baselines=0)


        _ = viz.visualize_image_attr_multiple(np.transpose(attributions_occ.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            np.transpose(self.transformed_img.squeeze().cpu().detach().numpy(), (1,2,0)),
                                            ["heat_map", "heat_map", "masked_image"],
                                            ["positive", "negative", "positive"],
                                            show_colorbar=False,
                                            # titles=["Attribution", "Negative Attribution", "Masked"],
                                            )