"""plots saliency maps of images to determine
which pixels most contriute to the final output"""

import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import cocpit.config as config
import numpy as np
import torch.nn.functional as F


plt_params = {
    "axes.labelsize": "large",
    "axes.titlesize": "large",
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


def plot_image(ax, row, image, class_, file):
    if row == 0:
        ax[row, 1].set_title("Saliency Map")

    ax[row, 0].imshow(image, aspect="auto")
    ax[row, 0].set_title(
        f"Label: {class_} \n Station: {file.split('_')[1].split('.')[0]}"
    )
    ax[row, 0].axes.xaxis.set_ticks([])
    ax[row, 0].axes.yaxis.set_ticks([])


def plot_saliency(image, ax, row, model):
    saliency, pred, probs = get_saliency(image, model)

    # code to plot the saliency map as a heatmap
    ax[row, 1].imshow(saliency[0], cmap=plt.cm.hot, aspect="auto")
    ax[row, 1].set_title(
        f"Prediction: {config.CLASS_NAMES[pred]} \n Probability: {np.round(probs.detach().numpy().max()*100, 2)}%"
    )
    ax[row, 1].axes.xaxis.set_ticks([])
    ax[row, 1].axes.yaxis.set_ticks([])


def saliency_runner(model, indices=11, savefig=True, size=224):
    """The saliency map will show the strength
    for each pixel contribution to the final output

    Args:
        model: loaded pytorch model
        indices (int): number of class iterations to plot (1 index= 1 iteration of all classes).
        savefig (bool): whether or not to save the figure
        size (int): the size to transform the original image
    """

    for index in range(indices):
        fig, ax = plt.subplots(len(config.CLASS_NAMES), 2, figsize=(10, 12))
        for row, class_ in enumerate(config.CLASS_NAMES):
            open_dir = f"{config.DATA_DIR}{config.CLASS_NAME_MAP[class_]}/"
            file = os.listdir(open_dir)[index]
            image = Image.open(open_dir + file).resize((size, size)).convert("RGB")
            plot_image(ax, row, image, class_, file)
            image = preprocess(image, size)
            plot_saliency(image, ax, row, model)

        if savefig:
            fig.savefig(f"{config.BASE_DIR}/plots/saliency_maps.png")


def preprocess(image, size):
    """Preprocess the image, convert to tensor, normalize,
    and convert to correct shape"""

    transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image).unsqueeze(0)
    tensor.requires_grad = True
    return tensor


def get_saliency(image, model):

    model.eval()

    """find the gradient with respect to
    the input image; call requires_grad_ on it"""
    image.requires_grad_()

    """
    forward pass through the model to get the scores
    note that VGG model doesn't perform softmax at the end
    we also don't need softmax, just need scores
    """
    scores = model(image)
    # turn scores into probabilty
    probs = F.softmax(scores, dim=1).cpu()

    """Get the index corresponding to the
    maximum score and the maximum score itself."""
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]

    """
    backward function on score_max performs the backward
    pass in the computation graph and calculates the gradient of
    score_max with respect to nodes in the computation graph
    """
    score_max.backward()

    """
    Saliency would be the gradient with respect to the input image now.
    But note that the input image has 3 channels,
    R, G and B. To derive a single class saliency
    value for each pixel (i, j), we take the maximum magnitude
    across all colour channels.
    """

    saliency, _ = torch.max(image.grad.data.abs(), dim=1)
    return saliency, score_max_index, probs
