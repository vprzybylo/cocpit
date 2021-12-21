"""plots saliency maps of images to determine
which pixels most contriute to the final output"""

import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

import cocpit.config as config

plt_params = {
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


def which_ax(x):
    """for plotting on the correct ax"""
    y = 0
    y1 = 1
    if x >= 3 and x < 6:
        x -= 3
        y = 2
        y1 = 3
    if x >= 6 and x < 9:
        x -= 6
        y = 4
        y1 = 5
    if x >= 9:
        x -= 9
    return x, y, y1


def plot_on_axis(image, ax, x, class_, model):
    '''plot saliency map on axis'''

    x, y, y1 = which_ax(x)

    ax[y, x].imshow(image)
    ax[y, x].set_title(class_[1])
    ax[y, 0].set_ylabel("Original Image")
    ax[y, x].axes.xaxis.set_ticks([])
    ax[y, x].axes.yaxis.set_ticks([])

    image = preprocess(image)
    saliency = get_saliency(image, model)

    # code to plot the saliency map as a heatmap
    ax[y1, x].imshow(saliency[0], cmap=plt.cm.hot)

    ax[y1, 0].set_ylabel("Saliency Map")
    ax[y1, x].axes.xaxis.set_ticks([])
    ax[y1, x].axes.yaxis.set_ticks([])


def plot_saliency(model, class_names, savefig=True):
    """The saliency map will show the strength
    for each pixel contribution to the final output"""
    fig, ax = plt.subplots(6, 3, figsize=(5, 13))
    for x, class_ in enumerate(class_names.items()):

        open_dir = (
            f"{config.BASE_DIR}/cpi_data/training_datasets/hand_labeled_resized_v1.3.0_no_blank/%s/"
            % class_[0]
        )

        file = os.listdir(open_dir)[21]
        image = Image.open(open_dir + file).convert("RGB")
        plot_on_axis(image, ax, x, class_, model)

    if savefig:
        fig.savefig(f"{config.BASE_DIR}/plots/saliency_maps.png", dpi=300)


def preprocess(image, size=224):
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
    note that VGG-19 model doesn't perform softmax at the end
    we also don't need softmax, just need scores
    """
    scores = model(image)

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
    return saliency
