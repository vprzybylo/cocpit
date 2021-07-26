"""
Check predictions from a saved CNN
called in check_classifications.ipynb for bar chart plot returns
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


def process_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = preprocess(image)
    return image


def predict_one_at_a_time(path, device, model, topk=9):
    '''
    Predict the class (or classes) of an image
    using a trained deep learning model.

    Params
    ------
    - path (str): path directory to the dataset.
    - device (obj): use cuda if available
    - model (obj): torch.nn.parallel.data_parallel.DataParallel loaded from saved file
    - num_workers (int): number of subprocesses to use when loading the dataset.

    Returns
    -------
    - top k probabilities in order of highest predicted class to lowest (list)
    """
    '''
    img = Image.open(path)
    img = img.convert("RGB")
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)

    model.eval()
    model.to(device)
    inputs = Variable(img).to(device)
    logits = model.forward(inputs)
    ps = F.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)  # top k predictions

    return (e.data.numpy().squeeze().tolist() for e in topk)


def view_classification(im, prob, crystal_names, savefig=False):
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
    plt.close()
    if savefig:
        fig.savefig(
            "/data/data/plots/" + current_time + ".png",
            bbox_inches="tight",
            pad_inches=0.3,
        )
