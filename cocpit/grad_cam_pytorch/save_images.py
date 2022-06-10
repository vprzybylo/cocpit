import cv2
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy()[0].transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))
    fig, (ax) = plt.subplots(constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1)
    ax.imshow(np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()[0]

    cmap = cm.hot(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam[0]))
    fig, (ax) = plt.subplots(constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1)
    ax.imshow(np.uint8(gcam[0]))
