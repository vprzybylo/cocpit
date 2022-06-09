import cv2
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    # cv2.imwrite(filename, np.uint8(gradient))
    fig, (ax) = plt.subplots(constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1)
    ax.imshow(np.uint8(gradient))


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    # cv2.imwrite(filename, np.uint8(gcam))
    fig, (ax) = plt.subplots(constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1)
    ax.imshow(np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(filename, maps)
    fig, (ax) = plt.subplots(constrained_layout=True, figsize=(5, 5), ncols=1, nrows=1)
    ax.imshow(np.uint8(maps))
