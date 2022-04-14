"""
grid_shader:
    - add alternating background color to matplotlib figure to distinguish classification report groups


plot_metrics:
    - calculation and plotting functions for reporting performance metrics
    - confusion matrices

plot_timing:
    - plot the efficiency of the model
    - how long it takes to make predictions from .csv during training

plot:
    - plotting functions used in publication for model accuracy

saliency:
    - plots saliency maps of images to determine which pixels most contriute to the final output
    - called in notebooks/saliency_maps.py
"""
import glob
from os.path import basename, dirname, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
from . import *  # noqa: F403 E402
