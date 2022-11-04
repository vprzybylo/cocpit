# sourcery skip: no-relative-imports
"""
COCPIT package:

Classification of Cloud Particle Imagery and Thermodynamics
"""
import glob
from os.path import basename, dirname, isfile, join

from comet_ml import Experiment  # isort:split

MODULES = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3]
    for f in MODULES
    if isfile(f) and not f.endswith("__init__.py")
]
from . import *  # noqa: F403 E402
