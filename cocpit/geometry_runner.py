import cocpit.config as config

import cocpit.geometry as geometry  # isort: split
import multiprocessing
import time
from functools import partial
from typing import List

import pandas as pd

from cocpit import image as image


def keys() -> List[str]:
    """Particle attribute names

    Returns:
        List[str]: attribute names
    """
    return [
        "perim [pixels]",
        "hull_area [pixels]",
        "convex_perim [pixels]",
        "blur",
        "contours [#]",
        "contrast",
        "cnt_area [pixels]",
        "circularity",
        "solidity",
        "complexity",
        "equiv_d",
        "phi",
        "extreme_points",
        "filled_circular_area_ratio",
        "roundness",
        "perim_area_ratio",
    ]


def properties_null() -> List[int]:
    """
    Set particle attributes to a null value
    if the image area = 0

    Returns:
        List[int]: list of -999 with a length of attributes
    """
    return [-999 for _ in keys()]


def properties(img: geometry.Geometry) -> List[float]:
    """
    Calculated properties

    Args:
        image (geometry.Image): Loaded PIL image
    Returns:
        List[float]: list of calculated particle properties
    """
    img.morph_contours()
    # img.mask_background()

    return [
        img.perim,
        img.hull_area,
        img.convex_perim,
        img.laplacian,
        len(img.contours),
        img.im.std(),
        img.area,
        img.circularity,
        img.solidity,
        img.complexity,
        img.equiv_d,
        img.phi,
        img.extreme_points,
        img.filled_circular_area_ratio,
        img.roundness,
        img.perim_area_ratio,
    ]


def get_attributes(filename: str, open_dir: str) -> pd.DataFrame:
    """
    Create df of particle geometric properties

    Args:
        filename (str): filename of the image to load
        open_dir (str): directory to open the image in
    """
    img = geometry.Geometry(open_dir, filename)

    if len(img.contours) != 0:
        values = properties(img) if img.area != 0.0 else properties_null()
        return pd.DataFrame(dict(zip(keys(), values)), index=[0])


def main(df: pd.DataFrame, open_dir: str) -> pd.DataFrame:
    """
    - Reads in dataframe for a campaign after ice classification
    - Calculates particle geometric properties

    Args:
        df (pandas.DataFrame): dataframe with image filenames
        open_dir (str): directory to the images
    Returns:
        df (pd.DataFrame): dataframe with image attributes appended
    """

    files = df["filename"]
    start = time.time()

    with multiprocessing.Pool(1) as p:
        properties = p.map(partial(get_attributes, open_dir=open_dir), files)
    p.close()

    #     properties = Parallel(n_jobs=num_cpus)(
    #         delayed(get_attributes)(open_dir, filename) for filename in files
    #     )

    # append new properties dictionary to existing dataframe
    properties = pd.concat(properties, ignore_index=True)
    df = pd.concat([df, properties], axis=1).round(3)

    end = time.time()
    print("Geometric attributes added in: %.2f sec" % (end - start))

    return df
