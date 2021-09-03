"""
calculates particle geometric properties
"""

import multiprocessing
import os
import time
from functools import partial

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import cocpit.pic as pic


def get_attributes(filename, open_dir):
    image = pic.Image(open_dir, filename)
    # image.resize_stretch(desired_size)
    image.find_contours()

    if len(image.contours) != 0:

        image.calculate_largest_contour()
        image.calculate_area()

        if image.area != 0.0:
            image.calculate_perim()
            image.calculate_hull_area()
            image.morph_contours()
            # image.mask_background()

            count_edge_px = np.count_nonzero(image.edges())
            std = np.std(np.nonzero(image.edges())) if count_edge_px > 0 else 0
            lapl = image.laplacian()
            contours = len(image.contours)
            edges = count_edge_px
            contrast = image.contrast()
            cnt_area = image.area
            solidity = image.solidity()
            complexity = image.complexity()
            equiv_d = image.equiv_d()
            convex_perim = image.convex_perim(True)
            hull_area = image.hull_area
            perim = image.perim
            phi = image.phi()
            circularity = image.circularity()
            perim_area_ratio = image.perim_area_ratio()
            roundness = image.roundness()
            filled_circular_area_ratio = image.filled_circular_area_ratio()
            extreme_points = image.extreme_points()
        else:
            lapl = -999
            contours = -999
            edges = -999
            contrast = -999
            cnt_area = -999
            solidity = -999
            complexity = -999
            equiv_d = -999
            convex_perim = -999
            hull_area = -999
            perim = -999
            phi = -999
            circularity = -999
            perim_area_ratio = -999
            roundness = -999
            filled_circular_area_ratio = -999
            extreme_points = -999
            std = -999

        keys = [
            "blur",
            "contours",
            "edges",
            "std",
            "cnt_area",
            "contrast",
            "circularity",
            "solidity",
            "complexity",
            "equiv_d",
            "convex_perim",
            "hull_area",
            "perim",
            "phi",
            "extreme_points",
            "filled_circular_area_ratio",
            "roundness",
            "perim_area_ratio",
        ]
        values = [
            lapl,
            contours,
            edges,
            std,
            cnt_area,
            contrast,
            circularity,
            solidity,
            complexity,
            equiv_d,
            convex_perim,
            hull_area,
            perim,
            phi,
            extreme_points,
            filled_circular_area_ratio,
            roundness,
            perim_area_ratio,
        ]
        properties = {key: val for key, val in zip(keys, values)}
        # turn dictionary into dataframe
        properties = pd.DataFrame(properties, index=[0])

        return properties


def main(df, open_dir, num_cpus):
    """
    reads in dataframe for a campaign after ice classification and
    calculates particle geometric properties using the cocpit.pic module

    returns
    -------
        df (pd.DataFrame): dataframe with image attributes appended
    """

    files = df['filename']
    start = time.time()

    p = multiprocessing.Pool(num_cpus)
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
