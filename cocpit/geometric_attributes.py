"""
calculates particle geometric properties
"""

import pandas as pd

import cocpit.pic as pic


def get_attributes(open_dir, filename, desired_size):

    image = cocpit.pic.Image(open_dir, filename)
    image.resize_stretch(desired_size)
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
            if count_edge_px > 0:
                std = np.std(np.nonzero(image.edges()))
            else:
                std = 0
            lapl = image.laplacian()
            contours = len(image.contours)
            edges = count_edge_px
            contrast = image.contrast()
            height = image.height_og
            width = image.width_og
            cnt_area = image.area
            solidity = image.solidity()
            complexity = image.complexity()
            equiv_d = image.equiv_d()
            convex_perim = image.convex_perim(True)
            hull_area = image.hull_area
            perim = image.perim
            phi = image.phi()
            circularity = image.circularity()
            cutoff = image.cutoff_perim()
            perim_area_ratio = image.perim_area_ratio()
            roundness = image.roundness()
            filled_circular_area_ratio = image.filled_circular_area_ratio()
            extreme_points = image.extreme_points()
        else:
            lapl = -999
            contours = -999
            edges = -999
            contrast = -999
            height = -999
            width = -999
            cnt_area = -999
            solidity = -999
            complexity = -999
            equiv_d = -999
            convex_perim = -999
            hull_area = -999
            perim = -999
            phi = -999
            circularity = -999
            cutoff = -999
            perim_area_ratio = -999
            roundness = -999
            filled_circular_area_ratio = -999
            extreme_points = -999
            std = -999

    dicts = {}

    keys = [
        'filename',
        'height',
        'width',
        'lapl',
        'contours',
        'edges',
        'std',
        'cnt_area',
        'contrast',
        'circularity',
        'solidity',
        'complexity',
        'equiv_d',
        'convex_perim',
        'hull_area',
        'perim',
        'phi',
        'cutoff',
        'extreme_points',
        'filled_circular_area_ratio',
        'roundness',
        'perim_area_ratio',
    ]
    values = [
        filenames,
        height,
        width,
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
        cutoff,
        extreme_points,
        filled_circular_area_ratio,
        roundness,
        perim_area_ratio,
    ]
    for key, val in zip(keys, values):
        dicts[key] = val
    df = pd.DataFrame(dicts, index=[0])

    return df


def make_dataframe(open_dir, desired_size, good=True, train=False):
    """


    returns
    -------
        df (pd.DataFrame): dataframe with image attributes from cocpit.pic module
    """

    files = os.listdir(open_dir)
    num_cores = multiprocessing.cpu_count()
    start = time.time()
    dfs = Parallel(n_jobs=num_cores - 2)(
        delayed(get_attributes)(open_dir, filename, desired_size, good, train)
        for filename in files
    )

    # Concat dataframes to one dataframe
    df = pd.concat(dfs, ignore_index=True)
    end = time.time()
    print('Completed in: %.2f sec' % (end - start))

    return df


def main(df, open_dir, num_workers):

    # file_list = df["filename"]

    for open_dir in open_dirs_spheres:
        if open_dir == open_dirs_spheres[0]:
            print('training good spheres')
            dfs.append(make_dataframe(open_dir, desired_size, good=True, train=True))
        else:
            print('training bad spheres')
            dfs.append(make_dataframe(open_dir, desired_size, good=False, train=True))
    df_spheres = pd.concat([dfs[0], dfs[1]])
    df_spheres.to_pickle(save_df_spheres)
