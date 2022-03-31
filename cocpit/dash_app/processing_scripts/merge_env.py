'''merge environmental data with particle predictions and properties from ML model based on filename'''
import pandas as pd


def merge_env(df, df_env, campaign):
    '''merge'''
    df = df.merge(df_env, on='filename')
    df.to_parquet(f'../../final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet')
    print(df)


def read_campaign(campaign):
    '''read df of particle properties from ML model based on campaign'''
    columns = [
        'filename',
        'date',
        'Frame Width',
        'Frame Height',
        'Particle Width',
        'Particle Height',
        'Cutoff',
        'Aggregate',
        'Budding',
        'Bullet Rosette',
        'Column',
        'Compact Irregular',
        'Fragment',
        'Planar Polycrystal',
        'Rimed',
        'Sphere',
        'Classification',
        'Blur',
        'Contours',
        'Edges',
        'Std',
        'Contour Area',
        'Contrast',
        'Circularity',
        'Solidity',
        'Complexity',
        'Equivalent Diameter',
        'Convex Perimeter',
        'Hull Area',
        'Perimeter',
        'Aspect Ratio',
        'Extreme Points',
        'Area Ratio',
        'Roundness',
        'Perimeter-Area Ratio',
    ]

    dtypes = {
        'Frame Width': 'float64',
        'Frame Height': 'float64',
        'Particle Width': 'float64',
        'Particle Height': 'float64',
        'Cutoff': 'float64',
        'Aggregate': 'float64',
        'Budding': 'float64',
        'Bullet Rosette': 'float64',
        'Column': 'float64',
        'Compact Irregular': 'float64',
        'Fragment': 'float64',
        'Planar Polycrystal': 'float64',
        'Rimed': 'float64',
        'Sphere': 'float64',
        'Blur': 'float64',
        'Contours': 'float64',
        'Edges': 'float64',
        'Std': 'float64',
        'Contour Area': 'float64',
        'Contrast': 'float64',
        'Circularity': 'float64',
        'Solidity': 'float64',
        'Complexity': 'float64',
        'Equivalent Diameter': 'float64',
        'Convex Perimeter': 'float64',
        'Hull Area': 'float64',
        'Perimeter': 'float64',
        'Aspect Ratio': 'float64',
        'Extreme Points': 'float64',
        'Area Ratio': 'float64',
        'Roundness': 'float64',
        'Perimeter-Area Ratio': 'float64',
    }
    return pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=columns,
        low_memory=False,
        skiprows=1,
        dtype=dtypes,
    )


def read_env(campaign):
    '''read environmental data from Carl
    - Drop date column due to it already being in particle property df
    - Carl's is truncated down to day not msec'''
    columns = [
        'filename',
        'Date',
        'Latitude',
        'Longitude',
        'Altitude',
        'Pressure',
        'Temperature',
        'Ice Water Content',
    ]

    dtypes = {
        'Latitude': 'float64',
        'Longitude': 'float64',
        'Altitude': 'float64',
        'Pressure': 'float64',
        'Temperature': 'float64',
        'Ice Water Content': 'float64',
    }

    return pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/environment/{campaign}.csv",
        names=columns,
        dtype=dtypes,
        skiprows=1,
    ).drop('Date', axis=1)


def main():
    df = read_campaign(campaign)
    df_env = read_env(campaign)
    merge_env(df, df_env, campaign)


if '__name__ == __main__':
    campaign = 'MPACE'
    main()
