"""merge environmental data with particle predictions and properties from ML model based on filename"""
import pandas as pd


def merge_env(df, df_env, campaign):
    """merge"""

    print(df['filename'], df_env['filename'])
    df = df.merge(df_env, on="filename")
    print(df)
    df.to_parquet(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet"
    )
    df.to_csv(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/merged_env/{campaign}.csv"
    )
    


def read_campaign(campaign):
    """read df of particle properties from ML model based on campaign"""
    columns = [
        "filename",
        "date",
        "Frame Width",
        "Frame Height",
        "Particle Width",
        "Particle Height",
        "Cutoff",
        "Aggregate",
        "Budding",
        "Bullet Rosette",
        "Column",
        "Compact Irregular",
        "Fragment",
        "Planar Polycrystal",
        "Rimed",
        "Sphere",
        "Classification",
        "Blur",
        "Contours",
        "Edges",
        "Std",
        "Contour Area",
        "Contrast",
        "Circularity",
        "Solidity",
        "Complexity",
        "Equivalent Diameter",
        "Convex Perimeter",
        "Hull Area",
        "Perimeter",
        "Aspect Ratio",
        "Extreme Points",
        "Area Ratio",
        "Roundness",
        "Perimeter-Area Ratio",
    ]
    dtype = "float16"
    dtypes = {
        "filename": "string",
        "Frame Width": dtype,
        "Frame Height": dtype,
        "Particle Width": dtype,
        "Particle Height": dtype,
        "Cutoff": dtype,
        "Aggregate": dtype,
        "Budding": dtype,
        "Bullet Rosette": dtype,
        "Column": dtype,
        "Compact Irregular": dtype,
        "Fragment": dtype,
        "Planar Polycrystal": dtype,
        "Rimed": dtype,
        "Sphere": dtype,
        "Blur": dtype,
        "Contours": dtype,
        "Edges": dtype,
        "Std": dtype,
        "Contour Area": dtype,
        "Contrast": dtype,
        "Circularity": dtype,
        "Solidity": dtype,
        "Complexity": dtype,
        "Equivalent Diameter": dtype,
        "Convex Perimeter": dtype,
        "Hull Area": dtype,
        "Perimeter": dtype,
        "Aspect Ratio": dtype,
        "Extreme Points": dtype,
        "Area Ratio": dtype,
        "Roundness": dtype,
        "Perimeter-Area Ratio": dtype,
    }

    return pd.read_csv(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=columns,
        low_memory=False,
        skiprows=1,
        dtype=dtypes,
    )


def read_env(campaign):
    """read environmental data from Carl
    - Drop date column due to it already being in particle property df
    - Carl's is truncated down to day not msec
    """
    columns = [
        "filename",
        "Date",
        "Latitude",
        "Longitude",
        "Altitude",
        "Pressure",
        "Temperature",
        "Ice Water Content",
        "Particle Size Distribution",
        "concentration ratio",
        "area ratio",
        "mass ratio",
    ]
    dtype = "float16"
    dtypes = {
        "filename": "string",
        "Latitude": dtype,
        "Longitude": dtype,
        "Altitude": dtype,
        "Pressure": dtype,
        "Temperature": dtype,
        "Ice Water Content": dtype,
    }

    return pd.read_csv(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/environment/{campaign}_atmospheric_V05.csv",
        names=columns,
        dtype=dtypes,
        skiprows=1,
    ).drop("Date", axis=1)


def main():
    df = read_campaign(campaign)
    df_env = read_env(campaign)
    merge_env(df, df_env, campaign)


if "__name__ == __main__":
    campaign = "MIDCIX"
    main()
