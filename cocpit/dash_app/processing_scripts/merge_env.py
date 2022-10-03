"""merge environmental data with particle predictions and properties from ML model based on filename"""
import pandas as pd


def merge_env(df, df_env, campaign):
    """merge"""
    df = df.merge(df_env, on="filename")
    df.to_parquet(
        f"../../final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet"
    )
    print(df)


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

    dtypes = {
        "Frame Width": "float16",
        "Frame Height": "float16",
        "Particle Width": "float16",
        "Particle Height": "float16",
        "Cutoff": "float16",
        "Aggregate": "float16",
        "Budding": "float16",
        "Bullet Rosette": "float16",
        "Column": "float16",
        "Compact Irregular": "float16",
        "Fragment": "float16",
        "Planar Polycrystal": "float16",
        "Rimed": "float16",
        "Sphere": "float16",
        "Blur": "float16",
        "Contours": "float16",
        "Edges": "float16",
        "Std": "float16",
        "Contour Area": "float16",
        "Contrast": "float16",
        "Circularity": "float16",
        "Solidity": "float16",
        "Complexity": "float16",
        "Equivalent Diameter": "float16",
        "Convex Perimeter": "float16",
        "Hull Area": "float16",
        "Perimeter": "float16",
        "Aspect Ratio": "float16",
        "Extreme Points": "float16",
        "Area Ratio": "float16",
        "Roundness": "float16",
        "Perimeter-Area Ratio": "float16",
    }
    return pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=columns,
        low_memory=False,
        skiprows=1,
        dtype=dtypes,
    )


def read_env(campaign):
    """read environmental data from Carl
    - Drop date column due to it already being in particle property df
    - Carl's is truncated down to day not msec"""
    columns = [
        "filename",
        "Date",
        "Latitude",
        "Longitude",
        "Altitude",
        "Pressure",
        "Temperature",
        "Ice Water Content",
    ]

    dtypes = {
        "Latitude": "float16",
        "Longitude": "float16",
        "Altitude": "float16",
        "Pressure": "float16",
        "Temperature": "float16",
        "Ice Water Content": "float16",
    }

    return pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/environment/{campaign}.csv",
        names=columns,
        dtype=dtypes,
        skiprows=1,
    ).drop("Date", axis=1)


def main():
    df = read_campaign(campaign)
    df_env = read_env(campaign)
    merge_env(df, df_env, campaign)


if "__name__ == __main__":
    campaign = "MPACE"
    main()
