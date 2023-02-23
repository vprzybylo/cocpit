"""merge environmental data with particle predictions and properties from ML model based on filename"""
import pandas as pd


def merge_env(df, df_env, campaign):
    """merge"""

    df = df.merge(df_env, on="filename")
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
        "Frame Width [pixels]",
        "Frame Height [pixels]",
        "Particle Width [micrometers]",
        "Particle Height [micrometers]",
        "Cutoff [%]",
        "Aggregate [%]",
        "Budding [%]",
        "Bullet Rosette [%]",
        "Column [%]",
        "Compact Irregular [%]",
        "Fragment [%]",
        "Planar Polycrystal [%]",
        "Rimed [%]",
        "Sphere [%]",
        "Classification",
        "Blur",
        "Contours [#]",
        "Edges",
        "Std",
        "Contour Area [pixels]",
        "Contrast",
        "Circularity",
        "Solidity",
        "Complexity",
        "Equivalent Diameter",
        "Convex Perimeter",
        "Hull Area",
        "Perimeter [pixels]",
        "Aspect Ratio",
        "Extreme Points",
        "Area Ratio",
        "Roundness",
        "Perimeter-Area Ratio",
    ]

    return pd.read_csv(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=columns,
        low_memory=False,
        skiprows=1,
    )


def read_env(campaign):
    """read environmental data from Carl
    - Drop date column due to it already being in particle property df
    - Carl's is truncated down to day not msec
    """
    columns = [
        "filename",
        "Date",
        "Latitude [degrees]",
        "Longitude [degrees]",
        "Altitude [m]",
        "Pressure [hPa]",
        "Temperature [C]",
        "Ice Water Content [g/m3]",
        "PSD IWC [g/m3]",
        "concentration ratio",
        "area ratio",
        "mass ratio",
    ]

    return pd.read_csv(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/environment/{campaign}_atmospheric.csv",
        names=columns,
        skiprows=1,
    ).drop("Date", axis=1)


def main():
    df = read_campaign(campaign)
    df_env = read_env(campaign)
    merge_env(df, df_env, campaign)


if "__name__ == __main__":
    campaign = "MC3E"
    main()
