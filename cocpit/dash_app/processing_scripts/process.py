"""CPI campaign data processing functions for dataframes"""

import globals
import numpy as np
import pandas as pd


def read_campaign(campaign):
    """read particle property df and environmental property df_env
    merged based on filename and date"""

    # replace spaces with underscores and remove parentheses
    # filenames don't have spaces or parentheses
    campaign = campaign.replace(" ", "_").replace("(", "").replace(")", "")

    return pd.read_parquet(
        f"/home/vanessa/hulk/cocpit/final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet",
        engine="fastparquet",
    )  


def remove_bad_data(df):
    """remove bad data"""
    df = df[
        (df["Pressure [hPa]"] != 0)
        & (df["Ice Water Content [g/m3]"] > 1e-5)
        & (df["Particle Height [micrometers]"] != 0.0)
        & (df["Particle Width [micrometers]"] != 0.0)
    ]
    df = df.replace([-999.99, -999.0], np.nan).dropna()
    return df


def rename(df):
    """remove underscores in particle properties in classification column"""
    rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
    df = df.replace(rename_types)
    rename_types = dict(zip(globals.campaigns, globals.campaigns_rename))
    df = df.replace(rename_types)
    return df


def update_layout(fig, contour=False, margin=20, height=300):
    """update figures to have white background"""
    fig.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        },
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        xaxis_showgrid=True,
        xaxis_zeroline=False,
        showlegend=False,
        height=height,
    )

    if not contour:
        return fig.update_traces(width=1, points=False)
    fig.update_layout(legend=dict(itemsizing="constant"))
    return fig.update_yaxes(showline=True, linewidth=1, linecolor="black")
