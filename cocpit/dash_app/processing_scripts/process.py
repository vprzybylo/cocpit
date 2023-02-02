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
    """remove bad data for particle properties"""
    print('1', len(df))
    df = df[
        # (df['Latitude'] != 0)
        # & (df['Longitude'] != 0)
        (df["Pressure"] != 0)
        & (df["Ice Water Content"] > 1e-5)
        # & (df["Complexity"] != -0.0)
        & (df["Particle Height"] != 0.0)
        & (df["Particle Width"] != 0.0)
    ]
    print('2', len(df))
    df = df.replace([-999.99, -999.0], np.nan).dropna()
    print('3', len(df))
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
