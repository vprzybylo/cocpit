'''CPI campaign data processing functions for dataframes'''

import pandas as pd
import globals
import numpy as np


def read_campaign(campaign):
    '''read particle property df and environmental property df_env
    merged based on filename and date'''

    # replace spaces with underscores and remove parentheses
    # filenames don't have spaces or parentheses
    campaign = campaign.replace(" ", "_").replace("(", "").replace(")", "")

    df = pd.read_parquet(
        f"../../final_databases/vgg16/v1.4.0/merged_env/{campaign}.parquet",
        engine='fastparquet',
    )

    return df


def remove_bad_data(df):
    '''remove missing or bad environmental data'''
    df = df.replace([-999.99, -999.0, np.inf, -np.inf], np.nan).dropna()

    df = df[
        (df['Latitude'] != 0)
        & (df['Longitude'] != 0)
        & (df['Pressure'] != 0)
        & (df['Ice Water Content'] > 1e-5)
        & (df["Complexity"] != -0.0)
        & (df['Particle Height'] != 0.0)
        & (df['Particle Width'] != 0.0)
    ]

    return df


def rename(df):
    '''remove underscores in particle properties in classification column'''
    rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
    df = df.replace(rename_types)
    rename_types = dict(zip(globals.campaigns, globals.campaigns_rename))
    df = df.replace(rename_types)
    return df


def update_layout(fig, contour=False, margin=20, height=300):
    '''update figures to have white background, and include and center sample size in title'''
    fig.update_layout(
        {
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        xaxis_showgrid=True,
        xaxis_zeroline=False,
        showlegend=False,
        height=height,
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    if contour:
        fig.update_layout(legend=dict(itemsizing='constant'))
        return fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    else:
        return fig.update_traces(width=1, points=False)
