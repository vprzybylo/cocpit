'''CPI campaign data processing functions for dataframes'''

import dask.dataframe as dd
import pandas as pd
import globals
import numpy as np
from dash_extensions.enrich import (
    Input,
    Output,
    State,
    ServersideOutput,
)
import datetime


def read_campaign(campaign):
    '''read particle property df and environmental property df_env
    merged based on filename and date'''

    campaign = 'CRYSTAL_FACE_NASA' if campaign == 'CRYSTAL FACE (NASA)' else campaign
    campaign = 'CRYSTAL_FACE_UND' if campaign == 'CRYSTAL FACE (UND)' else campaign
    campaign = 'ICE_L' if campaign == 'ICE L' else campaign
    campaign = 'AIRS_II' if campaign == 'AIRS II' else campaign
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
    ]
    return df


def rename(df):
    '''remove underscores in particle properties in classification column'''
    rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
    df = df.replace(rename_types)
    rename_types = dict(zip(globals.campaigns, globals.campaigns_rename))
    df = df.replace(rename_types)
    return df


def update_layout(fig, len_df, contour=False):
    '''update figures to have white background, and include and center sample size in title'''
    fig.update_layout(
        {
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },
        xaxis_showgrid=True,
        xaxis_zeroline=False,
        title={
            'text': f"n={len_df}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    if contour:
        fig.update_layout(legend=dict(itemsizing='constant'))
        return fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    else:
        return fig.update_traces(width=1, points=False)


def register(app):
    @app.callback(
        [
            ServersideOutput("df-classification", "data"),
            ServersideOutput("df-lat", "data"),
            ServersideOutput("df-lon", "data"),
            ServersideOutput("df-alt", "data"),
            ServersideOutput("df-prop", "data"),
            ServersideOutput("df-iwc", "data"),
            ServersideOutput("df-temp", "data"),
            ServersideOutput("len-df", "data"),
        ],
        [
            Input('submit-button', 'n_clicks'),
            State("campaign-dropdown", "value"),
            State("min-temp", "value"),
            State("max-temp", "value"),
            State("min-pres", "value"),
            State("max-pres", "value"),
            State("date-picker", 'start_date'),
            State("date-picker", 'end_date'),
            State("property-dropdown", "value"),
        ],
        memoize=True,
    )
    def preprocess(
        n_clicks,
        campaign,
        min_temp,
        max_temp,
        min_pres,
        max_pres,
        start_date,
        end_date,
        prop,
    ):
        '''read campaign data and process based on user input from menu'''
        df = read_campaign(campaign)
        df = rename(df)
        tic = datetime.datetime.now()
        df = remove_bad_data(df)
        df = df[df['Temperature'].between(min_temp, max_temp)]
        df = df[df['Pressure'].between(min_pres[0], max_pres[0])]
        df['date'] = df['date'].str.split(' ').str[0]
        df = df[df['date'].between(start_date, end_date)]

        toc = datetime.datetime.now()
        print(f"time to process data = {(toc-tic).total_seconds()}")

        return (
            df['Classification'],
            df['Latitude'],
            df['Longitude'],
            df['Altitude'],
            df[prop],
            df['Ice Water Content'],
            df['Temperature'],
            len(df),
        )

    @app.callback(
        [
            Output('date-picker', 'min_date_allowed'),
            Output('date-picker', 'max_date_allowed'),
            Output('date-picker', 'start_date'),
            Output('date-picker', 'end_date'),
        ],
        Input('campaign-dropdown', 'value'),
    )
    def set_date_picker(campaign):
        '''update date picker based on campaign start and end dates'''
        return (
            globals.min_dates[campaign],
            globals.max_dates[campaign],
            globals.campaign_start_dates[campaign],
            globals.campaign_end_dates[campaign],
        )
