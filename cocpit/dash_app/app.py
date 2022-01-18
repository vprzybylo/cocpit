import cocpit.config as config
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import app_layout
from globals import *
import os
from dotenv import load_dotenv

load_dotenv()

MY_ENV_VAR = os.getenv('MY_ENV_VAR')
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
    ],
)


def remove_baddata(df):
    df = df[df["filled_circular_area_ratio"] != -999.0]
    df = df[df["complexity"] != 0.0]
    df = df[df["complexity"] != -0.0]

    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df.dropna(inplace=True)
    return df


def read_campaign(campaign):
    df = pd.read_csv(f"{config.FINAL_DIR}{campaign}.csv")
    return df


def read_env_campaign(campaign):
    df = pd.read_csv(f"{config.FINAL_DIR}environment/{campaign}.csv")
    return df


def rename(df):
    rename_types = dict(zip(particle_types, particle_types_rename))
    rename_props = dict(zip(particle_properties, particle_properties_rename))
    df = df.rename(columns=rename_types)
    df = df.replace(rename_types)
    df = df.rename(columns=rename_props)
    return df


@app.callback(
    Output("pie", "figure"),
    Input("campaign-dropdown", "value"),
)
def percent_part_type(campaign):
    df = read_campaign(campaign)
    values = df['classification'].value_counts().values
    pie = px.pie(
        df,
        color_discrete_sequence=px.colors.qualitative.Antique,
        values=values,
        names=df['classification'].unique(),
    )
    return pie


@app.callback(
    Output("prop_fig", "figure"),
    [
        Input("campaign-dropdown", "value"),
        Input("property-dropdown", "value"),
    ],
)
def particle_property_fig(campaign, prop):
    df = read_campaign(campaign)
    df = remove_baddata(df)
    df = rename(df)

    fig = px.box(
        df,
        x='classification',
        y=prop,
        color="classification",
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "classification": "Particle Type",
        },
    )
    return fig


@app.callback(
    Output("top-down map", "figure"),
    Input("campaign-dropdown", "value"),
)
def map_top_down(campaign):

    df_env = read_env_campaign(campaign)
    # Find Lat Long center
    lat_center = df_env['latitude'][df_env['latitude'] != -999.99].mean()
    lon_center = df_env['longitude'][df_env['latitude'] != -999.99].mean()
    print(lat_center, lon_center)

    df = read_campaign(campaign)
    fig = px.scatter_mapbox(
        df_env,
        lat="latitude",
        lon="longitude",
        color=df['classification'],
        # color_discrete_sequence=px.colors.qualitative.Antique,
    )
    # Specify layout information
    fig.update_layout(
        mapbox=dict(
            style='satellite',
            accesstoken=os.getenv('MAPBOX_TOKEN'),
            center=dict(lon=lon_center, lat=lat_center),
            zoom=5,
        )
    )
    return fig


@app.callback(
    Output("3d map", "figure"),
    Input("campaign-dropdown", "value"),
)
def map_top_down(campaign):

    df = read_campaign(campaign)
    df_env = read_env_campaign(campaign)

    classes = df['classification'][
        (df_env['latitude'] != -999.99)
        & (df_env['longitude'] != -999.99)
        & (df_env['altitude'] != -999.99)
    ]

    df_env = df_env[
        (df_env['latitude'] != -999.99)
        & (df_env['longitude'] != -999.99)
        & (df_env['altitude'] != -999.99)
    ]

    fig = px.scatter_3d(
        df_env, x='latitude', y='longitude', z='altitude', color=classes
    )

    return fig


# Run local server
if __name__ == '__main__':
    app = app_layout.layout(app)
    app.run_server(port=8050, debug=True)
