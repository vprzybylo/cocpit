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


def read_campaign(campaign):
    df_env = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/environment/{campaign}.csv",
        names=[
            'filename',
            'date',
            'Latitude',
            'Longitude',
            'Altitude',
            'Pressure',
            'Temperature',
            'Ice Water Content',
        ],
        header=1,
    )
    df = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=[
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
        ],
        header=1,
    )
    df = pd.merge(df, df_env, on=['filename', 'date'])
    return df


def remove_bad_props(df):
    df = df[df["Area Ratio"] != -999.0]
    df = df[df["Complexity"] != 0.0]
    df = df[df["Complexity"] != -0.0]

    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df.dropna(inplace=True)
    return df


def remove_bad_env(df):
    df = df[
        (df['Latitude'] != -999.99)
        & (df['Latitude'] != 0)
        & (df['Longitude'] != -999.99)
        & (df['Longitude'] != 0)
        & (df['Altitude'] != -999.99)
        & (df['Temperature'] != -999.99)
        & (df['Temperature'].notna())
        & (df['Ice Water Content'] != -999.99)
        & (df['Ice Water Content'].notna())
        & (df['Ice Water Content'] > 1e-5)
    ]
    return df


def rename(df):
    '''remove underscores in particle properties in classification column'''
    rename_types = dict(zip(particle_types, particle_types_rename))
    df = df.replace(rename_types)
    return df


@app.callback(
    Output("pie", "figure"),
    Input("campaign-dropdown", "value"),
)
def percent_part_type(campaign):
    '''pie chart for percentage of particle types for a given campaign'''
    df = read_campaign(campaign)
    df = rename(df)
    values = df['Classification'].value_counts().values
    pie = px.pie(
        df,
        color_discrete_sequence=px.colors.qualitative.Antique,
        values=values,
        names=df['Classification'].unique(),
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
    df = remove_bad_props(df)
    df = rename(df)

    fig = px.box(
        df,
        x='Classification',
        y=prop,
        color="Classification",
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "Classification": "Particle Type",
        },
    )
    return fig


@app.callback(
    Output("top-down map", "figure"),
    Input("campaign-dropdown", "value"),
)
def map_top_down(campaign):
    '''aircraft location and particle type overlaid on map'''
    df = read_campaign(campaign)
    df = remove_bad_env(df)
    # Find Lat Long center
    lat_center = df['Latitude'][df['Latitude'] != -999.99].mean()
    lon_center = df['Longitude'][df['Latitude'] != -999.99].mean()

    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color='Classification',
        size=df['Ice Water Content'] * 4,
        color_discrete_sequence=px.colors.qualitative.Antique,
        hover_data=['Ice Water Content'],
    )
    # Specify layout information
    fig.update_layout(
        mapbox=dict(
            style='light',
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
def three_d_map(campaign):

    df = read_campaign(campaign)
    df = remove_bad_env(df)
    if campaign == 'CRYSTAL_FACE_NASA':
        df = df[df['Latitude'] > 23.0]

    fig = px.scatter_3d(
        df,
        x='Latitude',
        y='Longitude',
        z='Altitude',
        range_x=[min(df['Latitude']), max(df['Latitude'])],
        range_y=[min(df['Longitude']), max(df['Longitude'])],
        color='Temperature',
        color_continuous_scale=px.colors.diverging.balance,
        color_continuous_midpoint=0.0,
        range_color=[min(df['Temperature']), max(df['Temperature'])],
        size=df['Ice Water Content'] * 4,
        hover_data=['Ice Water Content'],
    )
    fig.update_traces(mode='markers', marker_line_width=0)
    fig.update_layout(title_text=f"{len(df)} datapoints", title_x=0.5)
    return fig


# Run local server
if __name__ == '__main__':
    app = app_layout.layout(app)
    app.run_server(port=8050, debug=True)
