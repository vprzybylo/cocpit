import cocpit.config as config
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.express as px
import app_layout
import globals
import os
from dotenv import load_dotenv
from dash import dcc

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
    '''read particle property df and environmental property df_env
    merge based on filename and date'''

    df_env = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/environment/{campaign}.csv",
        names=globals.col_names_env,
        header=0,
    )
    df = pd.read_csv(
        f"../../final_databases/vgg16/v1.4.0/{campaign}.csv",
        names=globals.col_names,
        header=0,
    )
    df = pd.merge(df, df_env, on=['filename', 'date'])
    return df


@app.callback(
    # dash.dependencies.Output('date-picker', 'start_date'),
    # dash.dependencies.Output('date-picker', 'end_date'),
    dash.dependencies.Output('date-picker', 'min_date_allowed'),
    dash.dependencies.Output('date-picker', 'max_date_allowed'),
    [dash.dependencies.Input('campaign-dropdown', 'value')],
)
def set_date_picker(campaign):
    '''update date picker based on campaign start and end dates'''
    print(globals.campaign_start_dates[campaign], globals.campaign_end_dates[campaign])
    return globals.campaign_start_dates[campaign], globals.campaign_end_dates[campaign]


def remove_bad_props(df):
    '''remove bad data for particle geometric properties (e.g., no particle area)'''
    df = df[df["Area Ratio"] != -999.0]
    df = df[df["Complexity"] != 0.0]
    df = df[df["Complexity"] != -0.0]

    df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df.dropna(inplace=True)
    return df


def remove_bad_env(df):
    '''remove missing or bad environmental data'''
    df = df[
        (df['Latitude'] != -999.99)
        & (df['Latitude'] != 0)
        & (df['Longitude'] != -999.99)
        & (df['Longitude'] != 0)
        & (df['Altitude'] != -999.99)
        & (df['Temperature'] != -999.99)
        & (df['Pressure'] != 0)
        & (df['Pressure'] != -999.99)
        & (df['Temperature'].notna())
        & (df['Ice Water Content'] != -999.99)
        & (df['Ice Water Content'].notna())
        & (df['Ice Water Content'] > 1e-5)
    ]
    return df


def check_date_range(df, start_date, end_date):
    df['date'] = df['date'].str.split(' ').str[0]
    df = df[df['date'].between(start_date, end_date)]
    return df


def check_temp_range(df, min_temp, max_temp):
    '''find temperature within a user range from input'''
    df = df[df['Temperature'] >= int(min_temp)]
    df = df[df['Temperature'] <= int(max_temp)]
    return df


def check_pres_range(df, min_pres, max_pres):
    '''find pressure within a user range from slider'''
    df = df[df['Pressure'] >= int(min_pres)]
    df = df[df['Pressure'] <= int(max_pres)]
    return df


def rename(df):
    '''remove underscores in particle properties in classification column'''
    rename_types = dict(zip(globals.particle_types, globals.particle_types_rename))
    df = df.replace(rename_types)
    return df


@app.callback(
    Output("pie", "figure"),
    Input("campaign-dropdown", "value"),
    Input("min-temp", "value"),
    Input("max-temp", "value"),
    Input("min-pres", "value"),
    Input("max-pres", "value"),
    Input("date-picker", 'start_date'),
    Input("date-picker", 'end_date'),
)
def percent_part_type(
    campaign, min_temp, max_temp, min_pres, max_pres, start_date, end_date
):
    '''pie chart for percentage of particle types for a given campaign'''
    df = read_campaign(campaign)
    df = remove_bad_props(df)
    df = remove_bad_env(df)
    # print(df['Pressure'].min(), df['Pressure'].max())
    df = rename(df)
    df = check_temp_range(df, min_temp, max_temp)
    df = check_pres_range(df, min_pres[0], max_pres[0])
    df = check_date_range(df, start_date, end_date)

    values = df['Classification'].value_counts().values
    fig = px.pie(
        df,
        color_discrete_sequence=px.colors.qualitative.Antique,
        values=values,
        names=df['Classification'].unique(),
    )
    fig.update_layout(
        title={
            'text': f"n={len(df)}",
            'x': 0.43,
            'xanchor': 'center',
            'yanchor': 'top',
        }
    )
    return fig


@app.callback(
    Output("prop_fig", "figure"),
    [
        Input("campaign-dropdown", "value"),
        Input("property-dropdown", "value"),
        Input("min-temp", "value"),
        Input("max-temp", "value"),
        Input("min-pres", "value"),
        Input("max-pres", "value"),
        Input("date-picker", 'start_date'),
        Input("date-picker", 'end_date'),
    ],
)
def particle_property_fig(
    campaign, prop, min_temp, max_temp, min_pres, max_pres, start_date, end_date
):
    df = read_campaign(campaign)
    df = remove_bad_props(df)
    df = remove_bad_env(df)
    df = rename(df)
    df = check_temp_range(df, min_temp, max_temp)
    df = check_pres_range(df, min_pres[0], max_pres[0])
    df = check_date_range(df, start_date, end_date)

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
    fig.update_layout(
        title={
            'text': f"n={len(df)}",
            'x': 0.43,
            'xanchor': 'center',
            'yanchor': 'top',
        }
    )
    return fig


@app.callback(
    Output("top-down map", "figure"),
    Input("campaign-dropdown", "value"),
    Input("map-particle_type", "value"),
    Input("min-temp", "value"),
    Input("max-temp", "value"),
    Input("min-pres", "value"),
    Input("max-pres", "value"),
    Input("date-picker", 'start_date'),
    Input("date-picker", 'end_date'),
)
def map_top_down(
    campaign, part_type, min_temp, max_temp, min_pres, max_pres, start_date, end_date
):
    '''aircraft location and particle type overlaid on map'''
    df = read_campaign(campaign)
    df = remove_bad_env(df)
    df = rename(df)
    df = df[df['Classification'].isin(part_type)]
    df = check_temp_range(df, min_temp, max_temp)
    df = check_pres_range(df, min_pres[0], max_pres[0])
    df = check_date_range(df, start_date, end_date)

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
        hover_data={'Ice Water Content': True, 'Temperature': True, 'Pressure': True},
        custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
    )
    fig.update_traces(
        hovertemplate="<br>".join(
            [
                "Latitude: %{lat}",
                "Longitude: %{lon}",
                "Temperature: %{customdata[0]}",
                "Pressure: %{customdata[1]}",
                "Ice Water Content: %{customdata[2]}",
            ]
        ),
    )
    # Specify layout information
    fig.update_layout(
        title={
            'text': f"n={len(df)}",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        mapbox=dict(
            style='light',
            accesstoken=os.getenv('MAPBOX_TOKEN'),
            center=dict(lon=lon_center, lat=lat_center),
            zoom=5,
        ),
    )

    return fig


@app.callback(
    Output("3d map", "figure"),
    Input("campaign-dropdown", "value"),
    Input("map-particle_type", "value"),
    Input("3d_vertical_prop", "value"),
    Input("min-temp", "value"),
    Input("max-temp", "value"),
    Input("min-pres", "value"),
    Input("max-pres", "value"),
    Input("date-picker", 'start_date'),
    Input("date-picker", 'end_date'),
)
def three_d_map(
    campaign,
    part_type,
    vert_prop,
    min_temp,
    max_temp,
    min_pres,
    max_pres,
    start_date,
    end_date,
):

    df = read_campaign(campaign)
    df = remove_bad_env(df)
    df = rename(df)
    if campaign == 'CRYSTAL_FACE_NASA':
        df = df[df['Latitude'] > 23.0]
    df = df[df['Classification'].isin(part_type)]
    df = check_temp_range(df, min_temp, max_temp)
    df = check_pres_range(df, min_pres[0], max_pres[0])
    df = check_date_range(df, start_date, end_date)

    if vert_prop == 'Temperature':
        zrange = [min(df['Temperature']), 10]
    else:
        zrange = [df[vert_prop].min(), df[vert_prop].max()]

    fig = px.scatter_3d(
        df,
        x='Latitude',
        y='Longitude',
        z=vert_prop,
        range_z=zrange,
        color=vert_prop,
        color_continuous_scale=px.colors.sequential.Blues[::-1],
        hover_data={'Ice Water Content': True, 'Temperature': True, 'Pressure': True},
        custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
        size=df['Ice Water Content'] * 5,
    )
    fig.update_traces(
        mode='markers',
        marker_line_width=0,
        hovertemplate="<br>".join(
            [
                "Latitude: %{x}",
                "Longitude: %{y}",
                "Temperature: %{customdata[0]}",
                "Pressure: %{customdata[1]}",
                "Ice Water Content: %{customdata[2]}",
            ]
        ),
    )
    fig.update_layout(
        title={
            'text': f"n={len(df)}",
            'x': 0.45,
            'xanchor': 'center',
            'yanchor': 'top',
        },
    )
    if vert_prop == 'Temperature' or vert_prop == 'Pressure':
        fig.update_scenes(zaxis_autorange="reversed")
    return fig


# Run local server
if __name__ == '__main__':
    sidebar = app_layout.sidebar()
    content = app_layout.content()
    app.layout = dbc.Container(
        [dcc.Location(id="url"), sidebar, content],
        fluid=True,
    )

    app.run_server(port=8050, debug=True)
