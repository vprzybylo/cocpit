import cocpit.config as config
from datetime import date
import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
# https://www.bootstrapcdn.com/bootswatch/
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
    ],
)

campaigns = [
    "AIRS_II",
    "ARM",
    "ATTREX",
    "CRYSTAL_FACE_UND",
    "CRYSTAL_FACE_NASA",
    "ICE_L",
    "IPHEX",
    "ISDAC",
    "MACPEX",
    "MC3E",
    "MIDCIX",
    "MPACE",
    "POSIDON",
    "OLYMPEX",
]

particle_types = [
    "agg",
    "budding",
    "bullet",
    "column",
    "compact_irreg",
    "fragment",
    "planar_polycrystal",
    "rimed",
    "sphere",
]

particle_types_rename = [
    "agg",
    "budding",
    "bullet",
    "column",
    "compact irregular",
    "fragment",
    "planar polycrystal",
    "rimed",
    "sphere",
]

particle_properties = [
    'cnt_area',
    'circularity',
    'solidity',
    'complexity',
    'equiv_d',
    'convex_perim',
    'hull_area',
    'perim',
    'phi',
    'filled_circular_area_ratio',
    'roundness',
    'perim_area_ratio',
]

particle_properties_rename = [
    'contour area',
    'circularity',
    'solidity',
    'complexity',
    'equivalent diameter',
    'convex perimeter',
    'hull area',
    'perimeter',
    'aspect ratio',
    'area ratio',
    'roundness',
    'perimeter-area ratio',
]

# Set up the app layout
app.layout = dbc.Container(
    [
        # dbc.Row(
        #     dbc.Col(
        #         html.H3("COCPIT", className='text-center text-primary mb-4'), width=12
        #     )
        # ),
        dbc.Row(
            dbc.Col(
                html.H3(
                    "Classification of Cloud Particle Imagery and Thermodynamics (COCPIT)",
                    className='text-center text-primary mb-4',
                ),
                width=12,
            ),
        ),
        dbc.Row(
            html.Div(
                children=[
                    html.H6(
                        'Images classified from the:',
                        # style={'display': 'inline-block'},
                    ),
                    html.A(
                        " Cloud Particle Imager",
                        href="http://www.specinc.com/cloud-particle-imager",
                        # style={'display': 'inline-block'},
                    ),
                ],
                className='text-center mb-4',
            ),
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id='campaign-dropdown',
                        multi=False,
                        options=[{'label': i, 'value': i} for i in campaigns],
                        placeholder="Campaign",
                        value='AIRS_II',
                    ),
                    xs=4,
                    sm=4,
                    md=4,
                    lg=2,
                    xl=2,
                    # width={'size': 5, 'offset': 0, 'order': 1},
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id='property-dropdown',
                        options=[
                            {'label': i, 'value': i} for i in particle_properties_rename
                        ],
                        placeholder="Particle Property",
                        value='complexity',
                    ),
                    # width={'size': 7, 'offset': 0, 'order': 2},
                    xs=4,
                    sm=4,
                    md=4,
                    lg=2,
                    xl=2,
                ),
            ],
            align="center",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='pie', figure={}), xs=12, sm=12, md=12, lg=5, xl=5
                ),
                dbc.Col(
                    dcc.Graph(id='prop_fig', figure={}), xs=12, sm=12, md=12, lg=6, xl=6
                ),
            ],
            align="center",
            justify="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='top-down map', figure={}),
                    xs=12,
                    sm=12,
                    md=12,
                    lg=5,
                    xl=5,
                ),
                dbc.Col(
                    dcc.Graph(id='vertical map', figure={}),
                    xs=12,
                    sm=12,
                    md=12,
                    lg=6,
                    xl=6,
                ),
            ],
            align="center",
            justify="center",
        ),
    ],
    fluid=True,
    style={"padding": "15px"},
)


def remove_baddata(df_CPI):
    df_CPI = df_CPI[df_CPI["filled_circular_area_ratio"] != -999.0]
    df_CPI = df_CPI[df_CPI["complexity"] != 0.0]
    df_CPI = df_CPI[df_CPI["complexity"] != -0.0]

    df_CPI = df_CPI[df_CPI.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df_CPI.dropna(inplace=True)
    return df_CPI


def choose_campaign(campaign):
    df = pd.read_csv(f"{config.FINAL_DIR}{campaign}.csv")
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
    df = choose_campaign(campaign)
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
    df = choose_campaign(campaign)
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


# Run local server
if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
