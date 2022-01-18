'''Set up the app layout'''

from globals import *
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
from datetime import date


def layout(app):
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
                            value='CRYSTAL_FACE_UND',
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
                                {'label': i, 'value': i} for i in particle_properties
                            ],
                            placeholder="Particle Property",
                            value='Complexity',
                        ),
                        # width={'size': 7, 'offset': 0, 'order': 2},
                        xs=4,
                        sm=4,
                        md=4,
                        lg=2,
                        xl=2,
                    ),
                    dbc.Col(
                        dcc.DatePickerRange(
                            end_date=date(2017, 6, 21),
                            display_format='MMM Do, YY',
                            start_date_placeholder_text='MMM Do, YY',
                        ),
                        style={"display": "inline-block"},
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
                        dcc.Graph(id='prop_fig', figure={}),
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
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id='top-down map', figure={}),
                        xs=12,
                        sm=12,
                        md=12,
                        lg=6,
                        xl=6,
                    ),
                    dbc.Col(
                        dcc.Graph(id='3d map', figure={}),
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
    return app
