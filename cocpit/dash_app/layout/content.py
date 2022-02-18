'''Set up the app layout'''

import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import globals
from dash import dcc, html
from layout import header_info


def content():

    storage = 'session'
    return html.Div(
        id="page-content",
        children=[
            dcc.Store(id='store-df', storage_type=storage),
            dcc.Store(id='df-classification', storage_type=storage),
            dcc.Store(id='df-lat', storage_type=storage),
            dcc.Store(id='df-lon', storage_type=storage),
            dcc.Store(id='df-alt', storage_type=storage),
            dcc.Store(id='df-env', storage_type=storage),
            dcc.Store(id='df-temp', storage_type=storage),
            dcc.Store(id='df-prop', storage_type=storage),
            dcc.Store(id='len-df', storage_type=storage),
            dls.Hash(
                [
                    html.Div(
                        className='row',
                        children=[
                            html.Div(
                                className='col-sm-12 col-md-12 col-lg-6 col-xl-6',
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Particle Location"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(
                                                        id='top-down-map',
                                                        figure={},
                                                    ),
                                                    html.P(
                                                        'Hover over image and choose box select icon to update all figures based on chosen location. \n \
                                                        Select empty region to reset view with all data points.',
                                                        className='p',
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                            ),
                            html.Div(
                                className='col-sm-12 col-md-12 col-lg-6 col-xl-6',
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Particle Type Percentage"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(id='pie', figure={})
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dls.Hash(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Cross-Section (Longitude)"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(
                                                        id='lon-alt-hist', figure={}
                                                    )
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                                className='col-sm-12 col-md-12 col-lg-6 col-xl-6',
                            ),
                            dbc.Col(
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Cross Section (Latitude)"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(
                                                        id='lat-alt-hist', figure={}
                                                    )
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                                className='col-sm-12 col-md-12 col-lg-6 col-xl-6',
                            ),
                        ],
                        # className='justify-content-center align-items-center',
                    ),
                ],
            ),
            dls.Hash(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Geometric Property"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(id='prop-fig', figure={})
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                                className='col col-sm-12 col-md-12 col-lg-6 col-xl-6',
                            ),
                            dbc.Col(
                                children=[
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Environmental Attributes"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(
                                                        id='type-env-violin', figure={}
                                                    )
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                                className='col col-sm-12 col-md-12 col-lg-6 col-xl-6',
                            ),
                        ],
                        # className='justify-content-center align-items-center',
                    ),
                ]
            ),
            html.Hr(),
            html.P(
                'Copyright All Rights Reserved',
                id='copyright',
            ),
        ],
    )
