'''Set up the app layout'''

import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import globals
from dash import dcc, html


def content():

    storage = 'session'
    return html.Div(
        id="page-content",
        children=[
            dcc.Store(id='side_click'),
            dcc.Store(id='store-df', storage_type=storage),
            dcc.Store(id='df-classification', storage_type=storage),
            dcc.Store(id='df-lat', storage_type=storage),
            dcc.Store(id='df-lon', storage_type=storage),
            dcc.Store(id='df-alt', storage_type=storage),
            dcc.Store(id='df-date', storage_type=storage),
            dcc.Store(id='df-env', storage_type=storage),
            dcc.Store(id='df-temp', storage_type=storage),
            dcc.Store(id='df-prop', storage_type=storage),
            dcc.Store(id='len-df', storage_type=storage),
            dls.Hash(
                [
                    html.Div(
                        className='row justify-content-around',
                        children=[
                            html.Div(
                                className='col-xs-12 col-sm-12 col-md-12 col-lg-12 col-xl-12',
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
                                                        Select an empty region to reset view with all data points.',
                                                        className='p',
                                                    ),
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
                                            dbc.CardHeader("Vertical Distribution"),
                                            dbc.CardBody(
                                                children=[
                                                    dcc.Graph(id='vert-dist', figure={})
                                                ]
                                            ),
                                        ],
                                        className='card-body',
                                    )
                                ],
                                className='col-sm-12 col-md-12 col-lg-6 col-xl-6',
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
                        #  className='justify-content-center align-items-center',
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
                                            dbc.CardHeader("Environmental Property"),
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
                        #  className='justify-content-center align-items-center',
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
