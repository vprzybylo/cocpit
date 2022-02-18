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
            dbc.Row(
                className="row",
                children=[
                    dbc.Col(
                        className="card col-xs-12 col-sm-12 col-md-2 col-lg-3 col-xl-3 mb-2 border-left-primary shadow h-100 py-2",
                        children=[
                            html.Div(
                                className='card-body',
                                children=[
                                    html.Div(
                                        className='row no-gutters align-items-center',
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        className='text-xs font-weight-bold text-primary text-uppercase mb-1',
                                                        children=['test'],
                                                    ),
                                                    html.Div(
                                                        className="card-content h5 mb-0 font-weight-bold text-gray-900",
                                                        children=['30000'],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="card col-xs-12 col-sm-12 col-md-2 col-lg-3 col-xl-3 mb-2 border-left-primary shadow h-100 py-2",
                        children=[
                            html.Div(
                                className='card-body',
                                children=[
                                    html.Div(
                                        className='row no-gutters align-items-center',
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        className='text-xs font-weight-bold text-primary text-uppercase mb-1',
                                                        children=['test'],
                                                    ),
                                                    html.Div(
                                                        className="card-content h5 mb-0 font-weight-bold text-gray-900",
                                                        children=['3000'],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="card col-xs-12 col-sm-12 col-md-2 col-lg-3 col-xl-3 mb-2 border-left-primary shadow h-100 py-2",
                        children=[
                            html.Div(
                                className='card-body',
                                children=[
                                    html.Div(
                                        className='row no-gutters align-items-center',
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.Div(
                                                        className='text-xs font-weight-bold text-primary text-uppercase mb-1',
                                                        children=['test'],
                                                    ),
                                                    html.Div(
                                                        className="card-content h5 mb-0 font-weight-bold text-gray-900",
                                                        children=['3999'],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
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
                                                style={
                                                    'text-align': 'center',
                                                    'margin': '0px',
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                # className='card-body',
                            )
                        ],
                        xs=12,
                        sm=12,
                        md=12,
                        lg=4,
                        xl=4,
                    ),
                    dbc.Col(
                        children=[
                            dbc.Card(
                                [
                                    dbc.CardHeader("Particle Type Percentage"),
                                    dbc.CardBody(
                                        children=[dcc.Graph(id='pie', figure={})]
                                    ),
                                ],
                                # className='card-body',
                            )
                        ],
                        xs=12,
                        sm=12,
                        md=12,
                        lg=4,
                        xl=4,
                    ),
                ],
                align="center",
                justify="center",
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
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
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
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
                            ),
                        ],
                        align="center",
                        justify="center",
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
                                    )
                                ],
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
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
                                    )
                                ],
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
                            ),
                        ],
                        align="center",
                        justify="center",
                    ),
                ]
            ),
            html.P(
                'Copyright All Rights Reserved',
                id='copyright',
            ),
        ],
    )
