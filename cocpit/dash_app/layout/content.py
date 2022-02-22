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
            dcc.Store(id='df-date', storage_type=storage),
            dcc.Store(id='df-env', storage_type=storage),
            dcc.Store(id='df-temp', storage_type=storage),
            dcc.Store(id='df-prop', storage_type=storage),
            dcc.Store(id='len-df', storage_type=storage),
            dbc.Row(id='header'),
            dbc.Row(
                className="d-flex justify-content-around h-50",
                children=[
                    dbc.Col(
                        className="col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3 legend shadow h-50",
                        children=[
                            dbc.Card(
                                children=[
                                    dbc.CardBody(
                                        className='m-auto justify-center text-gray-900 h4 font-weight-bold text-uppercase p-1',
                                        children=['Image Count:'],
                                    ),
                                    dbc.CardBody(
                                        className="m-auto h2 font-weight-bold p-1",
                                        id='image-count',
                                        children=[
                                            globals.campaign_image_count[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3 legend shadow h-50",
                        children=[
                            dbc.Card(
                                children=[
                                    dbc.CardBody(
                                        className='m-auto justify-center text-gray-900 h4 font-weight-bold text-uppercase p-1',
                                        children=['Number of Flights:'],
                                    ),
                                    dbc.CardBody(
                                        className="m-auto h2 font-weight-bold p-1",
                                        id='flight-count',
                                        children=[
                                            globals.campaign_flight_count[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3 legend shadow h-50",
                        children=[
                            dbc.Card(
                                children=[
                                    dbc.CardBody(
                                        className='m-auto justify-center text-gray-900 h4 font-weight-bold text-uppercase p-1',
                                        children=['Flight Hours:'],
                                    ),
                                    dbc.CardBody(
                                        className="m-auto h2 font-weight-bold p-1",
                                        id='flight-hours',
                                        children=[
                                            globals.campaign_flight_hours[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                className="row justify-content-center",
                children=[
                    dbc.Card(
                        className='col-xs-12 col-sm-12 col-md-4 col-lg-2 col-xl-2 m-2 font-weight-bold text-body text-uppercasefont-weight-bold text-body text-uppercase legend-card legend-border agg shadow',
                        children=[
                            dbc.CardImg(
                                className='legend-image m-auto p-2',
                                src="assets/agg.png",
                            ),
                            dbc.CardBody(
                                html.P(
                                    "Aggregate",
                                    className="card-text m-auto",
                                )
                            ),
                        ],
                    ),
                    dbc.Card(
                        className='col-xs-12 col-sm-12 col-md-4 col-lg-2 col-xl-2 m-2 font-weight-bold text-body text-uppercasefont-weight-bold text-body text-uppercase legend-card legend-border compact shadow',
                        children=[
                            dbc.CardImg(
                                className='legend-image m-auto p-2',
                                src="assets/compact.png",
                            ),
                            dbc.CardBody(
                                html.P(
                                    "Compact Irregular",
                                    className="card-text m-auto",
                                )
                            ),
                        ],
                    ),
                ],
            ),
            dbc.Row(
                className="row justify-content-center",
                children=[
                    dbc.Col(
                        className="col-xs-12 col-sm-12 col-md-12 col-lg-2 col-xl-2 m-2 legend legend-border budding shadow h-5 py-2",
                        children=[
                            html.Div(
                                className='row no-gutters align-items-center',
                                children=[
                                    html.Div(
                                        children=[
                                            html.Div(
                                                className='text-xs font-weight-bold text-body text-uppercase mb-1 py-2',
                                                children=['Budding Rosette'],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="col-xs-12 col-sm-12 col-md-12 col-lg-2 col-xl-2 m-2 legend legend-border bullet shadow h-5 py-2",
                        children=[
                            html.Div(
                                className='row no-gutters align-items-center',
                                children=[
                                    html.Div(
                                        children=[
                                            html.Div(
                                                className='text-xs font-weight-bold text-body text-uppercase mb-1 py-2',
                                                children=['Bullet Rosette'],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dls.Hash(
                [
                    html.Div(
                        className='row',
                        children=[
                            html.Div(
                                className='col-sm-12 col-md-12 col-lg-12 col-xl-12',
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
