import dash_bootstrap_components as dbc
from dash import html
import globals


def banners():
    return (
        dbc.Row(
            className="d-flex justify-content-around",
            children=[
                dbc.Col(
                    className="col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3 legend shadow",
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardBody(
                                    className='m-auto justify-center text-center text-gray-900 h4 text-uppercase p-1',
                                    children=['Image Count:'],
                                ),
                                dbc.CardBody(
                                    className="m-auto h3 p-1 text-center",
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
                    className="col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3 legend shadow",
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardBody(
                                    className='m-auto justify-center text-center text-gray-900 h4 text-uppercase p-1',
                                    children=['# Flights:'],
                                ),
                                dbc.CardBody(
                                    className="m-auto h3 p-1 text-center",
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
                    className="col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3 legend shadow",
                    children=[
                        dbc.Card(
                            children=[
                                dbc.CardBody(
                                    className='m-auto justify-center text-gray-900 text-center h4 text-uppercase p-1',
                                    children=['Flight Hours:'],
                                ),
                                dbc.CardBody(
                                    className="m-auto h3 p-1 text-center",
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
            className="row justify-content-around",
            children=[
                dbc.Card(
                    className='col-xs-12 col-sm-12 col-md-4 col-lg-2 col-xl-2 m-2 text-body text-uppercase legend-border agg shadow',
                    children=[
                        dbc.CardImg(
                            className='legend-image m-auto p-2',
                            src="assets/agg.png",
                        ),
                        dbc.CardBody(
                            html.P(
                                "Aggregate",
                                className="legend-text m-auto  mb-0",
                            )
                        ),
                    ],
                ),
                dbc.Card(
                    className='col-xs-12 col-sm-12 col-md-4 col-lg-2 col-xl-2 m-2 text-body text-uppercase legend-border compact shadow',
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
    )
