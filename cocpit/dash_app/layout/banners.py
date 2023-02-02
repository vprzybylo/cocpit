import dash_bootstrap_components as dbc
from dash import html
import globals


def banners():
    return html.Div(
        children=[
            dbc.Row(
                className="d-flex justify-content-around",
                children=[
                    dbc.Col(
                        className="d-flex col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3  justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className="m-2 d-flex justify-content-center text-center text-gray-900 h4 text-uppercase",
                                        children=["Image Count:"],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center fw-bolder",
                                        id="image-count",
                                        children=[396139],  # default for CF UND
                                    ),
                                ],
                                className="p-2",
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className="fa fa-image m-1 text-gray-300",
                                    ),
                                ],
                                className="my-auto",
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="d-flex col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3 justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className="m-2 justify-content-center text-center text-gray-900 h4 text-uppercase",
                                        children=["Flight Count:"],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center fw-bolder",
                                        id="flight-count",
                                        children=[4],
                                    ),
                                ],
                                className="p-2",
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className="fa fa-plane m-1 text-gray-300",
                                    ),
                                ],
                                className="my-auto",
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="d-flex col-xs-12 col-sm-3 col-md-3 col-lg-3 col-xl-3 justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className="m-2 justify-content-center text-center text-gray-900 h4 text-uppercase",
                                        children=["Flight Hours:"],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center fw-bolder",
                                        id="flight-hours",
                                        children=[
                                            15,  # default flight hours for CF UND
                                        ],
                                    ),
                                ],
                                className="p-2",
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className="fa fa-clock m-1 text-gray-300",
                                    ),
                                ],
                                className="my-auto",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
