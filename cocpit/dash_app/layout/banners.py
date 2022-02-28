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
                        className="d-flex justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className=' justify-content-center text-center text-gray-900  h4 text-uppercase',
                                        children=['Image Count:'],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center",
                                        id='image-count',
                                        children=[
                                            globals.campaign_image_count[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                                className='p-2',
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className='fa fa-image m-auto px-3 align-middle',
                                    ),
                                ],
                                className='d-flex icon-bgrd',
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="d-flex justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className=' justify-content-center text-center text-gray-900  h4 text-uppercase',
                                        children=['Flight Count:'],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center",
                                        id='flight-count',
                                        children=[
                                            globals.campaign_flight_count[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                                className='p-2',
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className='fa fa-plane m-auto px-3 align-middle',
                                    ),
                                ],
                                className='d-flex icon-bgrd',
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="d-flex justify-content-around legend shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className=' justify-content-center text-center text-gray-900  h4 text-uppercase',
                                        children=['Flight Hours:'],
                                    ),
                                    dbc.Row(
                                        className="justify-content-center h3 text-center",
                                        id='flight-hours',
                                        children=[
                                            globals.campaign_flight_hours[
                                                'CRYSTAL FACE (UND)'
                                            ],
                                        ],
                                    ),
                                ],
                                className='p-2',
                            ),
                            dbc.Col(
                                children=[
                                    html.I(
                                        className='fa fa-clock m-auto px-3 align-middle',
                                    ),
                                ],
                                className='d-flex icon-bgrd',
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
