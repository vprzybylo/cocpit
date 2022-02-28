import dash_bootstrap_components as dbc
from dash import html


def legend():
    return html.Div(
        children=[
            dbc.Row(
                className="d-flex justify-content-around",
                children=[
                    dbc.Col(
                        className="d-flex agg justify-content-around shadow m-2",
                        children=[
                            dbc.Col(
                                children=[
                                    dbc.Row(
                                        className='justify-content-center text-center text-gray-900  h4 text-uppercase',
                                        children=['Aggregate'],
                                    ),
                                ],
                                className='p-2',
                            ),
                            dbc.Col(
                                children=[
                                    html.Img(
                                        className='w-25 m-auto px-3 align-middle',
                                        src='assets/agg.png',
                                    ),
                                ],
                                className='d-flex',
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
