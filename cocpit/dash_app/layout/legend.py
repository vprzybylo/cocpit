import dash_bootstrap_components as dbc
from dash import html


def legend():
    return html.Div(
        children=[
            dbc.CardGroup(
                className="justify-content-around",
                children=[
                    dbc.Col(
                        className="agg legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/agg.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center py-2 h4 text-uppercase text-white',
                                children=['Aggregate'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="budding legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/budding.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white',
                                children=['Budding Rosette'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="bullet legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='img-fluid card-img shadow p-1',
                                    src='assets/bullet.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white',
                                children=['Bullet Rosette'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="column legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/column.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white ',
                                children=['Column'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="compact legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/compact.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white',
                                children=['Compact Irregular'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="planar legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/planar.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white',
                                children=['Planar Polycrystal'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="rimed legend p-1 d-flex justify-space-between",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/rimed.png',
                                ),
                            ),
                            dbc.Row(
                                className='my-auto justify-content-center text-center p-1 h4 text-uppercase text-white',
                                children=['Rimed'],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
