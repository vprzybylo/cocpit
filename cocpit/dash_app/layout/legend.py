import dash_bootstrap_components as dbc
from dash import html


def legend():
    return html.Div(
        children=[
            dbc.CardGroup(
                className="justify-content-around",
                children=[
                    dbc.Col(
                        className="agg legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow',
                                    src='assets/agg.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Aggregate'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="budding legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow',
                                    src='assets/budding.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Budding Rosette'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="bullet legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow',
                                    src='assets/bullet.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Bullet Rosette'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="column legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow p-1',
                                    src='assets/column.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Column'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="compact legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow p-1',
                                    src='assets/compact.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Compact Irregular'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="planar legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow p-1',
                                    src='assets/planar.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Planar Polycrystal'],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="rimed legend p-1",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='m-auto img-fluid card-img shadow p-1',
                                    src='assets/rimed.png',
                                ),
                            ),
                            dbc.Row(
                                className='justify-content-center text-center p-1 h4 text-uppercase',
                                children=['Rimed'],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
