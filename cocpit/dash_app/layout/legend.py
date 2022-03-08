import dash_bootstrap_components as dbc
from dash import html
import globals


def legend():
    return html.Div(
        children=[
            dbc.CardGroup(
                className="justify-content-around",
                children=[
                    dbc.Col(
                        className="agg legend p-1 d-flex",
                        children=[
                            dbc.Row(
                                html.Img(
                                    className='my-auto img-fluid card-img shadow p-1',
                                    src='assets/agg.png',
                                ),
                            ),
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Aggregate', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d' % globals.part_type_count['agg'],
                                            id='agg-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Budding Rosette', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d'
                                            % globals.part_type_count['budding'],
                                            id='budding-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Bullet Rosette', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d'
                                            % globals.part_type_count['bullet'],
                                            id='bullet-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Column', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d'
                                            % globals.part_type_count['column'],
                                            id='column-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Compact Irregular', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d'
                                            % globals.part_type_count['compact'],
                                            id='compact-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4(
                                            'Planar Polycrystal',
                                            className='m-1',
                                        ),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d'
                                            % globals.part_type_count['planar'],
                                            id='planar-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
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
                            dbc.Col(
                                className='py-2 h3 text-white',
                                children=[
                                    dbc.Row(
                                        html.H4('Rimed', className='m-1'),
                                    ),
                                    dbc.Row(
                                        html.H4(
                                            'n = %d' % globals.part_type_count['rimed'],
                                            id='rimed-count',
                                            className='m-1 mx-auto fw-normal',
                                        ),
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
