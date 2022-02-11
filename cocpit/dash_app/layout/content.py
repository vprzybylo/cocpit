'''Set up the app layout'''

import dash_bootstrap_components as dbc
import dash_loading_spinners as dls
import globals
from dash import dcc, html


def content():

    # padding for the page content
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "margin-top": "2rem",
    }
    return html.Div(
        id="page-content",
        children=[
            dcc.Store(id='store-df', storage_type='session'),
            dcc.Store(id='df-classification', storage_type='session'),
            dcc.Store(id='df-lat', storage_type='session'),
            dcc.Store(id='df-lon', storage_type='session'),
            dcc.Store(id='df-alt', storage_type='session'),
            dcc.Store(id='df-iwc', storage_type='session'),
            dcc.Store(id='df-temp', storage_type='session'),
            dcc.Store(id='df-prop', storage_type='session'),
            dcc.Store(id='len-df', storage_type='session'),
            dls.Hash(
                dbc.Row(
                    [
                        # dbc.Col(
                        #    [
                        #     dbc.Row(
                        #         dbc.Label('Particle Type'),
                        #     ),
                        #     dbc.Row(
                        #         dcc.Checklist(
                        #             id="topo-map-particle_type",
                        #             options=[
                        #                 {"label": i, "value": i}
                        #                 for i in globals.particle_types_rename
                        #             ],
                        #             value=["aggregate"],
                        #             inputStyle={'margin-right': "5px"},
                        #             labelStyle={
                        #                 'display': 'block',
                        #             },
                        #             style={
                        #                 'width': "120px",
                        #                 "overflow": "auto",
                        #             },
                        #         ),
                        #     ),
                        # ],
                        # xs=12,
                        # sm=12,
                        # md=12,
                        # lg=12,
                        # xl=2,
                        # ),
                        # dbc.Col(
                        #     [
                        #         dbc.Row(
                        #             dbc.Label('Vertical Axis Property:'),
                        #         ),
                        #         dbc.Row(
                        #             dcc.Dropdown(
                        #                 id='3d_vertical_prop',
                        #                 options=[
                        #                     {'label': i, 'value': i}
                        #                     for i in globals.vertical_vars
                        #                 ],
                        #                 placeholder="Vertical Axis Property",
                        #                 value='Temperature',
                        #             ),
                        #         ),
                        #     ],
                        #     xs=12,
                        #     sm=12,
                        #     md=12,
                        #     lg=12,
                        #     xl=2,
                        # ),
                        # dbc.Col(
                        #     dcc.Graph(id='top-down-map', figure={}),
                        #     xs=12,
                        #     sm=12,
                        #     md=12,
                        #     lg=6,
                        #     xl=6,
                        # ),
                        # dbc.Col(
                        #     dcc.Graph(id='pie', figure={}),
                        #     xs=12,
                        #     sm=12,
                        #     md=12,
                        #     lg=6,
                        #     xl=6,
                        # ),
                        dbc.Col(
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Particle Location"),
                                        dbc.CardBody(
                                            children=[
                                                dcc.Graph(id='top-down-map', figure={}),
                                            ]
                                        ),
                                    ],
                                )
                            ],
                            xs=12,
                            sm=12,
                            md=12,
                            lg=6,
                            xl=6,
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
                                )
                            ],
                            xs=12,
                            sm=12,
                            md=12,
                            lg=6,
                            xl=6,
                        ),
                    ],
                    align="center",
                    justify="center",
                ),
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
                                    )
                                ],
                                xs=12,
                                sm=12,
                                md=12,
                                lg=6,
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
                                    )
                                ],
                                xs=12,
                                sm=12,
                                md=12,
                                lg=6,
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
                                lg=6,
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
                                                        id='type-iwc-violin', figure={}
                                                    )
                                                ]
                                            ),
                                        ],
                                        style={"margin": "0px"},
                                    )
                                ],
                                xs=12,
                                sm=12,
                                md=12,
                                lg=6,
                                xl=6,
                            ),
                        ],
                        align="center",
                        justify="center",
                    ),
                ]
            ),
            # dls.Hash(
            #     dbc.Col(
            #         dbc.Row(
            #             dash_table.DataTable(
            #                 id="table",
            #                 columns=[{"name": i, "id": i} for i in df.columns],
            #                 data=df.to_dict("records"),
            #                 # export_columns='all',
            #                 export_headers='display',
            #                 export_format="csv",
            #                 fixed_rows={'headers': True},
            #                 style_table={'height': '300px', 'overflowY': 'auto'},
            #             )
            #         ),
            #         xs=12,
            #         sm=12,
            #         md=12,
            #         lg=12,
            #         xl=12,
            #     )
            # ),
            html.P(
                'Copyright All Rights Reserved',
                style={"color": '#D3D3D3', 'text-align': 'center'},
            ),
        ],
        style=CONTENT_STYLE,
    )
