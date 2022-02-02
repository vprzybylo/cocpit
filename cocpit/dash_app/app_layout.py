'''Set up the app layout'''

from globals import *
import dash_bootstrap_components as dbc
from dash import html, dash_table
from dash import dcc
import dash_loading_spinners as dls
import process


def sidebar():

    padding = '1px'

    # styling the sidebar
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    sidebar = html.Div(
        [
            html.H2("COCPIT", className="display-4"),
            html.Hr(),
            html.P(
                "Classification of Ice Particle Imagery and Thermodynamics",
                className="lead",
            ),
            html.Hr(),
            html.P(
                'Images taken from the:',
                style={'margin': "0px"},
            ),
            html.A(
                "Cloud Particle Imager",
                href="http://www.specinc.com/cloud-particle-imager",
                style={'margin': "0px"},
            ),
            html.Hr(),
            dbc.Row(
                dbc.Label('Campaign:'),
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='campaign-dropdown',
                    multi=False,
                    options=[{'label': i, 'value': i} for i in campaigns],
                    placeholder="Campaign",
                    value='CRYSTAL_FACE_UND',
                ),
                style={"padding": padding, "margin-bottom": "12px"},
            ),
            dbc.Row(
                dbc.Label('Particle Property:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='property-dropdown',
                    options=[{'label': i, 'value': i} for i in particle_properties],
                    placeholder="Particle Property",
                    value='Complexity',
                ),
                style={"padding": padding, "margin-bottom": "12px"},
            ),
            dbc.Row(
                dbc.Label('Date:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=date(2002, 7, 3),
                    end_date=date(2002, 7, 29),
                    month_format='MMM Do, YY',
                    display_format='MMM Do, YY',
                ),
                style={"padding": padding, "margin-bottom": "12px"},
            ),
            dbc.Row(
                dbc.Label('Temperature Range [C]:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.Input(
                    type='text',
                    placeholder='min [C], e.g., -70',
                    id='min-temp',
                    value=-70,
                ),
                style={
                    "padding": padding,
                    'width': '65%',
                    'margin-left': 6,
                },
                align="center",
            ),
            dbc.Row(
                dcc.Input(
                    type='text',
                    placeholder='max [C], e.g., 20',
                    id='max-temp',
                    value=40,
                ),
                style={
                    "padding": padding,
                    'width': '65%',
                    'margin-left': 6,
                    "margin-bottom": "12px",
                },
                align="center",
            ),
            dbc.Row(
                dbc.Label('Pressure Maximum:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.RangeSlider(
                    id='max-pres',
                    min=400,
                    max=1000,
                    value=[1000],
                    allowCross=False,
                    marks={
                        400: {'label': '400hPa'},
                        600: {'label': '600hPa'},
                        800: {'label': '800hPa'},
                        1000: {'label': '1000hPa'},
                    },
                ),
                style={"padding": padding, "margin-bottom": "12px"},
            ),
            dbc.Row(
                dbc.Label('Pressure Minimum:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.RangeSlider(
                    id='min-pres',
                    min=100,
                    max=400,
                    value=[100],
                    allowCross=False,
                    marks={
                        400: {'label': '400hPa'},
                        300: {'label': '300hPa'},
                        200: {'label': '200hPa'},
                        100: {'label': '100hPa'},
                    },
                ),
                style={"padding": padding, "margin-bottom": "12px"},
            ),
            dbc.Row(
                dbc.Label('Particle Size:'),
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return sidebar


def content():

    # padding for the page content
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "margin-top": "4rem",
    }
    df = process.read_campaign('CRYSTAL_FACE_UND')

    return html.Div(
        id="page-content",
        children=[
            dcc.Store(id='store-df'),
            dls.Hash(
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(
                                    dbc.Label('Particle Type'),
                                ),
                                dbc.Row(
                                    dcc.Checklist(
                                        id="map-particle_type",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in particle_types_rename
                                        ],
                                        value=["aggregate"],
                                        inputStyle={'margin-right': "5px"},
                                        labelStyle={
                                            'display': 'block',
                                        },
                                        style={
                                            'width': "120px",
                                            "overflow": "auto",
                                        },
                                    ),
                                ),
                            ],
                            xs=12,
                            sm=12,
                            md=12,
                            lg=12,
                            xl=2,
                        ),
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
                        #                     for i in vertical_vars
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
                        dbc.Col(
                            dcc.Graph(id='flat-topo', figure={}),
                            xs=12,
                            sm=12,
                            md=12,
                            lg=12,
                            xl=10,
                        ),
                    ],
                    align="center",
                    justify="center",
                )
            ),
            html.Hr(),
            dls.Hash(
                [
                    dbc.Row(
                        html.H4(
                            'Geographic Attributes',
                            className='text-center',
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id='lon-alt-hist', figure={}),
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
                            ),
                            dbc.Col(
                                dcc.Graph(id='lat-alt-hist', figure={}),
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
            html.Hr(),
            dls.Hash(
                [
                    dbc.Row(
                        html.H4(
                            'Environmental Attributes',
                            className='text-center',
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id='type-temp-violin', figure={}),
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
                            ),
                            dbc.Col(
                                dcc.Graph(id='type-iwc-violin', figure={}),
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
            html.Hr(),
            dls.Hash(
                [
                    dbc.Row(
                        html.H4(
                            'Geometric Attributes',
                            className='text-center',
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Graph(id='pie', figure={}),
                                xs=12,
                                sm=12,
                                md=12,
                                lg=12,
                                xl=6,
                            ),
                            dbc.Col(
                                dcc.Graph(id='prop_fig', figure={}),
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
            dls.Hash(
                dbc.Col(
                    dbc.Row(
                        dash_table.DataTable(
                            id="table",
                            columns=[{"name": i, "id": i} for i in df.columns],
                            data=df.to_dict("records"),
                            export_format="csv",
                            # page_size=10,
                            fixed_rows={'headers': True},
                            style_table={'height': '300px', 'overflowY': 'auto'},
                        )
                    ),
                    xs=12,
                    sm=12,
                    md=12,
                    lg=12,
                    xl=12,
                )
            ),
            #            html.Hr(),
            #            dls.Hash(
            # dbc.Row(
            #     [
            #         dbc.Col(
            #             [
            #                 dbc.Row(
            #                     dbc.Label('Particle Type'),
            #                 ),
            #                 dbc.Row(
            #                     dcc.Checklist(
            #                         id="map-particle_type",
            #                         options=[
            #                             {"label": i, "value": i}
            #                             for i in particle_types_rename
            #                         ],
            #                         value=["aggregate"],
            #                         inputStyle={'margin-right': "5px"},
            #                         labelStyle={
            #                             'display': 'block',
            #                         },
            #                         style={
            #                             'width': "120px",
            #                             "overflow": "auto",
            #                         },
            #                     ),
            #                 ),
            #             ],
            #             xs=12,
            #             sm=12,
            #             md=12,
            #             lg=12,
            #             xl=1,
            #         ),
            #         dbc.Col(
            #             dcc.Graph(id='top-down-map', figure={}),
            #             xs=12,
            #             sm=12,
            #             md=12,
            #             lg=12,
            #             xl=5,
            #         ),
            #         dbc.Col(
            #             [
            #                 dbc.Row(
            #                     dbc.Label('Vertical Axis Property:'),
            #                 ),
            #                 dbc.Row(
            #                     dcc.Dropdown(
            #                         id='3d_vertical_prop',
            #                         options=[
            #                             {'label': i, 'value': i}
            #                             for i in vertical_vars
            #                         ],
            #                         placeholder="Vertical Axis Property",
            #                         value='Temperature',
            #                     ),
            #                 ),
            #             ],
            #             xs=12,
            #             sm=12,
            #             md=12,
            #             lg=12,
            #             xl=1,
            #         ),
            #         # dbc.Col(
            #         #     dcc.Graph(id='3d map', figure={}),
            #         #     xs=12,
            #         #     sm=12,
            #         #     md=12,
            #         #     lg=12,
            #         #     xl=5,
            #         # ),
            #     ],
            #     className="g-0",
            #     align="center",
            #     justify="center",
            # )
            #            ),
            #            html.Hr(),
            # dls.Hash(
            #     dbc.Row(
            #         dbc.Col(
            #             dcc.Graph(id='globe', figure={}),
            #             xs=12,
            #             sm=12,
            #             md=12,
            #             lg=12,
            #             xl=5,
            #         ),
            #     )
            # ),
            html.Hr(),
            html.P(
                'Copyright All Rights Reserved',
                style={"color": '#D3D3D3', 'text-align': 'center'},
            ),
        ],
        style=CONTENT_STYLE,
    )
