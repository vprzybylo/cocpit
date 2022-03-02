from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, html
import globals
from datetime import date


def navbar_collapse():
    return dbc.Navbar(
        id='navbar_collapse',
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        dbc.NavbarToggler(id="navbar-toggler", className='m-2'),
                    ),
                    dbc.Collapse(
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        dbc.Row(
                                            dbc.Label(
                                                'Campaign:', className='label h4'
                                            ),
                                            className='justify-content-around',
                                        ),
                                        dbc.Row(
                                            dcc.Dropdown(
                                                id='campaign-dropdown',
                                                multi=False,
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in globals.campaigns_rename
                                                ],
                                                placeholder="Campaign",
                                                value='CRYSTAL FACE (UND)',
                                                className='h4',
                                            ),
                                            className='d-inline',
                                        ),
                                    ],
                                    className='m-1 col-xs-10 col-sm-4 col-md-4 col-lg-4 col-xl-4',
                                ),
                                dbc.Col(
                                    children=[
                                        dbc.Row(
                                            dbc.Label(
                                                'Particle Type:', className='label h4'
                                            ),
                                            className='justify-content-around',
                                        ),
                                        dbc.Row(
                                            dcc.Dropdown(
                                                id='part-type-dropdown',
                                                multi=True,
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in globals.particle_types_rename
                                                ],
                                                placeholder="Particle Type",
                                                value=globals.particle_types_rename,
                                                className='h4',
                                            ),
                                            className='d-inline',
                                        ),
                                    ],
                                    className='m-1 col-xs-10 col-sm-4 col-md-4 col-lg-4 col-xl-4',
                                ),
                                dbc.Col(
                                    children=[
                                        dbc.Row(
                                            dbc.Label(
                                                'Particle Property:',
                                                className='label h4',
                                            ),
                                            className='justify-content-around',
                                        ),
                                        dbc.Row(
                                            dcc.Dropdown(
                                                id='property-dropdown',
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in globals.particle_properties
                                                ],
                                                placeholder="Particle Property",
                                                value='Complexity',
                                                className='h4',
                                            ),
                                            className='d-inline',
                                        ),
                                    ],
                                    className='m-1 col-xs-10 col-sm-4 col-md-4 col-lg-4 col-xl-4',
                                ),
                                dbc.Col(
                                    children=[
                                        dbc.Row(
                                            dbc.Label(
                                                'Environmental Property:',
                                                className='label h4',
                                            ),
                                            className='justify-content-around',
                                        ),
                                        dbc.Row(
                                            dcc.Dropdown(
                                                id='env-dropdown',
                                                options=[
                                                    {'label': i, 'value': i}
                                                    for i in globals.env_properties
                                                ],
                                                placeholder="Environmental Variable",
                                                value='Ice Water Content',
                                                className='h4',
                                            ),
                                            className='d-inline',
                                        ),
                                    ],
                                    className='m-1 col-xs-10 col-sm-4 col-md-4 col-lg-4 col-xl-4',
                                ),
                                # dbc.Col(
                                #     dcc.DatePickerRange(
                                #         id='date-picker',
                                #         start_date=date(2002, 7, 19),
                                #         end_date=date(2002, 7, 23),
                                #         month_format='MMM Do, YY',
                                #         display_format='MMM Do, YY',
                                #     ),
                                #     className='d-flex m-2 h4',
                                # ),
                                # dbc.Col(
                                #     dcc.Input(
                                #         type='text',
                                #         placeholder='min [C], e.g., -70',
                                #         id='min-temp',
                                #         value=-70,
                                #     ),
                                #     className='p-1',
                                #     align="center",
                                # ),
                                # dbc.Col(
                                #     dcc.Input(
                                #         type='text',
                                #         placeholder='max [C], e.g., 20',
                                #         id='max-temp',
                                #         value=40,
                                #     ),
                                #     className='p-1',
                                #     align="center",
                                # ),
                                # dbc.Col(
                                #     children=[
                                #         html.Div(
                                #             dcc.RangeSlider(
                                #                 id='max-pres',
                                #                 min=400,
                                #                 max=1000,
                                #                 value=[1000],
                                #                 allowCross=False,
                                #                 marks={
                                #                     400: {'label': '400hPa'},
                                #                     600: {'label': '600hPa'},
                                #                     800: {'label': '800hPa'},
                                #                     1000: {'label': '1000hPa'},
                                #                 },
                                #             ),
                                #             className='p-1 col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3',
                                #         ),
                                #         html.Div(
                                #             dcc.RangeSlider(
                                #                 id='min-pres',
                                #                 min=100,
                                #                 max=400,
                                #                 value=[100],
                                #                 allowCross=False,
                                #                 marks={
                                #                     400: {'label': '400hPa'},
                                #                     300: {'label': '300hPa'},
                                #                     200: {'label': '200hPa'},
                                #                     100: {'label': '100hPa'},
                                #                 },
                                #             ),
                                #             className='p-1 col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3',
                                #         ),
                                #     ],
                                #     className='m-2 justify-content-around',
                                # ),
                                # dbc.Col(
                                #     children=[
                                #         dbc.Row(
                                #             dcc.Input(
                                #                 type='text',
                                #                 placeholder='min, e.g., 100',
                                #                 id='min-size',
                                #                 value=30,
                                #             ),
                                #             className='p-1',
                                #             align="center",
                                #         ),
                                #         dbc.Row(
                                #             dcc.Input(
                                #                 type='text',
                                #                 placeholder='max, e.g., 2000',
                                #                 id='max-size',
                                #                 value=3000,
                                #             ),
                                #             className='p-1',
                                #             align="center",
                                #         ),
                                #     ],
                                #     className='d-flex m-2  justify-content-around',
                                # ),
                                # dbc.Col(
                                #     html.Button(
                                #         id='submit-button',
                                #         n_clicks=0,
                                #         children='Apply Filters',
                                #         className='btn btn-secondary white btn-lg m-auto my-1',
                                #     ),
                                #     className='d-flex m-2  justify-content-around',
                                # ),
                                # dcc.Download(id="download-df-csv"),
                                # dbc.Col(
                                #     html.Button(
                                #         id='download-button',
                                #         n_clicks=0,
                                #         children='Download Data',
                                #         className='btn btn-secondary white btn-lg m-auto my-1',
                                #     ),
                                #     className='d-flex m-2  justify-content-around',
                                # ),
                            ],
                            className='justify-content-around',
                        ),
                        is_open=False,
                        id="navbar-collapse",
                        navbar=False,
                    ),
                ],
            ),
        ],
        sticky="top",
    )
