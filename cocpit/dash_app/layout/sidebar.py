import dash_bootstrap_components as dbc
from dash import dcc, html
import globals
from datetime import date


def sidebar():

    padding = '1px'
    margin_bottom = '8px'

    # styling the sidebar
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        'overflow-y': 'scroll',
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }

    sidebar = html.Div(
        [
            html.H1(
                html.A(
                    "COCPIT",
                    href="http://www.specinc.com/cloud-particle-imager",
                    style={'margin': "0px"},
                    className='lead display-6 text-body text-decoration-none',
                ),
            ),
            html.Hr(),
            html.H5(
                "Classification of Ice Particle Imagery and Thermodynamics",
            ),
            html.Hr(),
            # html.H6(
            #     'Images taken from the:',
            #     style={'margin': "0px"},
            # ),
            # html.A(
            #     "Cloud Particle Imager",
            #     href="http://www.specinc.com/cloud-particle-imager",
            #     style={'margin': "0px"},
            # ),
            # html.Hr(),
            dbc.Row(
                dbc.Label('Campaign:'),
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='campaign-dropdown',
                    multi=False,
                    options=[
                        {'label': i, 'value': i} for i in globals.campaigns_rename
                    ],
                    placeholder="Campaign",
                    value='CRYSTAL FACE (UND)',
                ),
                style={"padding": padding, "margin-bottom": margin_bottom},
            ),
            dbc.Row(
                dbc.Label('Particle Property:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='property-dropdown',
                    options=[
                        {'label': i, 'value': i} for i in globals.particle_properties
                    ],
                    placeholder="Particle Property",
                    value='Complexity',
                ),
                style={"padding": padding, "margin-bottom": margin_bottom},
            ),
            dbc.Row(
                dbc.Label('Date:'),
                style={"padding": padding},
            ),
            dbc.Row(
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=date(2002, 7, 11),
                    end_date=date(2002, 7, 12),
                    month_format='MMM Do, YY',
                    display_format='MMM Do, YY',
                ),
                style={"padding": padding, "margin-bottom": margin_bottom},
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
                    #'width': '35%',
                    'margin-left': 6,
                    'margin-right': 1,
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
                    #'width': '35%',
                    'margin-left': 6,
                    'margin-right': 1,
                    "margin-bottom": margin_bottom,
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
                style={"padding": padding},
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
                style={"padding": padding, "margin-bottom": margin_bottom},
            ),
            dbc.Row(
                dbc.Label('Particle Size [micrometers]:'),
            ),
            dbc.Row(
                dcc.Input(
                    type='text',
                    placeholder='min, e.g., 100',
                    id='min-size',
                    value=30,
                ),
                style={
                    "padding": padding,
                    #'width': '35%',
                    'margin-left': 5,
                    'margin-right': 1,
                },
                align="center",
            ),
            dbc.Row(
                dcc.Input(
                    type='text',
                    placeholder='max, e.g., 2000',
                    id='max-size',
                    value=3000,
                ),
                style={
                    "padding": padding,
                    #'width': '35%',
                    'margin-left': 5,
                    'margin-right': 1,
                    "margin-bottom": margin_bottom,
                },
                align="center",
            ),
            dbc.Row(
                html.Button(
                    id='submit-button',
                    n_clicks=0,
                    children='Apply Filters',
                    className='btn btn-primary black-background white btn-lg ',
                ),
                style={
                    "padding": '4px',
                    'margin-left': 5,
                    'margin-right': 1,
                },
            ),
            dcc.Download(id="download-df-csv"),
            dbc.Row(
                html.Button(
                    id='download-button',
                    n_clicks=0,
                    children='Download Data',
                    className='btn btn-primary black-background white btn-lg ',
                ),
                style={
                    "padding": '4px',
                    'margin-left': 5,
                    'margin-right': 1,
                },
            ),
        ],
        style=SIDEBAR_STYLE,
    )

    return sidebar
