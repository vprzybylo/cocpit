import dash_bootstrap_components as dbc
from dash import dcc, html
import globals
from datetime import date


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
                    options=[{'label': i, 'value': i} for i in globals.campaigns],
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
                    options=[
                        {'label': i, 'value': i} for i in globals.particle_properties
                    ],
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