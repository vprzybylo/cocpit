import dash_bootstrap_components as dbc
from dash import dcc, html
import globals
from datetime import date


def sidebar():

    padding = '1px'
    margin_bottom = '8px'

    sidebar = html.Div(
        [
            html.H1(
                html.A(
                    "COCPIT",
                    href="http://www.specinc.com/cloud-particle-imager",
                    style={'margin': "0px"},
                    className='h1 text-body',
                ),
            ),
            html.Hr(),
            html.H6(
                "Classification of Ice Particle Imagery and Thermodynamics",
            ),
            html.Hr(),
            dbc.Row(
                dbc.Label('Campaign:', className='label'),
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
            ),
            dbc.Row(
                dbc.Label('Particle Type:', className='label'),
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='part-type-dropdown',
                    multi=True,
                    options=[
                        {'label': i, 'value': i} for i in globals.particle_types_rename
                    ],
                    placeholder="Particle Type",
                    value=globals.particle_types_rename,
                ),
            ),
            dbc.Row(
                dbc.Label('Particle Property:', className='label'),
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
            ),
            dbc.Row(
                dbc.Label('Environmental Variable:', className='label'),
            ),
            dbc.Row(
                dcc.Dropdown(
                    id='env-dropdown',
                    options=[{'label': i, 'value': i} for i in globals.env_properties],
                    placeholder="Environmental Variable",
                    value='Ice Water Content',
                ),
            ),
            dbc.Row(
                dbc.Label('Date:', className='label'),
            ),
            dbc.Row(
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date=date(2002, 7, 11),
                    end_date=date(2002, 7, 12),
                    month_format='MMM Do, YY',
                    display_format='MMM Do, YY',
                ),
            ),
            dbc.Row(
                dbc.Label('Temperature Range [C]:', className='label'),
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
                dbc.Label('Pressure Minimum:', className='label'),
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
            ),
            dbc.Row(
                dbc.Label('Particle Size [micrometers]:', className='label'),
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
                    className='btn btn-primary white btn-lg',
                ),
            ),
            dcc.Download(id="download-df-csv"),
            dbc.Row(
                html.Button(
                    id='download-button',
                    n_clicks=0,
                    children='Download Data',
                    className='btn btn-primary white btn-lg',
                ),
            ),
        ],
        id='sidebar',
    )

    return sidebar
