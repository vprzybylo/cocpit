import dash_bootstrap_components as dbc
from dash import dcc, html
import globals
from datetime import date


def navbar_collapse():
    """navbar for selections, which collapses"""
    return dbc.Navbar(
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        dbc.NavbarToggler(
                            id="navbar-toggler", className="m-2"
                        ),
                    ),
                    dbc.Col(
                        html.P(
                            "All figures update based on filters"
                            " applied in the dropdown menu",
                            className="m-2 text-center",
                        ),
                    ),
                    dbc.Collapse(
                        children=[
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Campaign:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                dcc.Dropdown(
                                                    id="campaign-dropdown",
                                                    multi=False,
                                                    options=[
                                                        {
                                                            "label": i,
                                                            "value": i,
                                                        }
                                                        for i in globals.campaigns_rename
                                                    ],
                                                    placeholder="Campaign",
                                                    value="CRYSTAL FACE (UND)",
                                                    className="h4 dropdown",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Environmental Property:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                dcc.Dropdown(
                                                    id="env-dropdown",
                                                    options=[
                                                        {
                                                            "label": i,
                                                            "value": i,
                                                        }
                                                        for i in globals.env_properties
                                                    ],
                                                    placeholder=(
                                                        "Environmental"
                                                        " Variable"
                                                    ),
                                                    value="Ice Water Content [g/m3]",
                                                    className="h4 dropdown",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Particle Property:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                dcc.Dropdown(
                                                    id="property-dropdown",
                                                    options=[
                                                        {
                                                            "label": i,
                                                            "value": i,
                                                        }
                                                        for i in globals.particle_properties
                                                    ],
                                                    placeholder=(
                                                        "Particle Property"
                                                    ),
                                                    value="Complexity",
                                                    className="h4 dropdown",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Date Range:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                dcc.DatePickerRange(
                                                    id="date-picker",
                                                    start_date=date(
                                                        2002, 7, 13
                                                    ),
                                                    end_date=date(2002, 7, 23),
                                                    month_format="MMM Do, YY",
                                                    display_format=(
                                                        "MMM Do, YY"
                                                    ),
                                                    className="h4",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Temperature Range [C]: ",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dcc.Input(
                                                        type="text",
                                                        placeholder=(
                                                            "min [C],"
                                                            " e.g., -70"
                                                        ),
                                                        id="min-temp",
                                                        value=-70,
                                                        className=(
                                                            "m-1 dash-input"
                                                        ),
                                                    ),
                                                    dcc.Input(
                                                        type="text",
                                                        placeholder=(
                                                            "max [C], e.g., 20"
                                                        ),
                                                        id="max-temp",
                                                        value=40,
                                                        className=(
                                                            "m-1 dash-input"
                                                        ),
                                                    ),
                                                ],
                                                className=(
                                                    "justify-content-center"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Size Range [µm]:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                children=[
                                                    dcc.Input(
                                                        type="text",
                                                        placeholder=(
                                                            "min, e.g., 100"
                                                        ),
                                                        id="min-size",
                                                        value=30,
                                                        className=(
                                                            "m-1 dash-input"
                                                        ),
                                                    ),
                                                    dcc.Input(
                                                        type="text",
                                                        placeholder=(
                                                            "max, e.g., 2000"
                                                        ),
                                                        id="max-size",
                                                        value=3000,
                                                        className=(
                                                            "m-1 dash-input"
                                                        ),
                                                    ),
                                                ],
                                                className=(
                                                    "p-1 mx-auto"
                                                    " justify-content-center"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Pressure Range:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                [
                                                    html.P("Minimum:"),
                                                    html.Div(
                                                        dcc.RangeSlider(
                                                            id="min-pres",
                                                            min=100,
                                                            max=400,
                                                            value=[100],
                                                            allowCross=False,
                                                            marks={
                                                                400: {
                                                                    "label": "400hPa"
                                                                },
                                                                300: {
                                                                    "label": "300hPa"
                                                                },
                                                                200: {
                                                                    "label": "200hPa"
                                                                },
                                                                100: {
                                                                    "label": "100hPa"
                                                                },
                                                            },
                                                        ),
                                                    ),
                                                ],
                                                className=(
                                                    "justify-content-around"
                                                    " d-inline"
                                                ),
                                            ),
                                            dbc.Row(
                                                [
                                                    html.P("Maximum:"),
                                                    dcc.RangeSlider(
                                                        id="max-pres",
                                                        min=400,
                                                        max=1000,
                                                        value=[1000],
                                                        allowCross=False,
                                                        marks={
                                                            400: {
                                                                "label": (
                                                                    "400hPa"
                                                                )
                                                            },
                                                            600: {
                                                                "label": (
                                                                    "600hPa"
                                                                )
                                                            },
                                                            800: {
                                                                "label": (
                                                                    "800hPa"
                                                                )
                                                            },
                                                            1000: {
                                                                "label": (
                                                                    "1000hPa"
                                                                )
                                                            },
                                                        },
                                                    ),
                                                ],
                                                className=(
                                                    "justify-content-around"
                                                    " d-inline"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-10 col-md-4"
                                            " col-lg-3 col-xl-3"
                                        ),
                                    ),
                                    dbc.Col(
                                        children=[
                                            dbc.Row(
                                                dbc.Label(
                                                    "Particle Type:",
                                                    className="h4 fw-bolder",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                            dbc.Row(
                                                dcc.Dropdown(
                                                    id="part-type-dropdown",
                                                    multi=True,
                                                    options=[
                                                        {
                                                            "label": i,
                                                            "value": i,
                                                        }
                                                        for i in globals.particle_types_rename
                                                    ],
                                                    placeholder=(
                                                        "Particle Type"
                                                    ),
                                                    value=globals.particle_types_rename,
                                                    className="h4",
                                                ),
                                                className=(
                                                    "justify-content-around"
                                                ),
                                            ),
                                        ],
                                        className=(
                                            "m-1 col-xs-10 col-sm-6 col-md-7"
                                            " col-lg-7 col-xl-7"
                                        ),
                                    ),
                                ],
                                className="justify-content-around",
                            ),
                            dbc.Row(
                                children=[
                                    dbc.Col(
                                        html.Button(
                                            id="submit-button",
                                            n_clicks=0,
                                            children="Apply Filters",
                                            className=(
                                                "btn btn-secondary white"
                                                " btn-lg m-auto"
                                            ),
                                        ),
                                        className="m-2",
                                    ),
                                    dcc.Download(id="download-df-csv"),
                                    dbc.Col(
                                        html.Button(
                                            id="download-button",
                                            n_clicks=0,
                                            children="Download Data",
                                            className=(
                                                "btn btn-secondary white"
                                                " btn-lg m-auto"
                                            ),
                                        ),
                                        className="m-2",
                                    ),
                                ],
                                className="justify-content-center",
                            ),
                        ],
                        is_open=False,
                        id="navbar-collapse",
                        navbar=False,
                    ),
                ],
            ),
        ],
        sticky="top",
    )
