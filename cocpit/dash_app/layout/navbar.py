import dash_bootstrap_components as dbc
from layout import sidebar
from dash import html


def navbar():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Container(
                    className='sticky-top',
                    children=[
                        dbc.Row(
                            html.H1(
                                html.A(
                                    "COCPIT",
                                    href="http://www.specinc.com/cloud-particle-imager",
                                    className='text-body my-2',
                                ),
                            ),
                        ),
                        dbc.Row(
                            html.Div(
                                "Classification of Ice Particle Imagery and Thermodynamics",
                                className='h3 mb-3',
                            ),
                        ),
                        dbc.NavbarToggler(id="navbar-toggler"),
                        dbc.Collapse(
                            dbc.Nav([], className='work-sans', navbar=True),
                            id="navbar-collapse",
                            navbar=True,
                            is_open=True,
                        ),
                    ],
                ),
            ],
        ),
        className='navbar-change',
        expand='lg',
    )
    return navbar
