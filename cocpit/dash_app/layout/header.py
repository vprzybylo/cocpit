import dash_bootstrap_components as dbc
from dash import html
import globals


def header():
    """Top text banner for COCPIT and Classification of Ice Particle Imagery and Thermodynamics"""
    return html.Div(
        id='header',
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        html.H3(
                            html.A(
                                "COCPIT",
                                href="https://github.com/vprzybylo/cocpit",
                                className='text-white px-3',
                            ),
                        ),
                    ),
                    dbc.Col(
                        html.H4(
                            "Classification of Ice Particle Imagery and Thermodynamics",
                            className='text-white px-3 ',
                        ),
                    ),
                ],
                className='d-flex justify-content-between align-items-center',
            ),
        ],
    )
