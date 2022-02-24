import dash_bootstrap_components as dbc
from layout import sidebar
from dash import html

# the style arguments for the sidebar. We use position:fixed and a fixed width


SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}


def navbar():

    navbar = html.Div(
        id='navbar',
        className='fixed-top',
        children=[
            # html.Div(id='black-nav-div'),
            dbc.Row(
                children=[
                    dbc.Col(
                        children=[
                            dbc.Col(
                                dbc.Button(
                                    "Filters",
                                    outline=True,
                                    className="mr-1 btn-secondary",
                                    id="btn_sidebar",
                                ),
                                className="col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3 mx-3",
                            ),
                        ],
                        className='my-2',
                    ),
                    dbc.Col(
                        children=[
                            dbc.Row(
                                html.H2(
                                    html.A(
                                        "COCPIT",
                                        href="http://www.specinc.com/cloud-particle-imager",
                                        className='text-white',
                                    ),
                                ),
                            ),
                            dbc.Row(
                                html.Div(
                                    "Classification of Ice Particle Imagery and Thermodynamics",
                                    className='h3 mb-3 text-white',
                                ),
                            ),
                        ],
                        className="col-xs-12 col-sm-12 col-md-3 col-lg-3 col-xl-3 my-2",
                    ),
                ],
                justify='between',
                align="center",
            ),
        ],
    )
    return navbar
