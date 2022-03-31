import dash_bootstrap_components as dbc
from dash import html
import globals

# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

items = [dbc.DropdownMenuItem(i) for i in globals.campaigns_rename]
# make a reuseable dropdown for the different examples
campaign_dropdown = dbc.DropdownMenu(
    items,
    # dbc.DropdownMenuItem(
    #     id='campaign-dropdown',
    #     multi=False,
    #     options=[{'label': i, 'value': i} for i in globals.campaigns_rename],
    #     placeholder="Campaign",
    #     value='CRYSTAL FACE (UND)',
    #     className='h4',
    # ),
    nav=True,
    in_navbar=True,
    label="Campaign",
)


def header():
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
