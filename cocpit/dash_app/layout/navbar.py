from layout import sidebar
import dash_bootstrap_components as dbc
from dash import html
import globals

# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

items = [dbc.DropdownMenuItem(i) for i in globals.campaigns_rename]
# make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        items
        # dbc.DropdownMenuItem(
        #     id='campaign-dropdown',
        #     multi=False,
        #     options=[{'label': i, 'value': i} for i in globals.campaigns_rename],
        #     placeholder="Campaign",
        #     value='CRYSTAL FACE (UND)',
        #     className='h4',
        # ),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)


def navbar():

    navbar = html.Div(
        id='navbar',
        children=[
            dbc.Row(
                children=[
                    dbc.Col(
                        html.H3(
                            html.A(
                                "COCPIT",
                                href="http://www.specinc.com/cloud-particle-imager",
                                className='text-white px-2',
                            ),
                        ),
                    ),
                    dbc.Col(
                        html.H4(
                            "Classification of Ice Particle Imagery and Thermodynamics",
                            className='text-white px-2',
                        ),
                    ),
                ],
                className='d-flex justify-content-between align-items-center',
            ),
            dbc.Navbar(
                dbc.Row(
                    [
                        dbc.Col(
                            children=[
                                dbc.NavbarToggler(id="navbar-toggler1"),
                                dbc.Collapse(
                                    dbc.Nav(
                                        [nav_item, dropdown],
                                        className="ms-auto",
                                        navbar=True,
                                    ),
                                    id="navbar-collapse1",
                                    navbar=True,
                                ),
                            ],
                        ),
                        # dbc.Col(
                        #     children=[
                        #         dbc.Col(
                        #             dbc.Button(
                        #                 'Apply Filters',
                        #                 outline=True,
                        #                 className="mr-1 btn-secondary fas fa-bars",
                        #                 id="btn_sidebar",
                        #             ),
                        #             className="col-xs-12 col-sm-12 col-md-4 col-lg-4 col-xl-4 mx-4",
                        #         ),
                        #     ],
                        # ),
                    ],
                    className='justify-content-between align-items-center',
                ),
            ),
        ],
    )
    return navbar
