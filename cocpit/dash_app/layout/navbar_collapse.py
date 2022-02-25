from dash import html
import dash_bootstrap_components as dbc


def navbar():
    return dbc.Container(
        [
            dbc.NavbarToggler(id="navbar-toggler2", className='navbar-toggler-icon'),
            dbc.Collapse(
                dbc.Nav(
                    [
                        dbc.NavItem(dbc.NavLink("Page1", href="#", n_clicks=0)),
                        dbc.NavItem(dbc.NavLink("Page2", href="#")),
                        dbc.NavItem(dbc.NavLink("Page3", href="#")),
                    ],
                    className="ml-auto",
                    navbar=True,
                    pills=False,
                    horizontal="center",
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    )
