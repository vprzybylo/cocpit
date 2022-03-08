import dash_bootstrap_components as dbc


# make a reuseable navitem for the different examples
nav_item = dbc.NavItem(dbc.NavLink("Link", href="#"))

# make a reuseable dropdown for the different examples
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Entry 1"),
        dbc.DropdownMenuItem("Entry 2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Entry 3"),
    ],
    nav=True,
    in_navbar=True,
    label="Menu",
)


def filters():
    return dbc.Row(
        children=[
            dbc.Navbar(
                children=[
                    dbc.NavbarBrand("Custom default", href="#"),
                    dbc.NavbarToggler(id="navbar-toggler1"),
                    dbc.Nav([nav_item, dropdown], className="ms-auto", navbar=True),
                ],
                id="navbar-collapse1",
            )
        ],
        className="mt-auto col-xs-12 col-sm-12 col-md-8 col-lg-3 col-xl-3",
        id='filters',
    )
