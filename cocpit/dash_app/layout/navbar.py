import dash_bootstrap_components as dbc
from layout import sidebar


def navbar():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarToggler(id="navbar-toggler"),
                dbc.Collapse(
                    dbc.Nav(
                        [sidebar.sidebar()], className='ml-auto work-sans', navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                    is_open=True,
                ),
            ],
        ),
        className='navbar-change',
        expand='lg',
    )
    return navbar
