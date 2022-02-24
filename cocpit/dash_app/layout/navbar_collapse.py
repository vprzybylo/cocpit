from dash import html
import dash_bootstrap_components as dbc


def navbar():
    return html.Ul(
        [
            html.Li(
                [
                    html.Div(
                        [
                            html.A(
                                'COCPIT',
                                href='#',
                                id='title',
                                className='text-white text-decoration-none',
                            ),
                            html.A(
                                href='#',
                                id='sidebarToggleHolder',
                                className='text-white float-right',
                                children=[
                                    html.Div(
                                        className='fas fa-bars', id='sidebarToggle'
                                    )
                                ],
                            ),
                        ],
                        className='text-center text-white logo py-4 mx-4',
                    )
                ],
                className='nav-item logo-holder',
            )
        ],
        className='nav flex-column shadow d-flex sidebar',
    )
