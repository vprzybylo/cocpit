from dash import html
import dash_bootstrap_components as dbc


def navbar():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                [
                                    html.Div(className='icon-bar'),
                                    html.Div(className='icon-bar'),
                                    html.Div(className='icon-bar'),
                                ],
                                className='navbar-toggle',
                                type='button',
                                **{
                                    'data-toggle': 'collapse',
                                    'data-target': '.sidebar-navbar-collapse',
                                }
                            ),
                            html.Div(
                                html.H2(
                                    html.A(
                                        "COCPIT",
                                        href="http://www.specinc.com/cloud-particle-imager",
                                        className='text-white navbar-brand',
                                    ),
                                ),
                                className='visible-xs',
                            ),
                        ],
                        className='navbar-header',
                    ),
                    html.Div(
                        [
                            html.Ul(
                                [
                                    html.Li(
                                        html.H2(
                                            html.A(
                                                "COCPIT",
                                                href="http://www.specinc.com/cloud-particle-imager",
                                                className='text-white navbar-brand',
                                            ),
                                        ),
                                        className='d-none d-sm-block',
                                    ),
                                    html.Li(
                                        [
                                            html.I(
                                                className='glyphicon glyphicon-th-list'
                                            ),
                                            html.A(
                                                "Campaign",
                                                href="#",
                                                className='dropdown-toggle',
                                                **{
                                                    'data-toggle': 'dropdown',
                                                    'data-target': '.sidebar-navbar-collapse',
                                                }
                                            ),
                                            html.B(className='caret'),
                                        ],
                                        className='dropdown',
                                    ),
                                    html.Li("Item 3"),
                                ],
                                className='nav navbar-nav',
                            )
                        ],
                        className='navbar-collapse collapse sidebar-navbar-collapse',
                    ),
                ],
                id="navbar-red",
                className="navbar navbar-default",
            ),
        ],
        id="column-red",
        className="column-nav",
    )
