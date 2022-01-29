'''main dashboard executable'''

import dash
import dash_bootstrap_components as dbc
import app_layout
from dotenv import load_dotenv
from dash import dcc
import callbacks


def main():
    load_dotenv()
    return dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        meta_tags=[
            {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
        ],
    )


# Run local server
if __name__ == '__main__':
    app = main()
    sidebar = app_layout.sidebar()
    content = app_layout.content()
    app.layout = dbc.Container(
        [dcc.Location(id="url"), sidebar, content],
        fluid=True,
    )
    callbacks.register_callbacks(app)
    app.run_server(port=8050, debug=True)
