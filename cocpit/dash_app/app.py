'''main dashboard executable'''

import dash_bootstrap_components as dbc
from callbacks import environment, geometric, process, topographic
from dash import dcc
from dotenv import load_dotenv
from layout import content, sidebar, header_info
from dash_extensions.enrich import Dash


def main():
    load_dotenv()

    return Dash(
        __name__,
        external_stylesheets=[
            'https://raw.githubusercontent.com/StartBootstrap/startbootstrap-sb-admin-2/master/css/sb-admin-2.css'
        ],
        meta_tags=[
            {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'},
        ],
    )


# Run local server
if __name__ == '__main__':

    app = main()
    app.title = 'COCPIT'

    app.layout = dbc.Container(
        [
            dcc.Location(id="url"),
            content.content(),
            sidebar.sidebar(),
        ],
        # fluid=True,
    )

    process.register(app)
    topographic.register(app)
    environment.register(app)
    geometric.register(app)
    app.run_server(port=8050, host='0.0.0.0', debug=True)
