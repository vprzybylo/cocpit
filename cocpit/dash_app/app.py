'''main dashboard executable'''

import dash_bootstrap_components as dbc
from callbacks import environment, geometric, process, topographic
from dash import dcc
from dotenv import load_dotenv
from layout import content, navbar, sidebar, navbar_collapse
from dash_extensions.enrich import Dash


def main():
    load_dotenv()

    return Dash(
        __name__,
        external_scripts=[
            'assets/sidebar.js',
            {
                'src': 'https://use.fontawesome.com/releases/v5.0.13/js/fontawesome.js',
                'integrity': 'sha384-6OIrr52G08NpOFSZdxxz1xdNSndlD4vdcf/q2myIUVO0VsqaGHJsB0RaBE01VTOY',
                'crossorigin': 'anonymous',
            },
            {
                'src': 'https://code.jquery.com/jquery-3.3.1.slim.min.js',
                'integrity': 'sha384-6OIrr52G08NpOFSZdxxz1xdNSndlD4vdcf/q2myIUVO0VsqaGHJsB0RaBE01VTOY',
                'crossorigin': 'anonymous',
            },
        ],
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            'assets/main.css',
            'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
        ],
        meta_tags=[
            {
                'name': 'viewport',
                'content': 'width=device-width, initial-scale=1',
            }
        ],
    )


# Run local server
if __name__ == '__main__':

    app = main()
    app.title = 'COCPIT'
    app.layout = dbc.Container(
        [navbar_collapse.navbar()],
        fluid=True,
    )

    # process.register(app)
    # topographic.register(app)
    # environment.register(app)
    # geometric.register(app)
    app.run_server(port=8050, host='0.0.0.0', debug=True)
