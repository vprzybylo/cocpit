'''main dashboard executable'''

import dash_bootstrap_components as dbc
from callbacks import environment, geometric, topographic, navbar
from processing_scripts import process
from dash import dcc
from dotenv import load_dotenv
from layout import content, header, legend, banners, sidebar, navbar_collapse
from dash_extensions.enrich import Dash


def main():
    load_dotenv()

    return Dash(
        __name__,
        external_stylesheets=[
            # dbc.themes.BOOTSTRAP,
            'https://use.fontawesome.com/releases/v5.8.1/css/all.css',
            'assets/main.css',
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
        [
            # sidebar.sidebar(),
            header.header(),
            navbar_collapse.navbar_collapse(),
            banners.banners(),
            legend.legend(),
            content.content(),
        ],
        fluid=True,
    )

    navbar.register(app)
    topographic.register(app)
    environment.register(app)
    geometric.register(app)
    app.run_server(port=8050, host='0.0.0.0', debug=True)
