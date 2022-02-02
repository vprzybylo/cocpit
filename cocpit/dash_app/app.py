'''main dashboard executable'''

# import dash
import dash_bootstrap_components as dbc
from layout import content, sidebar
from dotenv import load_dotenv
from dash import dcc
from callbacks import environment, geometric, process, topographic
from dash_extensions.enrich import Dash
from dash_extensions.enrich import FileSystemStore
from gevent.pywsgi import WSGIServer


def main():
    load_dotenv()

    return Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        meta_tags=[
            {'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}
        ],
    )


# Run local server
if __name__ == '__main__':
    app = main()

    app.layout = dbc.Container(
        [dcc.Location(id="url"), content.content(), sidebar.sidebar()],
        fluid=True,
    )

    process.register(app)
    topographic.register(app)
    environment.register(app)
    geometric.register(app)
    app.run_server(port=8050, debug=True)
