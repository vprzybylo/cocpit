from datetime import date

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import cocpit.config as config

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(external_stylesheets=external_stylesheets)

campaigns = [
    "AIRS_II",
    "ARM",
    "ATTREX",
    "CRYSTAL_FACE_UND",
    "CRYSTAL_FACE_NASA",
    "ICE_L",
    "IPHEX",
    "ISDAC",
    "MACPEX",
    "MC3E",
    "MIDCIX",
    "MPACE",
    "POSIDON",
    "OLYMPEX",
]

particle_types = [
    "agg",
    "budding",
    "bullet",
    "column",
    "compact_irreg",
    "fragment",
    "planar_polycrystal",
    "rimed",
    "sphere",
]

# Set up the app layout
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    'Classification of Cloud Particle Imagery and Thermodynamics (COCPIT)'
                ),
                dcc.Dropdown(
                    id='campaign-dropdown',
                    options=[{'label': i, 'value': i} for i in campaigns],
                    placeholder="Campaign",
                    value='campaign',
                ),
                dcc.Dropdown(
                    id='part-type-dropdown',
                    options=[{'label': i, 'value': i} for i in particle_types],
                    placeholder="Particle Type",
                    value='particle_type',
                ),
                dcc.DatePickerRange(
                    id="date_range",
                    min_date_allowed=date(2000, 3, 1),
                    max_date_allowed=date(2016, 10, 1),
                    initial_visible_month=date(2003, 1, 1),
                    start_date=date(2000, 1, 1),
                    end_date=date(2016, 10, 1),
                ),
            ]
        ),
        html.Div(
            [
                html.Br(),
                html.Div(id='output-state'),
                html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
            ]
        ),
        html.Div([html.Div(dcc.Graph(id='pie'))]),
    ]
)


def choose_campaign(campaign):
    df = pd.read_csv(f"{config.FINAL_DIR}{campaign}.csv")
    return df[df['campaign'] == campaign]


def choose_particle_type(part_type):
    df = pd.read_csv(f"{config.FINAL_DIR}/{part_type}.csv")
    return df[df['classification'] == part_type]


# @ app.callback(
#     #Output(component_id='output', component_property='children'),
#     Input(component_id='part-type-dropdown', component_property='value')
# )
@app.callback(
    Output("pie", "figure"),
    [
        Input("campaign-dropdown", "value"),
        Input("submit-button-state", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
    ],
)
def percent_part_type(campaign, submit, start_date, end_date):
    if submit is None:
        raise PreventUpdate
    else:
        df = choose_campaign(campaign)
        values = df['classification'].value_counts().values
        return px.pie(
            df, values=values, names='classification', title='Particle Type Percentage'
        )


@app.callback(
    Output("output-state", "children"),
    Input("campaign-dropdown", "value"),
    Input("part-type-dropdown", "value"),
)
def update_output(campaign, part_type):
    string = f"Campaign chosen: {campaign} Particle type chose: {part_type}"
    return string


# Run local server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False, threaded=True)
