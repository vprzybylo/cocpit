from datetime import date

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
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
                html.H1('COCPIT'),
                html.H2('Classification of Cloud Particle Imagery and Thermodynamics'),
                html.A(
                    "Images are classified from the Cloud Particle Imager",
                    href="http://www.specinc.com/cloud-particle-imager",
                ),
            ],
            style={'text-align': 'center'},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='campaign-dropdown',
                    options=[{'label': i, 'value': i} for i in campaigns],
                    placeholder="Campaign",
                    value='ATTREX',
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
            ],
            style={"width": "20%"},
        ),
        # html.Div(
        #     [
        #         html.Br(),
        #         html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
        #     ]
        # ),
        html.Div(
            [
                html.Div(dcc.Graph(id='pie')),
                html.Div(dcc.Graph(id='phi')),
                html.Div(dcc.Graph(id='complexity')),
            ],
            # style={
            #     'display': 'inline-block',
            #     #'justifyContent': 'center',
            #     'height': '80%',
            # },
            # className="row",
        ),
    ],
    className="container",
)


def remove_baddata(df_CPI):
    df_CPI = df_CPI[df_CPI["filled_circular_area_ratio"] != -999.0]
    df_CPI = df_CPI[df_CPI["complexity"] != 0.0]
    df_CPI = df_CPI[df_CPI["complexity"] != -0.0]

    df_CPI = df_CPI[df_CPI.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    df_CPI.dropna(inplace=True)
    return df_CPI


def choose_campaign(campaign):
    return pd.read_csv(f"{config.FINAL_DIR}{campaign}.csv")


def choose_particle_type(part_type):
    df = pd.read_csv(f"{config.FINAL_DIR}{part_type}.csv")
    return df[df['classification'] == part_type]


@app.callback(
    Output("pie", "figure"),
    [
        Input("campaign-dropdown", "value"),
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
    ],
)
def percent_part_type(campaign, start_date, end_date):
    df = choose_campaign(campaign)
    values = df['classification'].value_counts().values
    pie = px.pie(
        df,
        color_discrete_sequence=px.colors.qualitative.Antique,
        values=values,
        names=df['classification'].unique(),
    )
    pie.update_layout(title_text='Particle Type Percentage', title_x=0.45)
    return pie


@app.callback(
    Output("phi", "figure"),
    [
        Input("campaign-dropdown", "value"),
    ],
)
def phi(campaign):
    df = choose_campaign(campaign)
    df = remove_baddata(df)
    phi_fig = px.box(
        df,
        x='classification',
        y="phi",
        color="classification",
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "phi": "Aspect Ratio",
            "classification": "Particle Type",
        },
    )
    phi_fig.update(layout_yaxis_range=[min(df['phi']), max(df['phi'])])
    return phi_fig


@app.callback(
    Output("complexity", "figure"),
    [
        Input("campaign-dropdown", "value"),
    ],
)
def complexity(campaign):
    df = choose_campaign(campaign)
    df = remove_baddata(df)
    complexity_fig = px.box(
        df,
        x='classification',
        y="complexity",
        color="classification",
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "complexity": "Complexity",
            "classification": "Particle Type",
        },
    )
    complexity_fig.update(
        layout_yaxis_range=[min(df['complexity']), max(df['complexity'])]
    )
    return complexity_fig


# @app.callback(
#     Output("output-state", "children"),
#     Input("campaign-dropdown", "value"),
#     Input("part-type-dropdown", "value"),
# )
# def update_output(campaign, part_type):
#     string = f"Campaign chosen: {campaign} Particle type chose: {part_type}"
#     return string


# Run local server
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False, threaded=True)
