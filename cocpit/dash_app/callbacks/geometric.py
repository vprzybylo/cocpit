'''
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash_extensions.enrich import Input, Output, State
import plotly.graph_objects as go


def register(app):
    @app.callback(
        Output("pie", "figure"),
        [
            Input("df-classification", "data"),
            Input("df-lat", "data"),
            Input("df-lon", "data"),
            Input('top-down-map', 'selectedData'),
        ],
    )
    def pie(df_classification, df_lat, df_lon, selectedData):
        '''pie chart for percentage of particle types for a given campaign'''
        if selectedData and selectedData['points']:
            print(selectedData['points'])

            sel_data = pd.DataFrame(selectedData['points'])
            df_classification = df_classification[
                (df_lat.isin(sel_data['lat'])) & (df_lon.isin(sel_data['lon']))
            ]
            print(len(df_classification))
        values = df_classification.value_counts()
        labels = df_classification.unique()

        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        pie.update_layout(
            title={
                'text': f"n={len(df_classification)}",
                'x': 0.43,
                'xanchor': 'center',
                'yanchor': 'top',
            }
        )
        return pie

    @app.callback(
        Output("prop-fig", "figure"),
        [
            Input("df-classification", "data"),
            Input("df-prop", "data"),
            Input("property-dropdown", "value"),
        ],
    )
    def percent_part_type(classification, prop, prop_name):
        '''box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters'''

        prop_fig = px.violin(
            x=classification,
            y=prop,
            color=classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Particle Type",
                "y": prop_name,
            },
        )
        prop_fig = process.update_layout(prop_fig, len(classification))
        return prop_fig
