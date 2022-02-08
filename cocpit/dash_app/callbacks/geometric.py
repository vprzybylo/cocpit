'''
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash_extensions.enrich import Input, Output
import plotly.graph_objects as go


def register(app):
    @app.callback(
        Output("pie", "figure"),
        Input("pie-values", "data"),
        Input("pie-labels", "data"),
        Input("len-df", "data"),
    )
    def pie(values, labels, len_df):
        '''pie chart for percentage of particle types for a given campaign'''

        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        pie.update_layout(
            title={
                'text': f"n={len_df}",
                'x': 0.43,
                'xanchor': 'center',
                'yanchor': 'top',
            }
        )
        return pie

    @app.callback(
        Output("prop_fig", "figure"),
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
