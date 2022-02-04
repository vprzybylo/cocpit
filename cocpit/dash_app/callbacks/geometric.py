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
    )
    def percent_part_type(values, labels):
        '''pie chart for percentage of particle types for a given campaign'''
        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        pie.update_layout(
            title={
                'text': f"n={len(values)}",
                'x': 0.43,
                'xanchor': 'center',
                'yanchor': 'top',
            }
        )
        return pie

    @app.callback(
        Output("prop_fig", "figure"),
        [
            Input("property-dropdown", "value"),
            Input("store-df", "data"),
        ],
    )
    def percent_part_type(prop, df):
        '''box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters'''

        prop_fig = px.violin(
            x=df['Classification'],
            y=df[prop],
            color=df["Classification"],
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "Classification": "Particle Type",
            },
        )
        prop_fig = process.update_layout(prop_fig, df)
        return prop_fig
