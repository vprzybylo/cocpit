'''
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
'''

import plotly.express as px
from callbacks import process
from dash_extensions.enrich import Input, Output


def register(app):
    @app.callback(
        Output("pie", "figure"),
        Input("df-classification", "data"),
    )
    def pie(df_classification):
        '''pie chart for percentage of particle types for a given campaign'''

        values = df_classification.value_counts()
        labels = df_classification.unique()

        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color_discrete_sequence=px.colors.qualitative.Antique,
        )

        return process.update_layout(pie, contour=True)

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
        return process.update_layout(prop_fig)
