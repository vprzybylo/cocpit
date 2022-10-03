"""
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
"""

import plotly.express as px
from processing_scripts import process
from dash_extensions.enrich import Input, Output
import globals


def register(app):
    @app.callback(
        Output("pie", "figure"),
        Input("df-classification", "data"),
    )
    def pie(df_classification):
        """pie chart for percentage of particle types for a given campaign"""

        values = df_classification.value_counts()
        labels = values.keys()
        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color=labels,
            color_discrete_map=globals.color_discrete_map,
        )
        return process.update_layout(pie, contour=True)

    @app.callback(
        Output("pie-area", "figure"),
        Input("df-area", "data"),
        Input("df-classification", "data"),
    )
    def pie_area(df_area, df_classification):
        """pie chart for percentage of particle types for a given campaign"""
        # df_area[''].value_counts()

        labels = df_area["Classification"].unique()
        values = df_area.groupby(by="Classification", sort=True).sort_values(
            "a"
        )["Contour Area"].sum() / df_classification.value_counts(sort=True)
        print(df_classification.value_counts(sort=True))
        print(
            df_area.groupby(by="Classification", sort=True)
            .sort_values("a")["Contour Area"]
            .sum()
        )
        print(values)
        pie = px.pie(
            labels,
            values=values,
            names=labels,
            color=labels,
            color_discrete_map=globals.color_discrete_map,
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
        """box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters"""

        prop_fig = px.violin(
            x=classification,
            y=prop,
            color=classification,
            color_discrete_map=globals.color_discrete_map,
            labels={
                "x": "Particle Type",
                "y": prop_name,
            },
        )
        return process.update_layout(prop_fig)
