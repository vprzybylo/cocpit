'''
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash_extensions.enrich import Input, Output


def register(app):
    @app.callback(
        [Output("pie", "figure"), Output("prop_fig", "figure")],
        [Input("property-dropdown", "value"), Input("store-df", "data")],
    )
    def percent_part_type(prop, df):
        '''pie chart for percentage of particle types for a given campaign'''

        # df = pd.read_json(df['Classification', prop])
        pie = px.pie(
            df,
            color_discrete_sequence=px.colors.qualitative.Antique,
            values=df["Classification"].value_counts(),
            names=df["Classification"].unique(),
        )

        # pie.update_layout(
        #     title={
        #         'text': f"n={values}",
        #         'x': 0.43,
        #         'xanchor': 'center',
        #         'yanchor': 'top',
        #     }
        # )

        '''box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters'''
        prop_fig = px.violin(
            df,
            x=df['Classification'],
            y=df[prop],
            color=df["Classification"],
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "Classification": "Particle Type",
            },
        )
        prop_fig = process.update_layout(prop_fig, df)
        return pie, prop_fig
