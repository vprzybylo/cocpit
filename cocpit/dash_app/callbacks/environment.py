'''
plot figures for environmental attributes of ice crystals
2d histogram contours for particle type in vertical cross section
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash_extensions.enrich import Input, Output, State


def register(app):
    @app.callback(
        Output("type-env-violin", "figure"),
        [
            Input("df-env", "data"),
            Input("df-classification", "data"),
            State("env-dropdown", "value"),
        ],
    )
    def environment_violin(env_prop, classification, label):
        '''violin plot of particle type vs ice water content'''
        fig = px.violin(
            x=classification,
            y=env_prop,
            color=classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Particle Type",
                "y": label,
            },
        )
        return process.update_layout(fig)
