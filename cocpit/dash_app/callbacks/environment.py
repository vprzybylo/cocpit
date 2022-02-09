'''
plot figures for environmental attributes of ice crystals
2d histogram contours for particle type in vertical cross section
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash import dash_table
from dash_extensions.enrich import Input, Output


def register(app):
    @app.callback(
        Output("type-temp-violin", "figure"),
        [
            Input("df-classification", "data"),
            Input("df-temp", "data"),
        ],
    )
    def type_temp_violin(classification, temp):
        '''violin plot of particle type vs temperature'''
        temp_fig = px.violin(
            x=classification,
            y=temp,
            color=classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            points=False,
            labels={
                "x": "Particle Type",
                "y": 'Temperature',
            },
        )

        temp_fig = process.update_layout(temp_fig, len(temp))
        temp_fig.update_yaxes(autorange="reversed")
        return temp_fig

    @app.callback(
        Output("type-iwc-violin", "figure"),
        [
            Input("df-classification", "data"),
            Input("df-iwc", "data"),
        ],
    )
    def type_temp_violin(classification, iwc):
        '''violin plot of particle type vs ice water content'''
        iwc_fig = px.violin(
            x=classification,
            y=iwc,
            color=classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Particle Type",
                "y": 'Ice Water Content',
            },
        )
        iwc_fig = process.update_layout(iwc_fig, len(classification))

        return iwc_fig

    # @app.callback(
    #     Output('table', 'style_data_conditional'),
    #     Input("store-df", "data"),
    # )
    # def datatable(df):
    #     return [
    #         dash_table.DataTable(
    #             id='table',
    #             columns=[{"name": i, "id": i} for i in df.columns],
    #             data=df.to_dict('records'),
    #             # export_columns='all',
    #             export_headers='display',
    #             export_format="csv",
    #             fixed_rows={'headers': True},
    #             style_table={'height': '300px', 'overflowY': 'auto'},
    #         )
    #     ]
