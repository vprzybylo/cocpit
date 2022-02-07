'''
plot figures for environmental attributes of ice crystals
2d histogram contours for particle type in vertical cross section
includes callbacks
'''

import pandas as pd
import plotly.express as px
from callbacks import process
from dash import dash_table
from dash_extensions.enrich import Input, Output, State


def register(app):
    @app.callback(
        [Output("type-temp-violin", "figure"), Output("type-iwc-violin", "figure")],
        [
            Input("store-df", "data"),
        ],
    )
    def type_temp_violin(df):
        print(df)
        temp_fig = px.violin(
            x=df["Classification"],
            y=df["Temperature"],
            color=df["Classification"],
            color_discrete_sequence=px.colors.qualitative.Antique,
            points=False,
        )

        temp_fig = process.update_layout(temp_fig, df)
        temp_fig.update_yaxes(autorange="reversed")

        iwc_fig = px.violin(
            x=df["Classification"],
            y=df["Ice Water Content"],
            color=df["Classification"],
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        iwc_fig = process.update_layout(iwc_fig, df)

        return temp_fig, iwc_fig

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
