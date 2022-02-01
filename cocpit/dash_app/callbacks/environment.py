'''
plot figures for environmental attributes of ice crystals
2d histogram contours for particle type in vertical cross section
includes callbacks
'''

from dash_extensions.enrich import Output, Input
import plotly.express as px
from callbacks import process
from dash import dash_table


def register(app):
    @app.callback(
        Output("type-temp-violin", "figure"),
        Input("store-df", "data"),
    )
    def type_temp_violin(df):

        fig = px.violin(
            df,
            x='Classification',
            y='Temperature',
            color='Classification',
            color_discrete_sequence=px.colors.qualitative.Antique,
            points=False,
        )

        fig = process.update_layout(fig, df)
        fig.update_yaxes(autorange="reversed")
        return fig

    @app.callback(
        Output("type-iwc-violin", "figure"),
        Input("store-df", "data"),
    )
    def type_iwc_violin(df):

        fig = px.violin(
            df,
            x='Classification',
            y='Ice Water Content',
            color='Classification',
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        fig = process.update_layout(fig, df)

        return fig

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
