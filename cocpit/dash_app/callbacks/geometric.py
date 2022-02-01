'''
plot figures for geometric attributes of ice crystals
pie chart for particle percentage and violin plot for particle property for each class
includes callbacks
'''

from dash_extensions.enrich import Output, Input
import plotly.express as px
from callbacks import process


def register(app):
    @app.callback(Output("pie", "figure"), Input("store-df", "data"))
    def percent_part_type(df):
        '''pie chart for percentage of particle types for a given campaign'''
        # df = pd.read_json(json_file)

        values = df['Classification'].value_counts().values
        fig = px.pie(
            df,
            color_discrete_sequence=px.colors.qualitative.Antique,
            values=values,
            names=df['Classification'].unique(),
        )
        fig.update_layout(
            title={
                'text': f"n={len(df)}",
                'x': 0.43,
                'xanchor': 'center',
                'yanchor': 'top',
            }
        )
        return fig

    @app.callback(
        Output("prop_fig", "figure"),
        Input("property-dropdown", "value"),
        Input("store-df", "data"),
    )
    def geometric_properties(prop, df):
        '''box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters'''
        # df = pd.read_json(json_file)
        fig = px.violin(
            df,
            x='Classification',
            y=prop,
            color="Classification",
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "Classification": "Particle Type",
            },
        )
        fig = process.update_layout(fig, df)
        return fig
