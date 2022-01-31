'''
plotly figures and callback functionality based on user input
'''
import plot_globe
import plotly.graph_objects as go
from topo_flat import Etopo

# from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pandas as pd

# import dash
import os
import globals
import orjson
import process
import datetime
from dash_extensions.enrich import Output, Input, ServersideOutput
import seaborn as sns
from joypy import joyplot
from dash import dash_table


def register_callbacks(app):
    @app.callback(
        Output("flat-topo", "figure"),
        Input("store-df", "data"),
        Input("map-particle_type", "value"),
        # Input("3d_vertical_prop", "value"),
    )
    def topo_flat(df, part_type, resolution=0.15):
        '''particle location overlaid on topographic map'''

        # df = pd.read_json(json_file)
        df = df[df['Classification'].isin(part_type)]
        fig = px.scatter_3d(
            df,
            x=-df['Latitude'],
            y='Longitude',
            z='Altitude',
            # range_z=zrange,
            color='Classification',
            # hover_data={'Ice Water Content': True, 'Temperature': True, 'Pressure': True},
            # custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
            size=df['Ice Water Content'],
        )
        # don't outline scatter markers
        # fig.update_traces(marker=dict(line=dict(width=0)))
        # select plot range for Earth [[lon min, lon max], [lat min, lat max]]
        lon, lat, topo = Etopo(
            [min(df['Longitude']) - 10, max(df['Longitude']) + 10],
            [-min(df['Latitude']) - 10, -max(df['Latitude']) + 10],
            resolution,
        )
        fig.add_trace(
            go.Surface(x=lat, y=lon, z=np.array(topo), colorscale=globals.Ctopo)
        )

        noaxis = dict(
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks='',
            title='',
            zeroline=False,
        )
        camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=0.6, y=0.0, z=3),
        )
        fig.update_layout(
            width=1100,
            height=400,
            xaxis_title='Latitude',
            # coloraxis_colorbar=dict(x=-0.1),
            margin=dict(l=0, r=0, b=0, t=0),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=2, y=5, z=0.3),
            # scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis),
            scene_camera=camera,
            legend=dict(
                x=1.2,
                y=1.0,
            ),
        )
        fig = process.update_layout(fig, df)
        return fig

    @app.callback(
        ServersideOutput("store-df", "data"),
        Input("campaign-dropdown", "value"),
        Input("min-temp", "value"),
        Input("max-temp", "value"),
        Input("min-pres", "value"),
        Input("max-pres", "value"),
        Input("date-picker", 'start_date'),
        Input("date-picker", 'end_date'),
    )
    def preprocess(
        campaign, min_temp, max_temp, min_pres, max_pres, start_date, end_date
    ):
        df = process.read_campaign(campaign)
        df = process.remove_bad_props(df)
        df = process.remove_bad_env(df)
        # print(df['Pressure'].min(), df['Pressure'].max())
        df = process.rename(df)
        df = process.check_temp_range(df, min_temp, max_temp)
        df = process.check_pres_range(df, min_pres[0], max_pres[0])
        df = process.check_date_range(df, start_date, end_date)
        # tic = datetime.datetime.now()
        # orjson.dumps(df.to_dict(orient='records'))
        # toc = datetime.datetime.now()
        # print(f"TIME TO SERIALIZE = {(toc-tic).total_seconds()}")
        return df

    @app.callback(
        Output('date-picker', 'min_date_allowed'),
        Output('date-picker', 'max_date_allowed'),
        [Input('campaign-dropdown', 'value')],
    )
    def set_date_picker(campaign):
        '''update date picker based on campaign start and end dates'''
        return (
            globals.campaign_start_dates[campaign],
            globals.campaign_end_dates[campaign],
        )

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

    @app.callback(
        Output("lon-alt-hist", "figure"),
        Input("store-df", "data"),
    )
    def lon_alt_hist(df):
        fig = px.density_contour(
            df,
            x="Longitude",
            y="Altitude",
            color='Classification',
            marginal_x="violin",
            marginal_y="violin",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        fig = process.update_layout(fig, df, contour=True)
        return fig

    @app.callback(
        Output("lat-alt-hist", "figure"),
        Input("store-df", "data"),
    )
    def lat_alt_hist(df):
        fig = px.density_contour(
            df,
            x="Latitude",
            y="Altitude",
            color='Classification',
            marginal_x="violin",
            marginal_y="violin",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        fig = process.update_layout(fig, df, contour=True)
        return fig

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

    @app.callback(
        Output('table', 'data'),
        Input("store-df", "data"),
    )
    def datatable(df):
        return [
            dash_table.DataTable(rows=df.to_dict(orient='records'), columns=df.columns)
        ]

    # @app.callback(
    #     Output("top-down-map", "figure"),
    #     Input("store-df", "data"),
    #     Input("map-particle_type", "value"),
    # )
    # def map_top_down(
    #     df,
    #     part_type,
    # ):
    #     '''aircraft location and particle type overlaid on map'''
    #     # df = pd.read_json(json_file)
    #     df = df[df['Classification'].isin(part_type)]

    #     # Find Lat Long center
    #     lat_center = df['Latitude'][df['Latitude'] != -999.99].mean()
    #     lon_center = df['Longitude'][df['Latitude'] != -999.99].mean()

    #     fig = px.scatter_mapbox(
    #         df,
    #         lat="Latitude",
    #         lon="Longitude",
    #         color='Classification',
    #         size=df['Ice Water Content'] * 4,
    #         color_discrete_sequence=px.colors.qualitative.Antique,
    #         hover_data={
    #             'Ice Water Content': True,
    #             'Temperature': True,
    #             'Pressure': True,
    #         },
    #         custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
    #     )
    #     fig.update_traces(
    #         hovertemplate="<br>".join(
    #             [
    #                 "Latitude: %{lat}",
    #                 "Longitude: %{lon}",
    #                 "Temperature: %{customdata[0]}",
    #                 "Pressure: %{customdata[1]}",
    #                 "Ice Water Content: %{customdata[2]}",
    #             ]
    #         ),
    #     )
    #     # Specify layout information
    #     fig.update_layout(
    #         title={
    #             'text': f"n={len(df)}",
    #             'x': 0.5,
    #             'xanchor': 'center',
    #             'yanchor': 'top',
    #         },
    #         mapbox_layers=[
    #             {
    #                 "below": 'traces',
    #                 "sourcetype": "raster",
    #                 "sourceattribution": "United States Geological Survey",
    #                 "source": [
    #                     "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
    #                 ],
    #             }
    #         ],
    #         mapbox=dict(
    #             style='light',
    #             accesstoken=os.getenv('MAPBOX_TOKEN'),
    #             center=dict(lon=lon_center, lat=lat_center),
    #             zoom=5,
    #         ),
    #     )
    #     return fig

    # @app.callback(
    #     Output("3d map", "figure"),
    #     Input("campaign-dropdown", "value"),
    #     Input("map-particle_type", "value"),
    #     Input("3d_vertical_prop", "value"),
    #     Input("min-temp", "value"),
    #     Input("max-temp", "value"),
    #     Input("min-pres", "value"),
    #     Input("max-pres", "value"),
    #     Input("date-picker", 'start_date'),
    #     Input("date-picker", 'end_date'),
    # )
    # def three_d_map(
    #     campaign,
    #     part_type,
    #     vert_prop,
    #     min_temp,
    #     max_temp,
    #     min_pres,
    #     max_pres,
    #     start_date,
    #     end_date,
    # ):

    #     df = process.read_campaign(campaign)
    #     df = process.remove_bad_env(df)
    #     df = process.rename(df)
    #     if campaign == 'CRYSTAL_FACE_NASA':
    #         df = df[df['Latitude'] > 23.0]
    #     df = df[df['Classification'].isin(part_type)]
    #     df = process.check_temp_range(df, min_temp, max_temp)
    #     df = process.check_pres_range(df, min_pres[0], max_pres[0])
    #     df = process.check_date_range(df, start_date, end_date)

    #     if vert_prop == 'Temperature':
    #         zrange = [min(df['Temperature']), 10]
    #     else:
    #         zrange = [df[vert_prop].min(), df[vert_prop].max()]

    #     fig = px.scatter_3d(
    #         df,
    #         x='Latitude',
    #         y='Longitude',
    #         z=vert_prop,
    #         range_z=zrange,
    #         color=vert_prop,
    #         color_continuous_scale=px.colors.sequential.Blues[::-1],
    #         hover_data={
    #             'Ice Water Content': True,
    #             'Temperature': True,
    #             'Pressure': True,
    #         },
    #         custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
    #         size=df['Ice Water Content'] * 5,
    #     )
    #     fig.update_traces(
    #         mode='markers',
    #         marker_line_width=0,
    #         hovertemplate="<br>".join(
    #             [
    #                 "Latitude: %{x}",
    #                 "Longitude: %{y}",
    #                 "Temperature: %{customdata[0]}",
    #                 "Pressure: %{customdata[1]}",
    #                 "Ice Water Content: %{customdata[2]}",
    #             ]
    #         ),
    #     )
    #     fig.update_layout(
    #         title={
    #             'text': f"n={len(df)}",
    #             'x': 0.45,
    #             'xanchor': 'center',
    #             'yanchor': 'top',
    #         },
    #     )
    #     if vert_prop == 'Temperature' or vert_prop == 'Pressure':
    #         fig.update_scenes(zaxis_autorange="reversed")
    #     return fig

    # @app.callback(
    #     Output("globe", "figure"),
    #     Input("campaign-dropdown", "value"),
    #     Input("map-particle_type", "value"),
    #     Input("3d_vertical_prop", "value"),
    #     Input("min-temp", "value"),
    #     Input("max-temp", "value"),
    #     Input("min-pres", "value"),
    #     Input("max-pres", "value"),
    #     Input("date-picker", 'start_date'),
    #     Input("date-picker", 'end_date'),
    # )
    # def globe(
    #     campaign,
    #     part_type,
    #     vert_prop,
    #     min_temp,
    #     max_temp,
    #     min_pres,
    #     max_pres,
    #     start_date,
    #     end_date,
    # ):

    #     df = process.read_campaign(campaign)
    #     df = process.remove_bad_env(df)
    #     df = process.rename(df)
    #     if campaign == 'CRYSTAL_FACE_NASA':
    #         df = df[df['Latitude'] > 23.0]
    #     df = df[df['Classification'].isin(part_type)]
    #     df = process.check_temp_range(df, min_temp, max_temp)
    #     df = process.check_pres_range(df, min_pres[0], max_pres[0])
    #     df = process.check_date_range(df, start_date, end_date)

    #     fig = plot_globe.main(df)
    #     return fig
