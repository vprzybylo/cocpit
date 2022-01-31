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


def register_callbacks(app):
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
    def percent_part_type(prop, df):
        '''box plots of geometric attributes of ice crystals
        with respect to particle classification on sidebar menu filters'''
        # df = pd.read_json(json_file)
        fig = px.box(
            df,
            x='Classification',
            y=prop,
            color="Classification",
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "Classification": "Particle Type",
            },
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
        Output("flat-topo", "figure"),
        Input("store-df", "data"),
        Input("map-particle_type", "value"),
        Input("3d_vertical_prop", "value"),
    )
    def topo_flat(df, part_type, vert_prop):
        '''particle location overlaid on topographic map'''

        # df = pd.read_json(json_file)
        df = df[df['Classification'].isin(part_type)]

        Ctopo = [
            [0, 'rgb(0, 0, 70)'],
            [0.2, 'rgb(0,90,150)'],
            [0.4, 'rgb(150,180,230)'],
            [0.5, 'rgb(210,230,250)'],
            [0.50001, 'rgb(0,120,0)'],
            [0.57, 'rgb(220,180,130)'],
            [0.65, 'rgb(120,100,0)'],
            [0.75, 'rgb(80,70,0)'],
            [0.9, 'rgb(200,200,200)'],
            [1.0, 'rgb(255,255,255)'],
        ]

        fig = px.scatter_3d(
            df,
            x=-df['Latitude'],
            y='Longitude',
            z='Altitude',
            # range_z=zrange,
            color=vert_prop,
            # hover_data={'Ice Water Content': True, 'Temperature': True, 'Pressure': True},
            # custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
            size=df['Ice Water Content'],
        )
        # don't outline scatter markers
        fig.update_traces(marker=dict(line=dict(width=0)))

        lon, lat, topo = Etopo([-180, 180], [-90, 90], 0.5)
        fig.add_trace(go.Surface(x=lat, y=lon, z=np.array(topo), colorscale=Ctopo))

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
            width=1300,
            height=300,
            margin=dict(l=0, r=0, b=0, t=0),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=2, y=5, z=0.3),
            scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis),
            scene_camera=camera,
        )
        return fig

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
