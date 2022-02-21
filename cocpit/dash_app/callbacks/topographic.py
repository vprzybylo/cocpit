'''
plot topographic map
scatter plot overlaid where particle images were captured
includes callbacks
'''

import os
import pandas as pd
import plotly.express as px
from callbacks import process
from callbacks.topo_map import TopoMap as TopoMap
from dash_extensions.enrich import Input, Output


def register(app):
    @app.callback(
        Output("top-down-map", "figure"),
        [
            Input("df-classification", "data"),
            Input("df-lat", "data"),
            Input("df-lon", "data"),
        ],
    )
    def map_top_down(df_classification, df_lat, df_lon):
        '''aircraft location and particle type overlaid on map'''

        # Find Lat Long center
        lat_center = df_lat[df_lat != -999.99].mean()
        lon_center = df_lon[df_lon != -999.99].mean()

        fig = px.scatter_mapbox(
            lat=df_lat,
            lon=df_lon,
            color=df_classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            mapbox_style="stamen-terrain"
            # hover_data={
            #     'Ice Water Content': True,
            #     'Temperature': True,
            #     'Pressure': True,
            # },
            # custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
        )

        # fig.update_traces(
        #     hovertemplate="<br>".join(
        #         [
        #             "Latitude: %{lat}",
        #             "Longitude: %{lon}",
        #             "Temperature: %{customdata[0]}",
        #             "Pressure: %{customdata[1]}",
        #             "Ice Water Content: %{customdata[2]}",
        #         ]
        #     ),
        # )
        # Specify layout information
        fig.update_layout(
            # mapbox_layers=[
            #     {
            #         "below": 'traces',
            #         "sourcetype": "raster",
            #         "sourceattribution": "United States Geological Survey",
            #         "source": [
            #             "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            #         ],
            #     }
            # ],
            mapbox=dict(
                accesstoken=os.getenv('MAPBOX_TOKEN'),
                center=dict(lon=lon_center, lat=lat_center),
                zoom=5,
            ),
        )
        return process.update_layout(fig, contour=True, margin=5)

    @app.callback(
        Output("vert-dist", "figure"),
        Input("df-lon", "data"),
        Input("df-alt", "data"),
        Input("df-classification", "data"),
    )
    def vert_dist(df_lon, df_alt, df_classification):

        vert_dist = px.violin(
            x=df_classification,
            y=df_alt,
            color=df_classification,
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Particle Type",
                "y": 'Altitude',
            },
        )
        # print(df_classification.value_counts())

        vert_dist = process.update_layout(vert_dist, contour=False)
        return vert_dist
