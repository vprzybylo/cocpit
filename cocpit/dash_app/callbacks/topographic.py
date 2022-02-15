'''
plot topographic map
scatter plot overlaid where particle images were captured
includes callbacks
'''

import os

import globals
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from callbacks import process, topo_flat
from callbacks.topo_map import TopoMap as TopoMap
from dash_extensions.enrich import Input, Output


def register(app):
    @app.callback(
        Output("lon-alt-hist", "figure"),
        Input("df-lon", "data"),
        Input("df-alt", "data"),
        Input("df-classification", "data"),
    )
    def lon_alt_hist(longitude, altitude, classification):
        lon_fig = px.density_contour(
            x=longitude,
            y=altitude,
            color=classification,
            marginal_x="box",
            marginal_y="box",
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Longitude",
                "y": 'Altitude',
            },
        )
        lon_fig = process.update_layout(lon_fig, len(classification), contour=True)
        return lon_fig

    @app.callback(
        Output("lat-alt-hist", "figure"),
        Input("df-lat", "data"),
        Input("df-alt", "data"),
        Input("df-classification", "data"),
    )
    def lat_alt_hist(latitude, altitude, classification):
        lat_fig = px.density_contour(
            x=latitude,
            y=altitude,
            color=classification,
            marginal_x="box",
            marginal_y="box",
            color_discrete_sequence=px.colors.qualitative.Antique,
            labels={
                "x": "Latitude",
                "y": 'Altitude',
            },
        )
        lat_fig = process.update_layout(lat_fig, len(classification), contour=True)
        return lat_fig

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
            title={
                'text': f"n={len(df_classification)}",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            },
            mapbox_layers=[
                {
                    "below": 'traces',
                    "sourcetype": "raster",
                    "sourceattribution": "United States Geological Survey",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
                    ],
                }
            ],
            mapbox=dict(
                style='light',
                accesstoken=os.getenv('MAPBOX_TOKEN'),
                center=dict(lon=lon_center, lat=lat_center),
                zoom=5,
            ),
        )
        return fig
