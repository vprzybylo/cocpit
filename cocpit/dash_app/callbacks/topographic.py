'''
plot topographic map
scatter plot overlaid where particle images were captured
includes callbacks
'''
import plotly.graph_objects as go
import os
import pandas as pd
import plotly.express as px
from processing_scripts import process
from callbacks.topo_map import TopoMap as TopoMap
from dash_extensions.enrich import Input, Output
import globals
import numpy as np


def register(app):
    @app.callback(
        Output('density-contour', 'figure'),
        [
            Input('df-classification', 'data'),
            Input('df-lat', 'data'),
            Input('df-lon', 'data'),
        ],
    )
    def density_contour(df_classification, df_lat, df_lon):
        '''2d histogram of particles in space with particle type plotted as color'''

        # re-sort by original index so that rimed isn't plotted on top
        #   - was blocking all other colors/particle types
        # the df was originally sorted alphabetically
        #   - so that particle type colors are always consistent across viiolin figures
        df_classification = df_classification.sort_index()
        df_lat = df_lat.sort_index()
        df_lon = df_lon.sort_index()
        lat_center = df_lat[df_lat != -999.99].mean()
        lon_center = df_lon[df_lon != -999.99].mean()
        # fig = px.scatter(
        #     x=df_lon,
        #     y=df_lat,
        #     marginal_x='histogram',
        #     marginal_y='histogram',
        #     color=df_classification,
        #     color_discrete_map=globals.color_discrete_map,
        #     labels={"x": "Longitude", "y": "Latitude"},
        # )

        gridx = np.linspace(df_lon.min(), df_lon.max())
        gridy = np.linspace(df_lat.min(), df_lat.max())
        grid, _, _ = np.histogram2d(df_lon, df_lat, bins=[gridx, gridy])

        # find center points of grid in x and y
        avg_x = [(a + b) / 2 for a, b in zip(gridx[::1], gridx[1::1])]
        avg_y = [(a + b) / 2 for a, b in zip(gridy[::1], gridy[1::1])]

        center_lats = []
        center_lons = []
        counts = []
        for x, center_lat in enumerate(avg_x):
            for y, center_lon in enumerate(avg_y):
                counts.append(grid[x, y])
                center_lats.append(center_lat)
                center_lons.append(center_lon)

        fig = px.density_mapbox(
            lat=center_lons,
            lon=center_lats,
            z=counts,
            color_continuous_scale=px.colors.sequential.OrRd_r,
            radius=10,
            center=dict(lat=lat_center, lon=lon_center),
            zoom=5,
            mapbox_style="stamen-terrain",
        )

        return process.update_layout(fig, contour=True, margin=5)

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
        df_classification = df_classification.sort_index()
        df_lat = df_lat.sort_index()
        df_lon = df_lon.sort_index()

        fig = px.scatter_mapbox(
            lat=df_lat,
            lon=df_lon,
            color=df_classification,
            color_discrete_map=globals.color_discrete_map,
            mapbox_style="stamen-terrain",
        )

        # Specify layout information
        fig.update_layout(
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
            color_discrete_map=globals.color_discrete_map,
            labels={
                "x": "Particle Type",
                "y": 'Altitude',
            },
        )
        # print(df_classification.value_counts())

        vert_dist = process.update_layout(vert_dist, contour=False)
        return vert_dist
