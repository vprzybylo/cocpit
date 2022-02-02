'''
plot topographic map
scatter plot overlaid where particle images were captured
includes callbacks
'''

from dash_extensions.enrich import Output, Input, ServersideOutput
import plotly.express as px
from callbacks import process
import numpy as np
import plotly.graph_objects as go
from callbacks.topo_map import TopoMap as TopoMap
from globe import plot_globe
import os
import globals
from topo_flat import Etopo


def create_topo_map(lon_area, lat_area):
    '''topographic map calls to topo_map.py'''
    map = TopoMap(lon_area=lon_area, lat_area=lat_area)
    map.read_netCDF_globe()
    map.mesh_grid()
    map.reshape()
    map.skip_for_resolution()
    map.select_range()
    map.convert_2D()
    return map.lon, map.lat, map.topo


def register(app):
    @app.callback(
        Output("flat-topo", "figure"),
        Input("store-df", "data"),
        Input("topo-map-particle_type", "value"),
    )
    def topo_flat(df, part_type):
        '''particle location overlaid on topographic map'''
        df = df[df['Classification'].isin(part_type)]
        fig = px.scatter_3d(
            df,
            x=-df["Latitude"],
            y=df["Longitude"],
            z=df["Altitude"],
            color=df["Classification"],
            color_discrete_sequence=px.colors.qualitative.Antique,
            # hover_data={'Ice Water Content': True, 'Temperature': True, 'Pressure': True},
            # custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
            size=df['Ice Water Content'],
        )

        # don't outline scatter markers
        fig.update_traces(marker=dict(line=dict(width=0)))

        # select plot range for Earth [[lon min, lon max], [lat min, lat max]]
        lon, lat, topo = Etopo(
            [min(df["Longitude"]) - 20, max(df["Longitude"]) + 20],
            [-min(df["Latitude"]) - 20, -max(df["Latitude"]) + 20],
            resolution=0.15,
        )
        fig.add_trace(
            go.Surface(x=lat, y=lon, z=np.array(topo), colorscale=globals.Ctopo)
        )

        camera = dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=0.6, y=0.0, z=3),
        )
        noaxis = dict(
            showbackground=False,
            showgrid=False,
            showline=False,
            # showticklabels=False,
            ticks='',
            title='',
            zeroline=False,
        )
        fig.update_layout(
            width=1100,
            height=450,
            xaxis_title="Latitude",
            yaxis_title="Longitude",
            # coloraxis_colorbar=dict(x=-0.1),
            margin=dict(l=0, r=0, b=0, t=0),
            scene_aspectmode="manual",
            scene_aspectratio=dict(x=2, y=5, z=0.3),
            scene=dict(xaxis=noaxis, yaxis=noaxis, zaxis=noaxis),
            scene_camera=camera,
            legend=dict(
                x=1.2,
                y=1.0,
            ),
        )
        fig = process.update_layout(fig, df, contour=True)
        return fig

    @app.callback(
        Output("lon-alt-hist", "figure"),
        Output("lat-alt-hist", "figure"),
        Input("store-df", "data"),
    )
    def lon_alt_hist(df):
        lon_fig = px.density_contour(
            x=df["Longitude"],
            y=df["Altitude"],
            color=df["Classification"],
            marginal_x="violin",
            marginal_y="violin",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        lon_fig = process.update_layout(lon_fig, df, contour=True)

        lat_fig = px.density_contour(
            x=df["Latitude"],
            y=df["Altitude"],
            color=df['Classification'],
            marginal_x="violin",
            marginal_y="violin",
            color_discrete_sequence=px.colors.qualitative.Antique,
        )
        lat_fig = process.update_layout(lat_fig, df, contour=True)
        return lon_fig, lat_fig

    # @app.callback(
    #     Output("top-down-map", "figure"),
    #     Input("store-df", "data"),
    #     Input("topo-map-particle_type", "value"),
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
