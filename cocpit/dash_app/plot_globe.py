'''
runner for plotting an interactive globe with topography
 and overlaid ice crystals at altitude'''

import topo_map.TopoMap as TopoMap
import app.read_campaign as read_campaign
from topo_map import mapping_map_to_sphere
import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
from colorbar import Cscale_EQ


def create_topo_map():
    '''call TopoMap in topo_map.py to create globe'''
    map = TopoMap()
    map.read_netCDF_globe()
    map.mesh_grid()
    map.reshape()
    map.skip_for_resolution()
    map.select_range()
    map.convert_2D
    xs, ys, zs = mapping_map_to_sphere(map.lon, map.lat)
    return map.topo, xs, ys, zs


def globe_layout(titlecolor='white', bgcolor='white'):
    noaxis = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks='',
        title='',
        zeroline=False,
    )

    layout = go.Layout(
        autosize=False,
        width=1200,
        height=800,
        title='3D spherical topography map',
        titlefont=dict(family='Courier New', color=titlecolor),
        showlegend=False,
        scene=dict(
            xaxis=noaxis,
            yaxis=noaxis,
            zaxis=noaxis,
            aspectmode='manual',
            aspectratio=go.layout.scene.Aspectratio(x=1, y=1, z=1),
        ),
        paper_bgcolor=bgcolor,
        plot_bgcolor=bgcolor,
    )
    return layout


def scatter_coords():
    df = read_campaign('CRYSTAL_FACE_NASA')
    lon = np.array(df['Longitude'])
    lat = np.array(df['Latitude'])
    alt = np.array(df['Altitude'])

    xs_ev_org, ys_ev_org, zs_ev_org = mapping_map_to_sphere(lon, lat)

    # Create three-dimensional effect
    # ratio = 1. - alt*2e-4
    # xs_ev = xs_ev_org*ratio
    # ys_ev = ys_ev_org*ratio
    # zs_ev = zs_ev_org*ratio

    ratio = 1.15 - alt * 2e-4
    xs_ev_up = xs_ev_org * ratio
    ys_ev_up = ys_ev_org * ratio
    zs_ev_up = zs_ev_org * ratio
    return xs_ev_up, ys_ev_up, zs_ev_up


def create_crystal_scatter(xs_ev_up, ys_ev_up, zs_ev_up, alt):
    return go.Scatter3d(
        x=xs_ev_up,
        y=ys_ev_up,
        z=zs_ev_up,
        mode='markers',
        name='measured',
        marker=dict(
            size=3,
            colorbar=dict(
                title='Altitude',
                titleside='right',
                titlefont=dict(size=16, color='black', family='Courier New'),
                tickmode='array',
                tickcolor='black',
                tickfont=dict(size=14, color='black', family='Courier New'),
            ),
            color=alt,
            ### choose color option
            colorscale=Cscale_EQ,
            showscale=True,
            opacity=1.0,
        ),
    )


def plot_globe(topo, xs, ys, zs):
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
    cmin = -8000
    cmax = 8000

    ratio_topo = 1.0 + topo * 1e-5
    xs_3d = xs * ratio_topo
    ys_3d = ys * ratio_topo
    zs_3d = zs * ratio_topo

    return dict(
        type='surface',
        x=xs_3d,
        y=ys_3d,
        z=zs_3d,
        colorscale=Ctopo,
        surfacecolor=topo,
        opacity=1.0,
        cmin=cmin,
        cmax=cmax,
        showscale=False,
        hoverinfo='skip',
    )


def main():
    topo, xs, ys, zs = create_topo_map()
    layout = globe_layout()
    topo_sphere_3d = plot_globe(topo, xs, ys, zs)
    crystal_scatter = create_crystal_scatter()

    plot_data = [topo_sphere_3d, crystal_scatter]
    fig = go.Figure(data=plot_data, layout=layout)
    plot(fig, validate=False, filename='SphericalTopography.html', auto_open=True)


if __name__ == 'main':
    main()
