'''
Executable for plotting an interactive globe with topography
and overlaid ice crystals at altitude
'''

from topo_map import TopoMap as TopoMap
from topo_map import mapping_map_to_sphere
import plotly.graph_objs as go
from plotly.offline import plot
from scatter import Scatter


def create_topo_map():
    '''call TopoMap in topo_map.py to create globe'''
    map = TopoMap()
    map.read_netCDF_globe()
    map.mesh_grid()
    map.reshape()
    map.skip_for_resolution()
    map.select_range()
    map.convert_2D()
    xs, ys, zs = mapping_map_to_sphere(map.lon, map.lat)
    return map.topo, xs, ys, zs


def create_scatter(df):
    '''create 3d scatter plot on globe'''
    s = Scatter(df)
    s.scatter_coords()
    return s.plot_scatter()


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


def camera():
    return dict(
        # up=dict(x=0, y=1, z=1),
        # center=dict(x=0, y=0, z=0),
        # eye=dict(x=0.5, y=0.2, z=1),
    )


def main(df):
    # create interactive topographic globe
    topo, xs, ys, zs = create_topo_map()
    topo_sphere_3d = plot_globe(topo, xs, ys, zs)

    # create scatter above globe with ice crystal location
    crystal_scatter = create_scatter(df)

    plot_data = [topo_sphere_3d, crystal_scatter]
    layout = globe_layout()
    fig = go.Figure(data=plot_data, layout=layout)
    fig.update_layout(scene_camera=camera())
    # plot(fig, validate=False, filename='SphericalTopography.html', auto_open=True)
    return fig
