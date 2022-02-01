from dataclasses import dataclass, field
from callbacks.topo_map import mapping_map_to_sphere
import pandas as pd
import numpy as np
from globe.colorbar import Cscale_EQ
import plotly.graph_objs as go


@dataclass
class Scatter:
    df: pd.core.frame.DataFrame
    xs_ev_up: float = field(init=False)
    ys_ev_up: float = field(init=False)
    zs_ev_up: float = field(init=False)

    def __post_init__(self):
        self.lon = np.array(self.df['Longitude'])
        self.lat = np.array(self.df['Latitude'])
        self.alt = np.array(self.df['Altitude'])
        self.iwc = np.array(self.df['Ice Water Content'])
        self.temp = np.array(self.df['Temperature'])
        # self.pres = np.ndarray(self.df['Pressure'])

    def scatter_coords(self):
        '''remap ice crystal coordinates on globe'''
        self.xs, self.ys, self.zs = mapping_map_to_sphere(self.lon, self.lat)

    def plot_scatter(self):
        '''create 3d scatter plot with ice crystal locations on globe'''
        cmin = 6000
        cmax = max(self.alt)
        cbin = 1000.0

        return go.Scatter3d(
            x=self.xs,
            y=self.ys,
            z=self.zs,
            mode='markers',
            marker=dict(
                cmax=cmax,
                cmin=cmin,
                colorbar=dict(
                    title='Altitude',
                    titleside='right',
                    titlefont=dict(size=16, color='black', family='Courier New'),
                    tickmode='array',
                    ticks='outside',
                    ticktext=list(np.arange(cmin, cmax + cbin, cbin)),
                    tickvals=list(np.arange(cmin, cmax + cbin, cbin)),
                    tickcolor='black',
                    tickfont=dict(size=14, color='black', family='Courier New'),
                ),
                color=self.alt,
                colorscale=Cscale_EQ,
                # hover_data={
                #     'Ice Water Content': True,
                #     'Temperature': True,
                #     'Pressure': True,
                # },
                # custom_data=['Temperature', 'Pressure', 'Ice Water Content'],
                showscale=True,
                opacity=1.0,
            ),
        )
        # fig.update_traces(
        #     mode='markers',
        #     marker_line_width=0,
        #     hovertemplate="<br>".join(
        #         [
        #             "Latitude: %{x}",
        #             "Longitude: %{y}",
        #             "Temperature: %{customdata[0]}",
        #             "Pressure: %{customdata[1]}",
        #             "Ice Water Content: %{customdata[2]}",
        #         ]
        #     ),
        # )
        # return fig
