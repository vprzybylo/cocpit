'''
create interactive topographic globe using plotly

closely follows:
https://towardsdatascience.com/create-interactive-globe-earthquake-plot-in-python-b0b52b646f27
'''

import numpy as np
from netCDF4 import Dataset
from dataclasses import dataclass, field
from typing import List


@dataclass
class TopoMap:
    resolution: float = 1.0
    lon_area: List = field(default_factory=list)
    lat_area: List = field(default_factory=list)
    lon_range: float = field(init=False)
    lat_range: float = field(init=False)
    spacing: float = field(init=False)
    z: float = field(init=False)
    lon_num: float = field(init=False)
    lat_num: float = field(init=False)
    lat: float = field(init=False)
    lon: float = field(init=False)
    topo: float = field(init=False)

    def read_netCDF_globe(self):
        '''Read NetCDF data downloaded from https://www.ngdc.noaa.gov/mgg/global/'''
        data = Dataset("data/ETOPO1_Ice_g_gdal.grd.nosync", "r")
        self.lon_range = data.variables['x_range'][:]
        self.lat_range = data.variables['y_range'][:]
        self.spacing = data.variables['spacing'][:]
        dimension = data.variables['dimension'][:]
        self.z = data.variables['z'][:]
        self.lon_num = dimension[0]
        self.lat_num = dimension[1]

    def mesh_grid(self):
        '''create 2D array'''
        lon_input = np.zeros(self.lon_num)
        lat_input = np.zeros(self.lat_num)
        for i in range(self.lon_num):
            lon_input[i] = self.lon_range[0] + i * self.spacing[0]
        for i in range(self.lat_num):
            lat_input[i] = self.lat_range[0] + i * self.spacing[1]

        self.lon, self.lat = np.meshgrid(lon_input, lat_input)

    def reshape(self):
        '''Convert 2D array from 1D array for z value'''
        self.topo = np.reshape(self.z, (self.lat_num, self.lon_num))

    def skip_for_resolution(self):
        '''Skip the data for resolution'''
        if (self.resolution < self.spacing[0]) | (self.resolution < self.spacing[1]):
            print('Set the highest resolution')
        else:
            skip = int(self.resolution / self.spacing[0])
            self.lon = self.lon[::skip, ::skip]
            self.lat = self.lat[::skip, ::skip]
            self.topo = self.topo[::skip, ::skip]

        self.topo = self.topo[::-1]

    def select_range(self):
        '''Select the range of map'''
        print(
            f'lon area {self.lon_area}, lat area {self.lat_area}, res {self.resolution}'
        )
        range1 = np.where(
            (self.lon >= self.lon_area[0]) & (self.lon <= self.lon_area[1])
        )
        self.lon = self.lon[range1]
        self.lat = self.lat[range1]
        self.topo = self.topo[range1]
        range2 = np.where(
            (self.lat >= self.lat_area[0]) & (self.lat <= self.lat_area[1])
        )
        self.lon = self.lon[range2]
        self.lat = self.lat[range2]
        self.topo = self.topo[range2]

    def convert_2D(self):
        '''Convert 2D again after selecting range'''
        self.lon_num = len(np.unique(self.lon))
        self.lat_num = len(np.unique(self.lat))
        self.lon = np.reshape(self.lon, (self.lat_num, self.lon_num))
        self.lat = np.reshape(self.lat, (self.lat_num, self.lon_num))
        self.topo = np.reshape(self.topo, (self.lat_num, self.lon_num))


def degree2radians(degree):
    '''convert degrees to radians'''
    return degree * np.pi / 180


def mapping_map_to_sphere(lon, lat, radius=1):
    '''maps the points of coords (lon, lat)
    to points onto the sphere of radius radius'''
    lon = np.array(lon, dtype=np.float64)
    lat = np.array(lat, dtype=np.float64)
    lon = degree2radians(lon)
    lat = degree2radians(lat)
    xs = radius * np.cos(lon) * np.cos(lat)
    ys = radius * np.sin(lon) * np.cos(lat)
    zs = radius * np.sin(lat)
    return xs, ys, zs
