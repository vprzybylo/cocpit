'''Create color bar in Matplotlib for 3D globe'''
import matplotlib
import numpy as np


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def MlibCscale_to_Plotly(cbar):
    cmap = matplotlib.cm.get_cmap(cbar)
    rgb = []
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)

    for i in range(0, 255):
        k = matplotlib.colors.colorConverter.to_rgb(cmap(norm(i)))
        rgb.append(k)

    Cscale = matplotlib_to_plotly(cmap, 255)

    return Cscale


cbar = 'jet_r'
Cscale_EQ = MlibCscale_to_Plotly(cbar)
