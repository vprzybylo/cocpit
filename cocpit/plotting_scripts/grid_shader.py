'''
add alternating background color to matplotlib figure to distinguish classification report groups
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class GridShader:
    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="x")
        self.cid = self.ax.callbacks.connect('xlim_changed', self.shade)
        self.shade()

    def clear(self):
        for span in self.spans:
            span.remove()

    def shade(self, evt=None):
        self.clear()
        xticks = self.ax.get_xticks()
        xlim = self.ax.get_xlim()
        xticks = xticks[(xticks > xlim[0]) & (xticks < xlim[-1])]
        xticks = np.arange(xlim[0], xlim[1], 1)
        locs = np.concatenate(([[xlim[0]], xticks, [xlim[-1]]]))
        start = locs[1 - int(self.sf) :: 2]
        end = locs[2 - int(self.sf) :: 2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axvspan(s, e, zorder=0, **self.kw))
