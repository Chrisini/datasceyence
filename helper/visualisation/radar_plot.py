import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

import seaborn as sns

import glob


import pandas as pd

import os

import visualisation.colour


def plot_all_filters(data, theta, colours, concept_labels, save_path=""): # only works for 4 ...
    
    fig, axs = plt.subplots(figsize=(9, 9), nrows=2, ncols=2,
                        subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    
    # Plot the n filters
    for ax, (title, filter_data) in zip(axs.flat, data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(filter_data, colours):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.05, label='_nolegend_')
        ax.set_varlabels(concept_labels)

    # add legend relative to top-left plot
    legend = axs[0, 0].legend(disease_labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Filter-wise disease-concept matching', # (cluster afterwards!!!)
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.savefig(os.path.join(save_path, "radar.png"), bbox_inches='tight') 
    plt.close()
    
def plot_each_filter(data, theta, make_pretty=False, colours=None, concept_labels=None, save_path="", xlim_value=250):
    
    styles = ["-", "--", ]
    
    # Plot the n filters
    for title, filter_data in data:
        
        try:
            fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                            subplot_kw=dict(projection='radar'))
            fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
        
            if make_pretty:
                ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                             horizontalalignment='center', verticalalignment='center')
            for i_length, (d, color) in enumerate(zip(filter_data, colours)):
                plt.plot(theta, d, color=color, linewidth=5, linestyle=(i_length*5, (5, (len(filter_data)*5)-i_length )))  # Custom dashed line
                #plt.plot(x, y, linestyle=(5, (5, 15)))
                #plt.plot(x, y, linestyle=(10, (5, 15)))
                #plt.plot(x, y, linestyle=(15, (5, 15)))
                # ax.plot(theta, d, color=color, linewidth=5, alpha=0.6, linestyle='-')
                # ax.fill(theta, d, facecolor=color, alpha=0.05, label='_nolegend_')
            ax.set_varlabels(concept_labels)

            if make_pretty:
                # add legend relative to top-left plot
                _ = ax.legend(disease_labels, loc=(0.9, .95),
                                          labelspacing=0.1, fontsize='small')

            plt.ylim(0, xlim_value)
            plt.savefig(os.path.join(save_path, f'''radar_{title}.png'''), bbox_inches='tight')    
            plt.close()    
        except Exception as E:
            print("RADAR INFO: probably not enough data")
            print(E)
            
            
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta




def get_csv_data(path):
    
    # this is actually median
    med_and_std_paths = glob.glob(os.path.join(path,"final_plots/mean_and_std_*.csv"))

    

    df = pd.read_csv(med_and_std_paths[0])

    #disease_labels = ('AMD', 'nvAMD', 'DR', 'normal')
    disease_labels = df['Disease label'].unique()
    area_labels = df['region'].unique()

    data = [list(area_labels)]

    for p in med_and_std_paths:
        # the rows are the diseases, the columns are the areas

        filter_id = p.split("std_")[1].split(".")[0]
        
        df = pd.read_csv(p)


        #df[df['disease'] == 'A']['texture'].reset_index(drop=True)

        list_list = []
        for disease in disease_labels:
            list_list.append(list(df[df['Disease label'] == disease]['median']))

        #print()
        #list_list = list(df["median"])


        data.append((filter_id, list_list)) # list_list

    return data, disease_labels
        
    

def example_data():
    
    # filter wise data can also be clustered afterwards
    # 'others', 'normal', 'AMD', 'DR', 'CSC', 'nvAMD', 'RVO'
    disease_labels = ('AMD', 'nvAMD', 'DR', 'normal') # , 'CSC', 'RVO')
    
    data = [
        ["something", "drusen", "intra-fluid", "sub-fluid", "neovasc", "hyperref", "anorysm", "this", "that"],
        ('0_2_3', [ # 5 diseases, n concepts
            [0.88, 0.01, 0.03, 0.03, 0.00, 0.06, 0.01, 0.00, 0.00], 
            [0.07, 0.95, 0.04, 0.05, 0.00, 0.02, 0.01, 0.00, 0.00],
            [0.01, 0.02, 0.85, 0.19, 0.05, 0.10, 0.00, 0.00, 0.00],
            [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.00, 0.00],
            #[0.01, 0.01, 0.02, 0.71, 0.74, 0.70, 0.00, 0.00, 0.00]
        ]),
        ('0_7_3', [
            [0.88, 0.02, 0.02, 0.02, 0.00, 0.05, 0.00, 0.05, 0.00],
            [0.08, 0.94, 0.04, 0.02, 0.00, 0.01, 0.12, 0.04, 0.00],
            [0.01, 0.01, 0.79, 0.10, 0.00, 0.05, 0.00, 0.31, 0.00],
            [0.00, 0.02, 0.03, 0.38, 0.31, 0.31, 0.00, 0.59, 0.00],
            #[0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.00, 0.00]
        ]),
        ('14_13_3', [
            [0.89, 0.01, 0.07, 0.00, 0.00, 0.05, 0.00, 0.00, 0.03],
            [0.07, 0.95, 0.05, 0.04, 0.00, 0.02, 0.12, 0.00, 0.00],
            [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.00, 0.00, 0.00],
            [0.01, 0.03, 0.00, 0.32, 0.29, 0.27, 0.00, 0.00, 0.95],
            #[0.02, 0.00, 0.03, 0.37, 0.56, 0.47, 0.87, 0.00, 0.00]
        ]),
        ('2_1_1', [
            [0.87, 0.01, 0.08, 0.00, 0.00, 0.04, 0.00, 0.00, 0.01],
            [0.09, 0.95, 0.02, 0.03, 0.50, 0.01, 0.13, 0.06, 0.00],
            [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.00, 0.50, 0.00],
            [0.01, 0.03, 0.00, 0.28, 0.24, 0.23, 0.00, 0.44, 0.88],
            #[0.02, 0.00, 0.18, 0.45, 0.64, 0.55, 0.86, 0.00, 0.16]
        ])
    ]
    
    return data, disease_labels