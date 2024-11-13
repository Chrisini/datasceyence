import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns


def plot_example(cmap):
    """
    helper function to plot one colormap
    """
    np.random.seed(19680801)
    data = np.random.randn(5, 5)
    
    psm = plt.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
    plt.colorbar(psm)
    plt.show()

def plot_two_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(5, 5)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 230/256, N)
vals[:, 1] = np.linspace(190/256, 140/256, N)
vals[:, 2] = np.linspace(190/256, 190/256, N)
variint_map_a = ListedColormap(vals, name='variint_map_a')


N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 231/256, N)
vals[:, 1] = np.linspace(118/256, 238/256, N)
vals[:, 2] = np.linspace(118/256, 238/256, N)
variint_map_1 = ListedColormap(vals, name='variint_map_1')

blue_light = variint_map_1.colors[160][0:3]
blue_dark = variint_map_1.colors[90][0:3]

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(243/256, 199/256, N)
vals[:, 1] = np.linspace(238/256, 21/256, N)
vals[:, 2] = np.linspace(237/256, 133/256, N)
variint_map_2 = ListedColormap(vals, name='variint_map_2')

red_light = variint_map_2.colors[90][0:3]
red_dark = variint_map_2.colors[160][0:3]

# map
top = cm.get_cmap(variint_map_1, 128)
bottom = cm.get_cmap(variint_map_2, 128)
newcolors = np.vstack((top(np.linspace(0, 0.8, 128)),
                       bottom(np.linspace(0.2, 1, 128))))
variint_map_b = ListedColormap(newcolors, name='variint_map_b')

sns_paired = sns.color_palette("Paired")
pal_list = [sns_paired[8], sns_paired[9], sns_paired[0], sns_paired[1], sns_paired[2], sns_paired[3]]
purple_blue_green = ListedColormap(pal_list, "purple_blue_green")

pal_list = [sns_paired[3], sns_paired[8], sns_paired[9], sns_paired[0]]
normal_amd_cnv_dr = ListedColormap(pal_list, "normal_amd_cnv_dr")
normal_amd_cnv_dr_classes = ["normal", "amd", "cnv", "dr"]

pal_list = [sns_paired[9], sns_paired[0], sns_paired[8], sns_paired[3]]
cnv_dr_amd_normal = ListedColormap(pal_list, "cnv_dr_amd_normal")
cnv_dr_amd_normal_classes = ["cnv", "dr", "amd", "normal"]

