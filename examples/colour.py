import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_examples(cms):
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

plot_examples([variint_map_a, variint_map_b])

