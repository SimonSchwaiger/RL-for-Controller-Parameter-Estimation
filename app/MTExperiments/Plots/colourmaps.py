
import numpy as np
import re

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

""" Creates colourmaps and matplotlib colours that match MT style """

def createColourmapToWhite(rgb, reverse=False):
    """ Creates a colourmap that fades from rgb colour to white """
    # Unpack tuple and manually create fades with a resolution of 256
    r, g, b = rgb
    N = 256
    vals = np.ones((N, 4))
    # Distinguish between reverse and non-reverse and invert linspace accordingly
    if reverse:
        vals[:, 0] = np.linspace(1, r/N, N)
        vals[:, 1] = np.linspace(1, g/N, N)
        vals[:, 2] = np.linspace(1, b/N, N)
    else:
        vals[:, 0] = np.linspace(r/N, 1, N)
        vals[:, 1] = np.linspace(g/N, 1, N)
        vals[:, 2] = np.linspace(b/N, 1, N)
    return ListedColormap(vals)

def combineColourmaps(top, bottom):
    """ Combines two colourmaps """
    newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                        bottom(np.linspace(0, 1, 128))))
    return ListedColormap(newcolors, name='OrangeBlue')

## Colours of Plots in thesis
colourScheme1 = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "lightblue": "#8DA3B3",
    "lightgrey": "#CBC0D5",
    "twgrey": "#72777A"
}

colourScheme2 = {
    "darkblue": "#143049",
    "twblue": "#00649C",
    "torquise": "#6BC55C",
    "beige": "#BABA90",
    "yellow": "#E3A049"
}

def hexstring2rgb(colourstring):
    """ Converts hex colours in string form to rgb values """
    i = re.compile('#')
    colourstring = re.sub(i, '', colourstring)
    return tuple(int(colourstring[i:i+2], 16) for i in (0, 2, 4))

## Colourmaps
# Sequential colourmaps from colour to white
CmTWBlue = createColourmapToWhite( hexstring2rgb(colourScheme1["twblue"]) )
CmDarkBlue = createColourmapToWhite( hexstring2rgb(colourScheme1["darkblue"]) )
CmYellow = createColourmapToWhite( hexstring2rgb(colourScheme2["yellow"]) )

# Sequential colourmap from yellow to white to twblue
CmYellowBlue = combineColourmaps(
    createColourmapToWhite( hexstring2rgb(colourScheme1["twblue"]), reverse=False ), # Top = TWBlue
    createColourmapToWhite( hexstring2rgb(colourScheme2["yellow"]), reverse=True  )  # Bottom = Yellow
)

if __name__=="__main__":

    def plot_examples(cms):
        """
        helper function to plot two colormaps
        Source = https://matplotlib.org/3.1.0/tutorials/colors/colormap-manipulation.html
        """
        np.random.seed(19680801)
        data = np.random.randn(30, 30)
        fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
        for [ax, cmap] in zip(axs, cms):
            psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
            fig.colorbar(psm, ax=ax)
        plt.show()

    plot_examples([CmTWBlue, CmYellow])
    plot_examples([CmYellowBlue, CmDarkBlue])




