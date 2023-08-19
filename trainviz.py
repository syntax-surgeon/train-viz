'''
############################################################
##### TRAIN-VIZ (Visualizing network training process) #####
#################################### -Siddharth Yadav- #####
'''

# Importing the necessary libraries
from itertools import pairwise
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d as con2d


# Function to generate the probability map given a model
def make_map(model, xlim:tuple, ylim:tuple, map_type="boundary",
             axial_gradation=0.05, square_axis_points=None,
             epoch_num=None, epoch_freq=None):
    '''
    Generates the probability map given by a classification model over a specified domain

    Parameters
    ----------
    model: torch.nn.Module
        A Pytorch FFN model which will classify the points in the domain

    xlim: tuple(float|int)
        The minimum and maximum values of the x-axis in the domain. Specifies the limit
        of x-axis and must be in ascending order.
        Example: (-4,10)

    ylim: tuple(float|int)
        The minimum and maximum values of the y-axis in the domain. Specifies the limit
        of x-axis and must be in ascending order.
        Example: (-8,5)

    map_type: str (default='boundary')
        Specifies the type of probability map to ouput.
        Possible values are:
            'boundary': Shows the boundaries between the predicted classes
            'region': Shows the regions occupied by the predicted classes

    axial_gradation: float|int (default=0.05)
        Specifies the minimum distance between the points considered for domain construction
        along the axes. Smaller values contruct a dense domain will closely situated
        points but may exponentially increase the computation time.

    square_axis_points: int|None (default=None)
        Specifies the number of points in the edge of a square domain.
        If None (default), axial_gradation is used to construct the domain.
        If specified, axial_gradation is ignored.
        This can be useful for scaling axes with significant range differences.

    epoch_num: int|None (default=None)
        The epoch to which the supplied model belongs to (i.e., current epoch).
        This is used together with the 'epoch_freq' parameter to control the intervals
        at which the probability maps are generated.

    epoch_freq: int|None (default=None)
        The inverse frequency of probability map generation given as a multiple.
        This is used together with the 'epoch_num' parameter to control the intervals
        at which the probability maps are generated.
        Example: epoch_freq=2, will check if epoch_num is a multiple of 2. If so, the
        probability map will be generated else a None value will be returned

    Returns
    -------
    prob_map: ndarray
        The class probability map produced by the model given the domain
    '''

    # Checking the validity of epoch_num and epoch_freq parameter combinations
    if not epoch_num and not epoch_freq:
        pass
    elif epoch_num is not None and epoch_freq:
        if epoch_num % epoch_freq != 0:
            return None
    else:
        raise ValueError("Must specify BOTH 'epoch_num' and 'epoch_freq' to utilize the epoch counter functionality")

    # Extracting the minimum and maximum from axes limits
    x0, x1 = xlim
    y0, y1 = ylim

    # Generating a sqaure domain if square_axis_points is specified
    if square_axis_points is not None:
        xs = torch.tensor(np.linspace(x0, x1, square_axis_points))
        ys = torch.tensor(np.linspace(y0, y1, square_axis_points))

    # Generating a domain using the axial_gradation parameter
    else:
        x_points = int((x1-x0)/axial_gradation)
        y_points = int((y1-y0)/axial_gradation)

        xs = torch.tensor(np.linspace(x0, x1, x_points))
        ys = torch.tensor(np.linspace(y0, y1, y_points))

    # Generating the domain points
    xs, ys = torch.meshgrid(xs, ys, indexing='xy')

    plot_data = torch.hstack((xs.flatten().reshape(-1, 1),
                              ys.flatten().reshape(-1, 1))).float()

    # Genrating the probability map of the specified type
    if map_type == "boundary":
        prob_map = torch.max(F.softmax(model(plot_data), dim=1),
                              axis=1).values.detach().reshape(*xs.shape)
    elif map_type == "region":
        prob_map = torch.argmax(
            F.softmax(model(plot_data), dim=1), axis=1).reshape(*xs.shape)
        
        # Smoothing the class region for aesthetic plotting
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])

        kernel = kernel/np.sum(kernel)

        prob_map = con2d(prob_map, kernel, mode="same")
        
    else:
        raise ValueError(
            "Unrecognized value for map_type: possible values are:\n\t'boundary' & 'region'")

    return prob_map


# Function to generate an animation from the probability maps
def plot_maps(maps, save_filepath,
              interpolation_factor=None, interpolation_type="linear",
              colormap="plasma_r", frame_interval=10, dpi=100,
              figsize=None, dark=False, **writer_kwargs):
    '''
    Generates an animation from the supplied probability maps

    Parameters
    ----------
    maps: list|ndarray
        The probability maps from which the animation will the generated

    save_filepath: str
        The filepath where the animation will be saved. The recommended extension is .mp4

    interpolation_factor: int|None (default=None)
        The number of proabability maps to interpolate between supplied maps.
        This can increase the number of maps to compensate for low capture rate during training
        and smoothen the appearance of the animation.
        The final of the maps is given by: (N-1)*interpolation_factor; where N=no.of supplied maps.

    interpolation_type: str|None (default=None)
        The type of interpolation of use. Possible values are: 'linear' & 'geometric'

    colormap: str (default='plasma_r')
        The colormap to use for the animation. Please refer to matplotlib.pyplot.colormaps().

    frame_interval: int (default=10)
        Milliseconds between each consecutive probability maps during animation.
        Small values render fast animations and vice-versa.

    dpi: int (default=100)
        Dots-per-inch of the figure used for animation

    figsize: tuple(int|float) (default=None)
        Width and height of the figure used for animation. If None, the figsize is determined
        automatically by matplotlib.pyplot.figure

    dark: bool (default=False)
        If true, sets a dark background for the animation.

    **writer_kwargs: keyword-arguments
        Arguments given to the animation.save() function to control output
    '''

    # Function to perform interpolation between supplied probability maps
    def interpolate_maps(maps, spacing=10, type='linear'):
        interp_type = {"linear": np.linspace,
                    "geometric": np.geomspace}

        pair_length = len(maps)-1
        map_shape = maps[0].shape
        return np.asarray([interp_type[type](a, b, num=spacing) for a, b in pairwise(maps)]).reshape(pair_length*spacing, *map_shape)

    # Removing any None values from the supplied maps
    maps = np.asarray([x for x in maps if x is not None])

    if interpolation_factor: # If interpolation is specified
        maps = interpolate_maps(maps, spacing=interpolation_factor, type=interpolation_type)
    
    # Building the figure and plot for animation
    if figsize:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig, ax = plt.subplots(dpi=dpi)

    if dark: # If dark is specified
        fig.set_facecolor("black")
    
    # Initializing the image
    im = ax.imshow(maps[0], cmap=colormap, origin='lower')
    ax.axis('off')
    plt.tight_layout()

    def update(frame): # Function to update the frame during animation
        im.set_array(maps[frame])
        im.set_clim(vmin=maps[frame].min(),
                    vmax=maps[frame].max())

    # Generating the animation
    animation = FuncAnimation(fig, update, frames=len(maps), interval=frame_interval)

    # Saving the animation
    animation.save(save_filepath, writer='ffmpeg', **writer_kwargs)
    
    return animation
