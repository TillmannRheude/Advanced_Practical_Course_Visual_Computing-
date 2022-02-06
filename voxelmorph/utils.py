""" 
    Documentation

    This file is used for many helper functions which are used in other files. 

"""


""" Imports """
import numpy as np

from matplotlib import image
from matplotlib import pyplot as plt

# third party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable  # plotting


def slices(slices_in,           # the 2D slices
           titles=None,         # list of titles
           cmaps=None,          # list of colormaps
           norms=None,          # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,          # option to plot the images in a grid or a single row
           width=15,            # width in in
           show=True,           # option to actually show the plot (plt.show())
           axes_off=True,
           plot_block=True,     # option to plt.show()
           facecolor=None,
           imshow_args=None):
    '''
    plot a grid of slices (2d images)

    Dalca AV, Guttag J, Sabuncu MR
    Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
    CVPR 2018

    Contact: adalca [at] csail [dot] mit [dot] edu
    License: GPLv3

    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    slices_in = list(map(np.squeeze, slices_in))
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, \
                'each slice has to be 2d or RGB (3 channels)'

    def input_check(inputs, nb_plots, name, default=None):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [default]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps', default='gray')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i],
                          interpolation="nearest", norm=norms[i], **imshow_args[i])

        # colorbars
        # http://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        if do_colorbars:  # and cmaps[i] is not None
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_ax, cax=cax)

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if facecolor is not None:
        fig.set_facecolor(facecolor)

    if show:
        plt.tight_layout()
        plt.show(block=plot_block)

    return (fig, axs)

def flow(slices_in,           # the 2D slices
         titles=None,         # list of titles
         cmaps=None,          # list of colormaps
         width=15,            # width in in
         indexing='ij',       # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
         img_indexing=True,   # whether to match the image view, i.e. flip y axis
         grid=False,          # option to plot the images in a grid or a single row
         show=True,           # option to actually show the plot (plt.show())
         quiver_width=None,
         plot_block=True,  # option to plt.show()
         scale=1):            # note quiver essentially draws quiver length = 1/scale
    '''
    plot a grid of flows (2d+2 images)
    
    Dalca AV, Guttag J, Sabuncu MR
    Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
    CVPR 2018

    Contact: adalca [at] csail [dot] mit [dot] edu
    License: GPLv3

    '''

    # input processing
    nb_plots = len(slices_in)
    for slice_in in slices_in:
        assert len(slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
        assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    assert indexing in ['ij', 'xy']
    slices_in = np.copy(slices_in)  # Since img_indexing, indexing may modify slices_in in memory

    if indexing == 'ij':
        for si, slc in enumerate(slices_in):
            # Make y values negative so y-axis will point down in plot
            slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

    if img_indexing:
        for si, slc in enumerate(slices_in):
            slices_in[si] = np.flipud(slc)  # Flip vertical order of y values

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    scale = input_check(scale, nb_plots, 'scale')

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        u, v = slices_in[i][..., 0], slices_in[i][..., 1]
        colors = np.arctan2(u, v)
        colors[np.isnan(colors)] = 0
        norm = Normalize()
        norm.autoscale(colors)
        if cmaps[i] is None:
            colormap = cm.winter
        else:
            raise Exception("custom cmaps not currently implemented for plt.flow()")

        # show figure
        ax.quiver(u, v,
                  color=colormap(norm(colors).flatten()),
                  angles='xy',
                  units='xy',
                  width=quiver_width,
                  scale=scale[i])
        ax.axis('equal')

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)
    plt.tight_layout()

    if show:
        plt.show(block=plot_block)

    return (fig, axs)

def plot_loss(hist, save_dir):
    plt.style.use('science') # needs latex
    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-', label = "Train Loss")
    if "val_loss" in hist.history: 
        plt.plot(hist.epoch, hist.history["val_loss"], '.-', label = "Val Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc = "upper right")
    plt.title("Loss History")
    plt.savefig(save_dir + ".svg")

def load_img(img_path):
    img = image.imread(img_path)
    return img

def save_image(np_array, save_path):
    plt.imsave(save_path, np_array, cmap = "gray")

def normalize(data):
    data = data.astype("float")/255

    return data