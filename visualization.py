""" Desc. """

import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.
  
      Args:
          images: A `torch.Tensor` with shape `NCHW` to process.
          min_val: The minimum value of the input tensor. (default: -1.0)
          max_val: The maximum value of the input tensor. (default: 1.0)
  
      Returns:
          A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
      """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(1, 2, 0)
  
    return images


def draw_chart(fig):
    """Draws a chart figure"""

    fig.canvas.draw()
    chart = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    chart = chart.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # crop borders
    nonzero_columns = np.count_nonzero(chart != 255, axis=0)[:, 0] > 0
    chart = chart.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)

    return chart


def semantic_edit(G, layers, gan_type, proj_code, direction, magnitude):
    """Produces an edited image : I' = G(z + εn)"""

    if gan_type == 'pggan':
        proj_code += direction * magnitude
        image = G(proj_code.cuda())['image'].detach().cpu()
    elif gan_type in ['stylegan', 'stylegan2']:
        proj_code[:, layers, :] += direction * magnitude
        image = G.synthesis(proj_code.cuda())['image'].detach().cpu()

    return image


def interpolation(G, layers, gan_type, proj_code, direction, distances):
  '''Creates a row interpolation grid generated using `direction`'''
  images_per_direction = []
  for col_id, d in enumerate(distances, start=1):
    temp_proj_code = torch.clone(proj_code).detach()
    image = semantic_edit(G, layers, gan_type, temp_proj_code, direction, d)
    images_per_direction.append(image.squeeze(0))

  return images_per_direction


def interpolation_chart_col(G, layers, gan_type, basis, proj_code, distances,
        step, directions_per_page, begin, end, title, **kwargs):
    '''Create a pyplot figure with `directions_per_page` interpolation grids centered around `proj_code` vector'''

    # prepare `directions_per_page` single row interpolation grids
    rowgrids_per_page = []
    for i in range(end - begin):
        direction = basis[:, i]
        rowgrids_per_page.append(interpolation(G, layers, gan_type, proj_code, direction, distances)) # create interpolation grid for direction `direction`

    # create a figure with `directions_per_page` + 1 rows
    rows_num = min(directions_per_page, basis.shape[1]) + 1
    fig, axs = plt.subplots(nrows=rows_num, **kwargs) # **kwargs are passed to pyplot.figure()
    fig.suptitle(title)

    # show original image in 1st row
    original_image = semantic_edit(G, layers, gan_type, proj_code, direction, 0).squeeze(0)
    axs[0].axis('off')
    axs[0].imshow(postprocess(original_image))

    # plot each interpolation grid on the corresponing row
    desc = range(begin, end)
    for ax, direction_interp, text in zip(axs[1:], rowgrids_per_page, desc):
        ax.axis('off')
        plt.subplots_adjust(left=0.25)  # setting left=0.2 or lower eliminates whitespace between charts
        ax.imshow(postprocess(make_grid(direction_interp, nrow=step)))
        #ax.text(0, 1, str(text), fontsize='xx-small')

    return fig


def lerp_matrix(G, layers, basis_matrix, proj_codes, n_samples, magnitudes,
        step, gan_type, results_dir, title='', max_columns=45, directions_per_page=15):
    """Linear interpolation using the columns of the basis matrix."""

    assert basis_matrix.ndim == 2
    max_columns = min(max_columns, basis_matrix.shape[1])

    # plot `directions_per_page` attribute vectors for each sample
    for begin in range(0, max_columns, directions_per_page):
        end = min(max_columns, begin + directions_per_page)
        matrix = basis_matrix[:, begin:end]
        charts = []

        # create an interpolation chart for each sample
        for sample_id in tqdm(range(n_samples), desc='Sample', leave=False):
          code = proj_codes[sample_id:sample_id + 1]
          fig = interpolation_chart_col(G, layers, gan_type, matrix, code,
                                        magnitudes, step, directions_per_page, begin, end, title,
                                        dpi=600)

          # draw chart and append it to `charts` list
          charts.append(draw_chart(fig))

          # conserve memory
          fig.clf()
          plt.close(fig)

        # concat charts into a single grid, save the grid
        out_file = os.path.join(results_dir, f'directions_{begin}_{end}.jpg')
        print('Saving chart to ', out_file)
        Image.fromarray(np.hstack(charts)).save(out_file) # concatenate figures column-wise
