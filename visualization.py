""" Desc. """

import os
import torch
import numpy as np

from utils import select_bases, parse_indices, key_to_title
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


def interpolation_chart(G,
                        layers,
                        gan_type,
                        basis,
                        proj_code,
                        distances,
                        step,
                        n_directions,
                        title=None,
                        begin=None,
                        end=None,
                        **kwargs):
    '''Create a pyplot figure with `directions_per_page` interpolation grids centered around `proj_code` vector'''

    # prepare `directions_per_page` single row interpolation grids
    rowgrids_per_page = []
    for i in range(n_directions):
        direction = basis[:, i]
        rowgrids_per_page.append(interpolation(G, layers, gan_type, proj_code, direction, distances)) # create interpolation grid for direction `direction`

    # create a figure with `directions_per_page` + 1 rows
    rows_num = min(n_directions, basis.shape[1]) + 1
    fig, axs = plt.subplots(nrows=rows_num, **kwargs) # **kwargs are passed to pyplot.figure()
    if title is not None:
        fig.suptitle(title)

    # show original image in 1st row
    original_image = semantic_edit(G, layers, gan_type, proj_code, direction, 0).squeeze(0)
    axs[0].axis('off')
    axs[0].imshow(postprocess(original_image))

    # plot each interpolation grid on the corresponing row
    if begin is not None and end is not None:
        desc = range(begin, end)
    else:
        desc = range(n_directions)
    for ax, direction_interp, text in zip(axs[1:], rowgrids_per_page, desc):
        ax.axis('off')
        #plt.subplots_adjust(left=0.25)  # setting left=0.2 or lower eliminates whitespace between charts
        ax.imshow(postprocess(make_grid(direction_interp, nrow=step)))
        ax.text(0, 0.5, str(text),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize='xx-small',
                transform=ax.transAxes)

    return fig


def lerp_matrix(G,
                gan_type,
                layers,
                basis_list,
                proj_codes,
                n_samples,
                magnitudes,
                step,
                results_dir,
                max_columns=45,
                directions_per_page=15):
    """Linear interpolation using the columns of the basis matrix."""

    assert all(basis.ndim == 2 for basis in basis_list)
    max_columns = min(max_columns, basis_list[0].shape[1])

    if len(basis_list) == 2:
        method_names = ['MddGAN', 'SeFa']
        n_samples = 1
    else:
        method_names = [None]

    # plot `directions_per_page` attribute vectors for each sample
    for begin in range(0, max_columns, directions_per_page):
        end = min(max_columns, begin + directions_per_page)
        charts = []

        # create an interpolation chart for each sample
        for sample_id in tqdm(range(n_samples), desc='Sample', leave=False):
          code = proj_codes[sample_id:sample_id + 1]
          for idx, name in enumerate(method_names):
              submatrix = basis_list[idx]
              submatrix = submatrix[:, begin:end]
              fig = interpolation_chart(G, layers, gan_type, submatrix, code,
                  magnitudes, step, end - begin, title=name,
                  begin=begin, end=end, dpi=600,
                  constrained_layout=True)

              # draw chart and append it to `charts` list
              charts.append(draw_chart(fig))

              # conserve memory
              fig.clf()
              plt.close(fig)

        # concat charts into a single grid, save the grid
        out_file = os.path.join(results_dir, f'directions_{begin}_{end}.jpg')
        print('Saving chart to ', out_file)
        Image.fromarray(np.hstack(charts)).save(out_file) # concatenate figures column-wise


def lerp_tensor(G, 
                layers,
                basis,
                basis_dims,
                proj_codes,
                n_samples,
                magnitudes,
                step,
                gan_type,
                results_dir,
                directions_per_page=15,
                n_secondary_bases=3):
    """Short description"""

    # tensorize basis matrix, if needed
    if basis.ndim == 2:
        basis = basis.reshape(basis.shape[0], *basis_dims)

    for primary_mode_idx, primary_mode_dim in enumerate(basis_dims):
        mode_dir = os.path.join(results_dir, f'Mode_{primary_mode_idx + 1}')
        os.makedirs(mode_dir, exist_ok=True)
        for secondary_mode_idx, secondary_mode_dim in enumerate(basis_dims):
            if primary_mode_idx == secondary_mode_idx:
                continue
            for base_idx in range(min(n_secondary_bases, secondary_mode_dim)):

        #for secondary_base_idx in range(n_secondary_bases):
                charts = []
                for sample_id in range(n_samples):
                    code = proj_codes[sample_id:sample_id + 1]
                    #bases, subscript = select_bases(basis, primary_mode_idx, secondary_base_idx, len(basis_dims))
                    bases, subscript = select_bases(basis, primary_mode_idx, secondary_mode_idx, base_idx, len(basis_dims))
                    directions_num = min(directions_per_page, bases.shape[1])

                    # create figure 
                    fig = interpolation_chart(G, layers, gan_type, bases, code,
                            magnitudes, step, directions_num, dpi=600,
                            constrained_layout=True)

                    # draw chart and append it to `charts` list
                    charts.append(draw_chart(fig))

                    # conserve memory
                    fig.clf()
                    plt.close(fig)

                # concat charts into a single grid, save the grid
                out_file = os.path.join(mode_dir, f'B[{subscript}].jpg')
                print('Saving chart to ', out_file)
                Image.fromarray(np.hstack(charts)).save(out_file) # concatenate figures column-wise


def create_attribute_chart(proj_codes,
                           layers,
                           generator,
                           magnitude,
                           gan_type,
                           semantic,
                           attr_name):

    interpolations = []
    for code_idx in range(proj_codes.shape[0]):
        interpolations.append(interpolation(generator, layers, gan_type,
            proj_codes[code_idx:code_idx+1], semantic, [magnitude]))

    assert len(interpolations) == (proj_codes.shape[0])

    fig, axs = plt.subplots(nrows=proj_codes.shape[0], dpi=600, constrained_layout=True)
    fig.suptitle(key_to_title(attr_name))

    for ax, interp in zip(axs, interpolations):
      ax.axis('off')
      ax.imshow(postprocess(interp[0]))

    return fig


def create_semantic_chart(G,
                          gan_type,
                          proj_codes,
                          attr_dict,
                          args,
                          n_samples_per_page=4):

    total_samples = proj_codes.size()[0]
    n_pages = int(total_samples / n_samples_per_page)

    for i in range(n_pages):
        start = i * n_samples_per_page
        end = min(start + n_samples_per_page, total_samples)
        codes = proj_codes[start:end]

        charts = []
        for idx, (key, item) in enumerate(attr_dict.items()):

            print(f'Creating {key} chart...')

            if idx == 0:
                fig = create_attribute_chart(codes,
                                             list(range(G.num_layers)),
                                             G,
                                             0.0,
                                             gan_type,
                                             torch.zeros(G.z_space_dim),
                                             key)

            else:
                # load the corresponding semantic 
                semantic = np.load(f'{args.semantic_dir}/{args.method_name}/{args.model_name}_{key}.npy')

                #
                layer_idx = item[0]
                layers = parse_indices(layer_idx, min_val=0, max_val=G.num_layers - 1)
                magnitude = item[1]

                # create attribute chart
                fig = create_attribute_chart(codes,
                                             layers,
                                             G,
                                             magnitude,
                                             gan_type,
                                             semantic,
                                             key)

            # draw vertical dotted line
            if idx != len(attr_dict) - 1:
                line = plt.Line2D([1.0, 1.0], [0, 1], color="k", linewidth=5, transform=fig.transFigure)
                fig.add_artist(line)

            # draw chart figure
            charts.append(draw_chart(fig))

            # conserve memory
            fig.clf()
            plt.close(fig)

        # save chart
        out_file = os.path.join(args.save_dir, f'{args.method_name}_{args.model_name}_{start}_{end}.jpg')
        print(f'Saving chart to {out_file}\n')
        Image.fromarray(np.hstack(charts)).save(out_file) # concatenate figures column-wis
