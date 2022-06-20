'''
File Description
'''

import os
import shutil
import argparse
import numpy as np

import torch

from models import parse_gan_type
from visualization import lerp_matrix, lerp_tensor
from utils import load_generator, analyze_latent_space


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('method_name', type=str, choices=['mddgan', 'sefa', 'both'],
                        help='Name of the method to use when analyzing the '
                        'GAN latent space.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=3,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-C', '--num_components', type=int, default=512,
                        help='Number of semantic boundaries. '
                            '(default: %(default)s)')
    parser.add_argument('-M', '--num_modes', type=int, default=1,
                        help='Number of modes of variation. '
                            '(default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-5.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=5.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=7,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    #parser.add_argument('--gpu_id', type=str, default='0',
    #                    help='GPU(s) to use. (default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Factorize weights.
    generator = load_generator(args.model_name)
    gan_type = parse_gan_type(generator)
    layers, basis, dims = analyze_latent_space('mddgan' if args.method_name == 'both' else args.method_name,
                                        generator,
                                        args.num_components,
                                        args.num_modes,
                                        args.layer_idx)

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare latent codes.
    codes = torch.randn(args.num_samples, generator.z_space_dim).cuda()
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan', 'stylegan2']:
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes,
                                     trunc_psi=args.trunc_psi,
                                     trunc_layers=args.trunc_layers)
    codes = codes.detach().cpu()

    # Visualization : linear interpolation in the GAN latent space.
    distances = np.linspace(args.start_distance, args.end_distance, args.step)

    vis_id = int(input('\n> Choose one of the visualization options below:\n'
        '1. Linear interpolation using the first K directions (columns) discovered\n'
        '2. Linear interpolation using the tensorized multilinear basis of mdd\n'
        '3. Compare MddGAN to SeFa\n'
        'Your option : '))

    assert vis_id in [1, 2, 3], 'Invalid visualization option!'
    
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    if vis_id == 1:
        print(basis.shape)
        lerp_matrix(generator, gan_type, layers, [basis], codes, args.num_samples, distances,
                args.step, args.save_dir)

    elif vis_id == 2:
        print(basis.shape)
        lerp_tensor(generator, layers, basis, dims, codes, args.num_samples,
                distances, args.step, gan_type, args.save_dir)

    else:
        assert args.method_name == 'both'
        if dims is not None:
            _, basis_sefa, _ = analyze_latent_space('sefa',
                    generator,
                    None,
                    None,
                    args.layer_idx)
        lerp_matrix(generator, gan_type, layers, [basis, basis_sefa], codes, args.num_samples,
                distances, args.step, args.save_dir)


if __name__ == '__main__':
    main()
