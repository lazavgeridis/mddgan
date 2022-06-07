'''
File Description
'''

import os
import shutil
import argparse
import numpy as np

import torch

from models import parse_gan_type
from utils import load_generator, analyze_latent_space


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('method_name', type=str, choices=['mddgan', 'sefa'],
                        help='Name of the method to use when analyzing the '
                        'GAN latent space.')
    parser.add_argument('--save_dir', type=str, default='semantics',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-C', '--num_components', type=int, default=512,
                        help='Number of semantic boundaries. '
                            '(default: %(default)s)')
    parser.add_argument('-M', '--num_modes', type=int, default=1,
                        help='Number of modes of variation. '
                            '(default: %(default)s)')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Factorize weights.
    generator = load_generator(args.model_name)
    gan_type = parse_gan_type(generator)
    layers, basis, dims = analyze_latent_space(args.method_name,
                                               generator,
                                               args.num_components,
                                               args.num_modes,
                                               args.layer_idx)

    semantic_idx = int(input(
                    '> Enter the index of the semantic you wish to save : '))
    attribute_name = input('> Enter a label describing the semantic '
            'attribute : ')

    semantics_dir = os.path.join(args.save_dir, args.method_name)
    os.makedirs(semantics_dir, exist_ok=True)
    np.save(os.path.join(semantics_dir, f'{args.model_name}_{attribute_name}.npy'), basis[:, semantic_idx])


if __name__ == '__main__':
    main()
