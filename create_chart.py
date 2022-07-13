'''
File Description
'''

import os
import shutil
import argparse
import numpy as np

import torch

from directions import ATTRIBUTES
from utils import load_generator
from models import parse_gan_type
from visualization import create_semantic_chart


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Create an image chart showcasing some of the discovered attribute vectors.'
        )
    parser.add_argument('model_name', type=str,
                        help='Name of the pre-trained GAN model.')
    parser.add_argument('method_name', type=str, choices=['mddgan', 'sefa', 'interfacegan'],
                        help='Use extracted semantics of the method specified.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the image chart(s). '
                             '(default: %(default)s)')
    parser.add_argument('--semantic_dir', type=str, default='semantics',
                        help='Directory to search for the attribute vectors. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=4,
                        help='Number of samples used for visualization. '
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
    return parser.parse_args()


def main():
    """Main function."""

    # Parse cmd args.
    args = parse_args()

    # Load pre-trained Generator.
    generator = load_generator(args.model_name).cuda()
    gan_type = parse_gan_type(generator)

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Prepare latent codes.
    codes = torch.randn(args.num_samples, generator.z_space_dim, device='cuda')
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan', 'stylegan2']:
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes,
                                     trunc_psi=args.trunc_psi,
                                     trunc_layers=args.trunc_layers)
    codes = codes.cpu()

    # Create image chart.
    attribute_vectors = ATTRIBUTES[args.model_name]
    create_semantic_chart(generator, gan_type, codes, attribute_vectors, args)


if __name__ == '__main__':
    main()
