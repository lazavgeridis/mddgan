'''
File Description
'''

import os
import shutil
import argparse
import numpy as np

import torch

from models.inception_net import InceptionV3
from models import parse_gan_type
from visualization import lerp_matrix, lerp_tensor
from utils import load_generator, analyze_latent_space


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate the FID of semantically edited images.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('method_name', type=str, choices=['mddgan', 'sefa', 'both'],
                        help='Name of the method to use when analyzing the '
                        'GAN latent space.')
    parser.add_argument('--attribute_name', type=str, choices=['pose', 'gender', 'age', 'smile', 'eyeglasses'],
                        help='Name of the semantic attribute to use to edit.')
    parser.add_argument('--semantic_dir', type=str, default='semantics',
                        help='Directory where the discovered semantics are stored.'
                             '(default: %(default)s)')
    parser.add_argument('-N', '--fid_sample', type=int, default=50000,
                        help='Number of samples to generate for FID calculation. '
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
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load Generator and Inception networks.
    generator = load_generator(args.model_name)
    gan_type = parse_gan_type(generator)
    inception_model = InceptionV3().eval().requires_grad_(False).cuda()

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



if __name__ == '__main__':
    main()
