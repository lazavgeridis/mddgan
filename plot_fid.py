'''
File Description
'''

import os
import argparse
import numpy as np

import torch

from utils import load_generator, parse_indices, get_fake_activations
from visualization import fid_plot, create_comparison_chart
from models import parse_gan_type


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Plot and compare the FID scores '
                                                'of MddGAN - InterFaceGAN - SeFa.')
    parser.add_argument('model_name', type=str,
                        choices=['stylegan_celebahq1024', 'stylegan_ffhq1024'],
                        help='Name of the pre-trained GAN model.')
    parser.add_argument('competing_method_name', type=str, choices=['interfacegan', 'sefa'],
                        help='Name of competing method.')
    parser.add_argument('attribute_name', type=str, choices=['pose', 'gender', 'age', 'smile', 'eyeglasses'],
                        help='Name of the semantic attribute to use when editing.')
    parser.add_argument('--fids_dir', type=str, default='fid_files',
                        help='Path to directory where the fid files are stored '
                             '(default: %(default)s).')
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # assemble file name
    fname = os.path.join(args.fids_dir,
            f'{args.model_name}_{args.competing_method_name}'
            f'_{args.attribute_name}.txt')

    # read file
    magnitudes = []
    competing_fid = []
    mddgan_fid = []
    with open(fname, 'r') as f:
        for line in f:
            values = line.split('\t')
            magnitudes.append(float(values[0]))
            competing_fid.append(float(values[1]))
            mddgan_fid.append(float(values[2].rstrip()))

    # set plot title
    if args.model_name == 'stylegan_celebahq1024':
        title = f'StyleGAN CelebaHQ {args.attribute_name.capitalize()}'
    elif args.model_name == 'stylegan_ffhq1024':
        title = f'StyleGAN FFHQ {args.attribute_name.capitalize()}'

    # plotting function
    fid_plot(title, magnitudes, competing_fid, mddgan_fid, args.competing_method_name)


if __name__ == '__main__':
    main()
