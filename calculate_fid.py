'''
File Description
'''

import os
import shutil
import argparse
import numpy as np

import torch

from utils import load_generator, parse_indices, get_fake_activations
from visualization import fid_plot, create_comparison_chart
from models.inception_net import InceptionV3
from models import parse_gan_type
from directions import ATTRIBUTES
from evaluate import fid50k


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Calculate the FID of semantically edited images.')
    parser.add_argument('model_name', type=str,
                        help='Name of the pre-trained model.')
    parser.add_argument('competing_method_name', type=str, choices=['interfacegan', 'sefa'],
                        help='Name of competing method.')
    parser.add_argument('attribute_name', type=str, choices=['pose', 'gender', 'age', 'smile', 'eyeglasses'],
                        help='Name of the semantic attribute to use to edit.')
    parser.add_argument('dataset_stats', type=str, help='path to pre-computed dataset mean '
                                            'and covariance')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Mini-batch size when generating fake images.'
                             '(default: %(default)s)')
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
    generator = load_generator(args.model_name).cuda()
    gan_type = parse_gan_type(generator)
    inception_model = InceptionV3().eval().requires_grad_(False).cuda()

    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load semantic attribute for each method
    mddgan_attr_vector = np.load(f'{args.semantic_dir}/mddgan'
                            f'/{args.model_name}_{args.attribute_name}.npy')
    mddgan_attr_vector = torch.from_numpy(mddgan_attr_vector).cuda()
    competing_attr_vector = np.load(f'{args.semantic_dir}/{args.competing_method_name}'
                            f'/{args.model_name}_{args.attribute_name}.npy')
    competing_attr_vector = torch.from_numpy(competing_attr_vector).cuda()

    # retrieve layers to apply the edit
    model_attributes = ATTRIBUTES[args.model_name]
    layer_idx, _ = model_attributes[args.attribute_name]
    mddgan_layers = parse_indices(layer_idx, min_val=0,
                            max_val=generator.num_layers - 1)

    reverse = False
    text = ['Ours']
    if args.model_name == 'stylegan_celebahq1024' and args.attribute_name in ['gender', 'age']:
        reverse = True
    if args.competing_method_name == 'interfacegan':
        competing_layers = list(range(generator.num_layers))
        text.insert(0, 'InterFaceGAN')
    elif args.competing_method_name == 'sefa':
        competing_layers = mddgan_layers
        text.insert(0, 'SeFa')

    # visualize the effect produced by both semantics
    magnitudes = np.linspace(args.start_distance, args.end_distance, args.step)
    print(magnitudes)
    create_comparison_chart(generator,
                            gan_type,
                            args.trunc_psi,
                            args.trunc_layers,
                            [competing_attr_vector, mddgan_attr_vector],
                            [competing_layers, mddgan_layers],
                            magnitudes,
                            args.step,
                            text,
                            reverse=reverse)

    # create fid comparison plot
    mddgan_fid = []
    competing_fid = []
    for magnitude in magnitudes:
        print(f'Calculating FID for magnitude={magnitude} ...')
        competing_activ, mddgan_activ = get_fake_activations(   generator,
                                                                inception_model,
                                                                [competing_attr_vector, mddgan_attr_vector],
                                                                [competing_layers, mddgan_layers],
                                                                gan_type,
                                                                magnitude,
                                                                args.fid_sample,
                                                                args.trunc_psi,
                                                                args.trunc_layers,
                                                                batch_size=args.batch_size,
                                                                reverse=reverse     )
        competing_fid.append(fid50k(args.dataset_stats, competing_activ))
        mddgan_fid.append(fid50k(args.dataset_stats, mddgan_activ))

    if args.model_name == 'stylegan_celebahq1024':
        title = f'StyleGAN CelebaHQ {args.attribute_name.capitalize()}'
    elif args.model_name == 'stylegan_ffhq1024':
        title = f'StyleGAN FFHQ {args.attribute_name.capitalize()}'

    fid_plot(title, magnitudes, competing_fid, mddgan_fid, args.competing_method_name)


if __name__ == '__main__':
    main()
