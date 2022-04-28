'''
'''
from tqdm import tqdm

import numpy as np
import random
import torch
import argparse

ATTRIBUTES_CELEBAHQ = {'pose' : '0-3', 'gender' : '0-1', 'age' : '5-7',
                        'eyeglasses' : '0-1', 'smile' : '2-3'}
ATTRIBUTES_FFHQ = {'pose' : '0-6', 'gender' : '2-4', 'age' : '2,4,5,6',
                        'eyeglasses' : '0-2', 'smile' : '3'}

def parse_indices(obj, min_val=None, max_val=None):
    if isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
          numbers = list(map(int, split.split('-')))
          if len(numbers) == 1:
            indices.append(numbers[0])
          elif len(numbers) == 2:
            indices.extend(list(range(numbers[0], numbers[1] + 1)))
          else:
            raise ValueError(f'Unable to parse the input!')

    else:
      raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
      assert isinstance(idx, int)
      if min_val is not None:
        assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
      if max_val is not None:
        assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices


def semantic_edit(G, gan_type, proj_codes, direction, magnitude, layers):
    if gan_type == 'pggan':
        proj_codes += direction * magnitude
        images = G(proj_codes)['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        proj_codes[:, layers, :] += direction * magnitude
        images = G.synthesis(proj_codes)['image']

    return images


def cosine_similarity(args):
    if args.model_name == 'stylegan_celebahq1024':
        ATTRIBUTES = ATTRIBUTES_CELEBAHQ
    elif args.model_name == 'stylegan_ffhq1024':
        ATTRIBUTES = ATTRIBUTES_FFHQ
    
    # load the 5 attribute vectors according to the method name specified
    attr_vectors = []
    #layer_indices = []
    for key, val in ATTRIBUTES.items():
        attr_vector = np.load(f'{args.semantics_dir}/{args.method_name}/{args.model_name}_{key}.npy')
        if attr_vector.ndim == 2:
           attr_vector = np.squeeze(attr_vector, axis=0)
        #layer_idx = parse_indices(val, min_val=0, max_val=G.num_layers - 1)
        attr_vectors.append(attr_vector / np.linalg.norm(attr_vector))
        #layer_indices.append(layer_idx)

    ## set random seed
    #seed = 0
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    results = []
    for idx, attr_vector in enumerate(attr_vectors):
        sims = []
        for attr_vector_d in attr_vectors:
            sims.append(attr_vector.dot(attr_vector_d))

        results.append(sims)

    for scores in results:
        print(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cosine similarity.')
    parser.add_argument('--semantics_dir', required=True, type=str,
                         help='path to dataset directory')
    parser.add_argument('--model_name', required=True, type=str,
                        help='path to directory where checkpoints will be saved')
    parser.add_argument('--method_name', required=True, type=str,
                         help='path to dataset mean, covariance calculated in advance')
    args = parser.parse_args()
    cosine_similarity(args)
