"""Utility functions."""

import base64
import os
import copy
import subprocess
import cv2
import scipy
import numpy as np
from tensorly.tenalg import khatri_rao
from tensorly.base import unfold
from sklearn.utils.extmath import randomized_svd

import torch

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type

__all__ = ['postprocess', 'load_generator', 'decompose_weights']

CHECKPOINT_DIR = 'checkpoints'

DIRECTIONS_DICT = {512 : {1 : [512], 2 : [32, 16], 3 : [8, 8, 8],  # 3 : [8, 8, 8], 3 : [16, 8, 4], 3 : [16, 16, 2], 3 : [32, 4, 4]
                          4 : [4, 4, 4, 8], 5 : [4, 4, 4, 4, 2], 6 : [4, 4, 4, 2, 2, 2], 
                          7 : [4, 4, 2, 2, 2, 2, 2]},
                   200 : {1 : [200], 2 : [20, 10], 3 : [5, 4, 10], 4 : [5, 4, 2, 5]},
                   100 : {1 : [100], 2 : [10, 10], 3 : [4, 5, 5], 4 : [2, 2, 5, 5]},
                   50  : {1 : [50],  2 : [10, 5],  3 : [2, 5, 5]},
                   12  : {1 : [12],  2 : [3, 4],   3 : [2, 2, 3], 4 : [1, 1, 1, 12]},
                   8   : {3 : [2, 2, 2], 4 : [2, 2, 2, 1]}
                  }


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`.

    Args:
      array: The input array to convert.

    Returns:
      A `torch.Tensor` with dtype `torch.FloatTensor` on cuda device.
    """
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


def load_generator(model_name):
    """Loads pre-trained generator.

    Args:
        model_name: Name of the model. Should be a key in `models.MODEL_ZOO`.

    Returns:
        A generator, which is a `torch.nn.Module`, with pre-trained weights
            loaded.

    Raises:
        KeyError: If the input `model_name` is not in `models.MODEL_ZOO`.
    """
    if model_name not in MODEL_ZOO:
        raise KeyError(f'Unknown model name `{model_name}`!')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print(f'  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()
    print(f'Finish loading checkpoint.')
    return generator


def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Args:
        obj: The input object to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this value. (default: None)

    Returns:
        A list of integers.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
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


def svd(X):
    '''singular value decomposition'''
    U, s, V_T = scipy.linalg.svd(X)

    return U, scipy.linalg.diagsvd(s, X.shape[0], X.shape[1]), V_T


def truncated_svd(X, t):
    '''truncated svd'''
    U, s, V_T = randomized_svd(X, n_components=t, random_state=0)

    return U, scipy.linalg.diagsvd(s, t, t), V_T


def hosvd(Q):
    '''calculate the best rank-1 approximation of
    tensor Q using truncated HOSVD
    '''
    U = []
    S = Q
    for dim in range(Q.ndim):
        Q_unfold = unfold(Q, dim)
        U_m, _, _ = svd(Q_unfold)
        U.append(U_m[:, 0])
        S = np.tensordot(S, U_m[:, 0].conj().T, axes=([0], [0]))

    return S, U


def tensorize(array, dims, reverse=True):
    if reverse:
        return array.reshape(*reversed(dims))
    else:
        return array.reshape(*dims)


def mdd(X, K, n_directions, n_iter=5):
    """Decomposes the input matrix X into
     a basis matrix B_(1) and variation
     coefficients A^(2), A^(3), ..., A^(M)

     Parameters
     ----------
     X : numpy.ndarray
         data matrix of shape (d, N)
     K : list
         contains the values K_m, where
         m in [2, M].
    """
    assert np.prod(K) == n_directions

    truncated = False
    d, N = X.shape
    if n_directions < d:
      truncated = True

    if truncated:
      U, S, V_T = truncated_svd(X, n_directions)
    else:
      U, S, V_T = svd(X)

    B = U.dot(np.sqrt(S))
    Q = np.sqrt(S).dot(V_T)

    M = len(K) + 1  # clustered modes of variation
    variation_coefs = [[] for _ in range(M - 1)]
    min_error, best_B, best_variation_modes = (1.0, None, None)

    for _ in range(n_iter):
        for i in range(N):
            Si, Ui = hosvd(tensorize(Q[:, i], K))
            sigma = abs(Si)**(1 / (M - 1))  # abs() is needed here after all
            #if Si < 0:
            #  sigma = -sigma
            for m in range(2, M + 1):
                x = sigma * (Ui[M - m - 1]).reshape(-1, 1)
                variation_coefs[m - 2].append(x)

        variation_modes = [np.concatenate(coefs, axis=1) for coefs in variation_coefs]
        out = X.dot(khatri_rao(variation_modes).T)
        if truncated:
            U, S, V_T = truncated_svd(out, n_directions)
        else:
            U, S, V_T = svd(out)
        B = U.dot(V_T)
        Q = B.T.dot(X)

        #convergence condition
        a = np.linalg.norm(X-B.dot(Q), ord='fro')**2
        b = np.linalg.norm(X, ord='fro')**2
        print(a/b, a, b, flush=True)
        if a / b < min_error:
            min_error = a / b
            best_B = np.copy(B)
            best_variation_modes = copy.deepcopy(variation_modes)

        for coefs in variation_coefs:
            coefs.clear()

    return best_B, best_variation_modes


def analyze_latent_space(method, generator, n_components, n_modes, layer_idx='all'):
    """Factorizes the generator weight to get semantics boundaries.

    Args:
        generator: Generator to factorize.
        layer_idx: Indices of layers to interpret, especially for StyleGAN and
            StyleGAN2. (default: `all`)

    Returns:
        A tuple of (layers_to_interpret, semantic_boundaries, eigen_values).

    Raises:
        ValueError: If the generator type is not supported.
    """
    # Get GAN type.
    gan_type = parse_gan_type(generator)

    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
        else:
            layers = parse_indices(layer_idx,
                                   min_val=0,
                                   max_val=generator.num_layers - 1)

    # Factorize semantics from weight.
    weights = []
    for idx in layers:
        layer_name = f'layer{idx}'
        if gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            layer_name = f'output{idx // 2}'
        if gan_type == 'pggan':
            weight = generator.__getattr__(layer_name).weight
            weight = weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1)
        elif gan_type in ['stylegan', 'stylegan2']:
            weight = generator.synthesis.__getattr__(layer_name).style.weight.T
        weights.append(weight.cpu().detach().numpy())
    weight = np.concatenate(weights, axis=1).astype(np.float32)
    
    # multilinear matrix decomposition
    if method == 'mddgan':
        modes_dict = DIRECTIONS_DICT[n_components]
        K = modes_dict[n_modes]
        basis, _ = mdd(weight, K, n_components, n_iter=5)
    # semantic factorization
    elif method == 'sefa':
        weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
        _, basis = np.linalg.eig(weight.dot(weight.T))
        K = None

    return layers, basis, K


def select_bases(basis_tensor, primary_mode_idx, secondary_mode_idx, base_idx, n_modes):
    """ desc. """
    # basis tensor has shape : (d, K2, K3)
    if n_modes == 2:
        if primary_mode_idx == 0:   # (:, :, idx)
            bases = basis_tensor[:, :, base_idx]
            subscript = f':, :, {base_idx}'
        elif primary_mode_idx == 1: # (:, idx, :)
            bases = basis_tensor[:, base_idx, :]
            subscript = f':, {base_idx}, :'

    # basis tensor has shape : (d, K2, K3, K4)
    elif n_modes == 3:
        if primary_mode_idx == 0:
            if secondary_mode_idx == 1:     # (:, :, idx, 0)
                bases = basis_tensor[:, :, base_idx, 0]
                subscript = f':, :, {base_idx}, 0'
            elif secondary_mode_idx == 2:   # (:, :, 0, idx)
                bases = basis_tensor[:, :, 0, base_idx]
                subscript = f':, :, 0, {base_idx}'
        elif primary_mode_idx == 1:
            if secondary_mode_idx == 0:     # (:, idx, :, 0)
                bases = basis_tensor[:, base_idx, :, 0]
                subscript = f':, {base_idx}, :, 0'
            elif secondary_mode_idx == 2:   # (:, 0, :, idx)
                bases = basis_tensor[:, 0, :, base_idx]
                subscript = f':, 0, :, {base_idx}'
        elif primary_mode_idx == 2:
            if secondary_mode_idx == 0:     # (:, idx, 0, :)
                bases = basis_tensor[:, base_idx, 0, :]
                subscript = f':, {base_idx}, 0, :'
            elif secondary_mode_idx == 1:   # (:, 0, idx, :)
                bases = basis_tensor[:, 0, base_idx, :]
                subscript = f':, 0, {base_idx}, :'

    # basis tensor has shape : (d, K2, K3, K4, K5)
    elif n_modes == 4:
        pass

    return bases, subscript
