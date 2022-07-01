"""Utility functions."""

import os
import copy
import subprocess
import scipy
import torch
import numpy as np
from tqdm import tqdm
from math import ceil
from tensorly.tenalg import khatri_rao
from tensorly.base import unfold
from sklearn.utils.extmath import randomized_svd

from models import MODEL_ZOO
from models import build_generator
from evaluate import get_inception_output


CHECKPOINT_DIR = 'checkpoints'
#CHECKPOINT_DIR = '/content/gdrive/My Drive/Discovering_Modes_of_Variation/checkpoints'

DIRECTIONS_DICT = {512 : {1 : [512], 2 : [32, 16], 3 : [8, 8, 8],  # 3 : [8, 8, 8], 3 : [16, 8, 4], 3 : [16, 16, 2], 3 : [32, 4, 4]
                          4 : [4, 4, 4, 8], 5 : [4, 4, 4, 4, 2], 6 : [4, 4, 4, 2, 2, 2], 
                          7 : [4, 4, 2, 2, 2, 2, 2]},
                   200 : {1 : [200], 2 : [20, 10], 3 : [5, 4, 10], 4 : [5, 4, 2, 5]},
                   100 : {1 : [100], 2 : [10, 10], 3 : [4, 5, 5], 4 : [2, 2, 5, 5]},
                   50  : {1 : [50],  2 : [10, 5],  3 : [2, 5, 5]},
                   12  : {1 : [12],  2 : [3, 4],   3 : [2, 2, 3], 4 : [1, 1, 1, 12]},
                   8   : {3 : [2, 2, 2], 4 : [2, 2, 2, 1]}
                  }


def load_generator(model_name):
    """
    Builds selected generator model and loads the pre-trained weight.

    Parameters
    ----------
    model_name : str
        Name of the GAN model. Should be a key in `models.MODEL_ZOO`.

    Returns
    -------
    torch.nn.Module
                    A generator network in evaluation mode with autograd
                    turned off, and with pre-trained weights loaded.

    Raises
    ------
    KeyError
        If the input `model_name` is not in `models.MODEL_ZOO`.

    """
    if model_name not in MODEL_ZOO:
        raise KeyError(f'Unknown model name `{model_name}`!')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url')  # URL to download model if needed.

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print('Finish building generator.')

    # Load pre-trained weights.
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pth')
    print(f'Loading checkpoint from `{checkpoint_path}` ...')
    if not os.path.exists(checkpoint_path):
        print(f'  Downloading checkpoint from `{url}` ...')
        subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
        print('  Finish downloading checkpoint.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'generator_smooth' in checkpoint:
        generator.load_state_dict(checkpoint['generator_smooth'])
    else:
        generator.load_state_dict(checkpoint['generator'])
    generator = generator.eval().requires_grad_(False)
    print('Finish loading checkpoint.')
    return generator


def parse_indices(obj, min_val=None, max_val=None):
    """
    Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Parameters
    ----------
    obj : list or tuple or str
        The input object to parse indices from.
    min_val : int
        If not `None`, this function will check that all indices are
        equal to or larger than this value. (default: None)
    max_val : int
        If not `None`, this function will check that all indices are
        equal to or smaller than this value. (default: None)

    Returns
    -------
    list of int
                A list of integers representing layer indices.

    Raises
    ------
    ValueError
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
                raise ValueError('Unable to parse the input!')

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
    """Singular value decomposition."""
    U, s, V_T = scipy.linalg.svd(X)

    return U, scipy.linalg.diagsvd(s, X.shape[0], X.shape[1]), V_T


def truncated_svd(X, t):
    """Truncated svd."""
    U, s, V_T = randomized_svd(X, n_components=t, random_state=0)

    return U, scipy.linalg.diagsvd(s, t, t), V_T


def hosvd(Q):
    """
    Truncated higher-order svd.


    Calculate the best rank-1 approximation of tensor `Q`
    using truncated HOSVD.

    Parameters
    ----------
    Q : numpy.ndarray 
        desc.

    Returns
    -------
    numpy.ndarray
                  desc.
    numpy.ndarray
                  desc.

    """
    U = []
    S = Q
    for dim in range(Q.ndim):
        Q_unfold = unfold(Q, dim)
        U_m, _, _ = svd(Q_unfold)
        U.append(U_m[:, 0])
        S = np.tensordot(S, U_m[:, 0].conj().T, axes=([0], [0]))

    return S, U


def tensorize(array, dims, reverse=True):
    """
    Rearrange the elements of vector or matrix `array`
    into a tensor.

    For example, if vector q has dimension d and
    d = K_2 * K_3 * ... * K_M, convert q to tensor
    Q of shape (K_2, K_3, ..., K_M).

    Parameters
    ----------
    array : numpy.ndarray
        vector or matrix to tensorize.
    dims : list of int
        stores the dimensions K_2, K_3, ..., K_M.
    reverse : bool
        whether the output tensor shape will be
        (K_2, K_3, ..., K_M) or (K_M, K_{M-1}, ..., K_2)
        (the default is True, so the 2nd shape).

    Returns
    -------
    numpy.ndarray
                  tensor containing exactly the same elements
                  as `array`.

    """
    if reverse:
        return array.reshape(*reversed(dims))
    else:
        return array.reshape(*dims)


def mdd(X, K, n_directions, n_iter=5):
    """
    Unsupervised multilinear data decomposition.

    Decomposes the input matrix `X` into a basis
    matrix B_(1) and variation coefficients
    A^(2), A^(3), ..., A^(M).

    Parameters
    ----------
    X : numpy.ndarray
        data matrix of shape (d, N).
    K : list of int
        contains the dimensions K_m, where m \in
        [2, M]. The product of these values must
        be equal to `n_directions`.
    n_directions : int
        number of discovered directions.
        if `n_directions` < d, use truncated
        svd.
    n_iter : int
        how many iterations of the algorithm
        to execute (the default is 5).

    Returns
    -------
    numpy.ndarray
                  mode-1 matricization, B_(1), of the
                  multilinear basis \mathcal{B}. It has
                  shape (d, `n_directions`).
    list of numpy.ndarray
                  variation coefficients A^(m), where
                  m \in [2, M]. Coefficient A^(m) has
                  shape (K_m, N).

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


def analyze_latent_space(method, generator, gan_type, n_components, n_modes, layer_range='all'):
    """
    Analyze the weight of the pre-trained generator
    and extract meaningful semantics.

    Parameters
    ----------
    method : {'sefa', 'mddgan'}
        name of the method to use when analyzing the
        weights of the pre-trained generator.
    generator : torch.nn.module
        generator network with pre-trained weights
        from which semantic information will be
        extracted.
    gan_type : {'pggan', 'stylegan', 'stylegan2'}
        GAN model type.
    n_components : int
        number of directions to discover. Used exclusively
        for mddgan.
    n_modes : int
        number of modes of variation. Used exclusively
        for mddgan.
    layer_range : 3 different types

    Returns
    -------

    """
    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2']:
        if layer_range == 'all':
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
    """
    desc.

    Parameters
    ----------

    Returns
    -------

    """
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


def semantic_edit(G, layers, gan_type, proj_code, direction, magnitude):
    """
    Produces an edited image : I' = G(z + εn).

    Parameters
    ----------

    Returns
    -------

    """
 
    if gan_type == 'pggan':
        proj_code += direction * magnitude
        image = G(proj_code.cuda())['image'].detach().cpu()
    elif gan_type in ['stylegan', 'stylegan2']:
        proj_code[:, layers, :] += direction * magnitude
        image = G.synthesis(proj_code.cuda())['image'].detach().cpu()

    return image


def get_fake_activations(G,
                        inception,
                        semantics,
                        layers,
                        gan_type,
                        magnitude,
                        fid_size,
                        trunc_psi,
                        trunc_layers,
                        batch_size=8,
                        reverse=False):
    """
    Collect `fid_size` fake activations by semantically editing
    synthesized images and passing them through the Inception network.
    Do this for both the MddGAN and competing method (InterFaceGAN/SeFa)
    semantics.

    Parameters
    ----------

    Returns
    -------

    """

    mddgan_inception_activations    = []
    competing_inception_activations = []
    print("Getting fake activations...\n")

    for _ in tqdm(range(ceil(fid_size / batch_size))):

        # generate batch of fake images
        z_vectors = torch.randn(batch_size, G.z_space_dim, device='cuda')

        # 
        if gan_type == 'pggan':
            codes = G.layer0.pixel_norm(z_vectors)
        elif gan_type in ['stylegan', 'stylegan2']:
            codes = G.mapping(z_vectors)['w']
            codes = G.truncation(codes, trunc_psi=trunc_psi, trunc_layers=trunc_layers)
        
        # edit using the selected attribute vector (`semantic`) and magnitude
        competing_edited_images = semantic_edit(G,
                                                layers[0],
                                                gan_type,
                                                codes,
                                                semantics[0],
                                                magnitude)
        mddgan_edited_images = semantic_edit(G,
                                            layers[1],
                                            gan_type,
                                            codes,
                                            semantics[1],
                                            -1.0 * magnitude if reverse else magnitude)

        # get inception network output for edited images
        competing_inception_activations.append(get_inception_output(inception, competing_edited_images.cuda()))
        mddgan_inception_activations.append(get_inception_output(inception, mddgan_edited_images.cuda()))

    competing_fake_activations = np.concatenate(competing_inception_activations, axis=0)[:fid_size, :]
    mddgan_fake_activations = np.concatenate(mddgan_inception_activations, axis=0)[:fid_size, :]
    assert mddgan_fake_activations.ndim == 2 and mddgan_fake_activations.shape[0] == fid_size

    return competing_fake_activations, mddgan_fake_activations


def key_to_title(attr_key):
    """."""
    title = attr_key.split('_')
    title = ' '.join(word.capitalize() for word in title)
    return title
