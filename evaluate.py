""" """

import os
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from scipy import linalg
#from dataset import get_dataloader, ImageDataset
#from inception_net import InceptionV3


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Borrowed from: https://github.com/mseitzer/pytorch-fid"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def get_activations(path,
                    model,
                    device,
                    image_size,
                    batch_size=32,
                    fid_sample=50000,
                    num_workers=4,
                    mirror_augment=False):
    model.eval()
    mirror = False

    if batch_size > fid_sample:
        print(('Warning: batch size is bigger than the FID sample size. '
               'Setting batch size to sample size'))
        batch_size = fid_sample

    dataloader = get_dataloader(path, image_size, batch_size)
    dataloader = iter(dataloader)
    if mirror_augment:
        mirror = True

    pred_arr = np.empty((fid_sample, 2048))
    num_batches = fid_sample // batch_size
    start_idx = 0

    for batch in tqdm(dataloader, total=num_batches):
        if start_idx + batch_size >= fid_sample:
            break

        assert batch.size(2) == image_size

        if mirror:
            mask = torch.rand(batch_size, 1, 1, 1)
            batch = torch.where(mask < 0.5, batch, batch.flip([3]))
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0]

        ##########################################################################
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
        #########################################################################

    # handle remaining images if any
    remaining = fid_sample - (num_batches * batch_size)
    if remaining > 0:
        batch = next(dataloader)
        if mirror:
            mask = torch.rand(batch_size, 1, 1, 1)
            batch = torch.where(mask < 0.5, batch, batch.flip([3]))
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + remaining] = pred[:remaining]

    assert pred_arr.shape[0] == fid_sample

    return pred_arr


def calculate_activation_statistics(path, model, device, image_size, batch_size=32, fid_sample=50000):
    """Borrowed from: https://github.com/mseitzer/pytorch-fid"""
    activations = get_activations(path, model, device, image_size, batch_size=batch_size, fid_sample=fid_sample)
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def compute_stats_of_path(path, model, device, image_size, batch_size=32, fid_sample=50000):
    """Borrowed from: https://github.com/mseitzer/pytorch-fid"""
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        m, s = calculate_activation_statistics(path,
                                               model, 
                                               device,
                                               image_size,
                                               batch_size=batch_size,
                                               fid_sample=fid_sample)

    return m, s


#def fidk50(precalc_stats_path, generated_images_dir, device, fid_sample_size=50000, batch_size=50, image_size=None):
#    model = InceptionV3().cuda()
#
#    mu_real, sigma_real = compute_stats_of_path(precalc_stats_path, 
#                                                model, 
#                                                device, 
#                                                batch_size=batch_size)
#
#    mu_fake, sigma_fake = compute_stats_of_path(generated_images_dir, 
#                                                model,
#                                                device, 
#                                                batch_size=batch_size, 
#                                                fid_sample_size=fid_sample_size, 
#                                                image_size=image_size)
#
#    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake) 
#
#    return fid


def fid50k(precalc_stats_path, fake_activations):
    """Borrowed from: https://github.com/mseitzer/pytorch-fid
       
       Parameters
       ----------
       precalc_stats_path: str
            file path where the precalculated dataset stats are stored
       fake_activations: numpy array
            inception network activations of generated images
    """
    model, device, image_size = None, None, None
    mu_real, sigma_real = compute_stats_of_path(precalc_stats_path,
                                                model,
                                                device,
                                                image_size)

    mu_fake = np.mean(fake_activations, axis=0)
    sigma_fake = np.cov(fake_activations, rowvar=False)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake) 

    return fid


def get_inception_output(inception_network, batch):
    activations = inception_network(batch)[0]                     # shape : (B, 2048, 1, 1)
    activations = activations.squeeze(3).squeeze(2).cpu().numpy() # shape : (B, 2048)
    assert activations.ndim == 2 and activations.shape[1] == 2048

    return activations
