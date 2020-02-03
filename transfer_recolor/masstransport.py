# Copyright (c) 2017 Lightricks. All rights reserved.

import math
from copy import deepcopy

import cv2.ximgproc # guidedFilter
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.linalg
import scipy.signal
import skimage.transform

# from utils import calc_rescale_factor
# test

# SMOOTH_WINDOW = scipy.signal.gaussian(7, 1)
# SMOOTH_WINDOW /= SMOOTH_WINDOW.sum()
SMOOTH_WINDOW = None


def calc_rescale_factor(size, max_long_axis_after_rescale=1200.):
    """
    :param max_long_axis_after_rescale:
    :param size:
    :return:
    """
    long_axis = max(size)
    return min(1., 2. ** -np.ceil(np.log2(long_axis / max_long_axis_after_rescale)))


def make_rgb_grid(grid_density=16):
    """
    Generate a fixed initial regularly spaced RGB grid as a LUT
    :param grid_density: Number of points in the grid in each dimension
    :return: The RGB grid
    """
    single_axis = np.linspace(0., 1., grid_density)
    return np.stack(
        np.meshgrid(
            single_axis,
            single_axis,
            single_axis,
            indexing='ij')
        , -1
    ).reshape(-1, 3)


def histogram_transfer_1d_fit(in_vals, ref_vals, n_bins=100, smooth_window=None):
    """
    Learn the parameters (model) of a 1D histogram transfer from given samples
    :param in_vals: Input values for 1D histogram transfer
    :param ref_vals: Reference values for 1D histogram transfer
    :param n_bins: Number of bins to use when calculating histogram
    :return: Histogram transfer model: two CDF functions sampled at the bin edges, bin edges
    """
    val_range = np.min(np.concatenate([in_vals, ref_vals])), np.max(
        np.concatenate([in_vals, ref_vals]))
    in_hist, bins = np.histogram(in_vals, n_bins, range=val_range, density=True)
    ref_hist, bins = np.histogram(ref_vals, n_bins, range=val_range, density=True)

    if smooth_window is not None:
        in_hist = scipy.signal.convolve(in_hist, smooth_window, mode='same')
        ref_hist = scipy.signal.convolve(ref_hist, smooth_window, mode='same')

    in_cdf = in_hist.cumsum()
    ref_cdf = ref_hist.cumsum()

    return in_cdf, ref_cdf, bins


def histogram_transfer_1d_apply(vals, model):
    """
    Apply the parameters (model) of a 1D histogram transfer to given samples, not necessarily
    the same samples as the ones used to learn the mdoel
    :param vals: Values on which to apply the 1D histogram transfer
    :param model: Model by which to apply the 1D histogram transfer (learned seprately)
    :return: Values after the 1D histogram transfer
    """
    in_cdf, ref_cdf, bins = model
    t_vals = np.interp(vals, bins[:-1], in_cdf)

    # assert np.all(np.diff(ref_cdf) > 0)
    if (ref_cdf == 1.).sum() > 0:
        idx_max = np.where(ref_cdf == 1.)[0][0]
        out_vals = np.interp(t_vals, ref_cdf[:idx_max + 1], bins[:idx_max + 1])
    else:
        out_vals = np.interp(t_vals, ref_cdf, bins[:-1])

    return out_vals


def regularize_image(image, n_dimensions_to_regularize=3, n_regularization_copies=3, regularization_noise_std=0.1):
    """
    Apply stochastic sampling (by adding additive white Gaussian noise) around given image
    and create multiple copies of the image in order to learn a smoother mass transport
    :param image: Image to regulaarlize
    :param n_dimensions_to_regularize: How many of the layers in the image to apply regularization on
    :return: 4-dimensional array of images, all copies but the first are regularlized
    """
    samples = np.tile(np.expand_dims(image, 0), (n_regularization_copies, 1, 1, 1))
    for k in range(1, n_regularization_copies):  # keep the original untouched!
        # add noise only to the color components!
        samples[k, :, :, :n_dimensions_to_regularize] += np.random.normal(
            0.,
            regularization_noise_std,
            samples[
            k,
            :, :, :n_dimensions_to_regularize].shape
        )
    samples = samples.reshape((-1, image.shape[-1]))
    return samples


def transport_single_iteration(i_iter, samples_in, samples_ref,
                               samples_in_mask, samples_ref_mask, rgb_grid=None, alpha=0.5,
                               n_bins=100, converge_threshold=0., smooth_window=None):
    """
    Single iteration of the Sliced Wasserstein mass transport algorithm
    :param i_iter: Number of iteration
    :param samples_in: Input samples
    :param samples_ref: Reference samples
    :param samples_in_mask: Mask selecting which input samples affect the transport
    :param samples_ref_mask: Mask selecting which refernece samples affect the transport
    :param rgb_grid: Optional RGB grid of values (LUT) to apply the transport on as well
    :return: Modified input samples, RGB grid after applying the transport, distance between
            histograms
    """
    n_dims = samples_in.shape[-1]
    use_grid = False#rgb_grid is not None
    distance = 0.
    # distance = compare_hists([
    #     samples_in[samples_in_mask > 0.5],
    #     samples_ref[samples_ref_mask > 0.5]
    #     ],
    #     draw=i_iter % 5 == 0,
    #     n_bins=n_bins, smooth_window=smooth_window)
    #
    # if distance < converge_threshold:
    #     compare_hists([samples_in[samples_in_mask > 0.5], samples_ref[samples_ref_mask > 0.5]],
    #                   draw=True,
    #                   n_bins=n_bins, smooth_window=smooth_window)
    #     return samples_in, rgb_grid, distance

    coords = np.random.uniform(-1., 1., (n_dims, n_dims))
    coords = scipy.linalg.orth(coords)
    # coords = np.load('rotations.3d.20.npy')[i_iter]

    samples_in_tilde = np.zeros_like(samples_in)
    if use_grid:
        rgb_tilde = np.zeros_like(rgb_grid)
    for d in range(n_dims):
        samples_in_proj = np.dot(samples_in, coords[d])
        samples_ref_proj = np.dot(samples_ref, coords[d])
        # print(np.unique(samples_ref_mask))
        histogram_transfer_model = histogram_transfer_1d_fit(
            samples_in_proj[samples_in_mask > 0.5],
            samples_ref_proj[samples_ref_mask > 0.5],
            n_bins,
            smooth_window
        )
        samples_in_proj_after = histogram_transfer_1d_apply(samples_in_proj,
                                                            histogram_transfer_model)
        samples_in_tilde += np.outer(samples_in_proj_after, coords[d])
        if use_grid:
            rgb_grid_proj = np.dot(rgb_grid, coords[d])
            rgb_grid_proj_after = histogram_transfer_1d_apply(rgb_grid_proj,
                                                              histogram_transfer_model)
            rgb_tilde += np.outer(rgb_grid_proj_after, coords[d])
    samples_in = (1. - alpha) * samples_in + alpha * samples_in_tilde
    if use_grid:
        rgb_grid = (1. - alpha) * rgb_grid + alpha * rgb_tilde
    return samples_in, rgb_grid, distance


def mask2binary(mask, T=0.5):
    bin_mask = deepcopy(mask)
    bin_mask[bin_mask < T] = 0
    bin_mask[bin_mask > 0] = 1.
    return bin_mask


def compare_hists(samples, draw=False, n_bins=100, smooth_window=None):
    """
    Compate the histograms of the provided samples using a simple L2 distance between the PDFs
    of their 1D projections onto the standard (elementary) axes
    :param samples: Two sets of samples
    :param draw: Whether or not to also draw the 1D PDFs calcualted
    :return: The L2 distance between the PDFs (on all dimensions combined)
    """
    err = 0.
    if draw:
        plt.figure()
    val_range = np.min(np.concatenate(samples)), np.max(np.concatenate(samples))
    n_dims = samples[0].shape[-1]
    for d in range(n_dims):
        cdfs = [None] * len(samples)
        pdfs = [None] * len(samples)
        for i_sample, sample in enumerate(samples):
            hist, bins = np.histogram(sample[:, d], bins=n_bins, range=val_range, density=True)
            if smooth_window is not None:
                hist = scipy.signal.convolve(hist, smooth_window, mode='same')
            pdfs[i_sample] = hist
            cdfs[i_sample] = hist.cumsum()
            if draw:
                plt.plot(bins[:-1], pdfs[i_sample],
                         ['-', ':', '.-'][i_sample], color=['r', 'g', 'b', 'c', 'm', 'y', 'k', '#808080'][d % 8],
                         label='{} {}'.format(['input', 'ref'][i_sample], d))
                plt.xlim(val_range)
        err += np.mean((pdfs[0] - pdfs[1]) ** 2)

    err = err ** 0.5
    if draw:
        plt.title(err)
        plt.legend()
        plt.show(block=False)
    return err


def visualize_2d_histograms(samples_1, samples_2):
    """
    Visualize 2-D projections of the color histograms
    :param samples_1: One set of samples (input samples)
    :param samples_2: Other set of samples (reference samples)
    :return: None
    """
    samples_1 = (np.clip(samples_1, 0., 1.) * 32).astype(np.uint8)
    samples_2 = (np.clip(samples_2, 0., 1.) * 32).astype(np.uint8)
    for i_dp, dp in enumerate([(0, 1), (0, 2), (1, 2)]):
        hist_1 = np.zeros((33, 33), dtype=np.int)
        hist_2 = np.zeros((33, 33), dtype=np.int)
        for sa in samples_1:
            hist_1[sa[dp[0]], sa[dp[1]]] += 1
        for sb in samples_2:
            hist_2[sb[dp[0]], sb[dp[1]]] += 1
        plt.subplot(3, 2, 2 * i_dp + 1)
        plt.contour(hist_1, 40)
        plt.subplot(3, 2, 2 * i_dp + 2)
        plt.contour(hist_2, 40)
    plt.show()


def interp_using_grid(img, grid, grid_density):
    """
    Interpolate the color
    :param img: Image to tri-linearly interpolate according to a LUT grid
    :param grid: LUT to tri-linearly interpolate by
    :return: The interpolation of the image according to the LUT
    """
    lookup_table = {}
    out_img = np.zeros_like(img)
    grid = np.pad(grid, ((0, 1), (0, 1), (0, 1), (0, 0)), 'edge')
    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            vr, vg, vb = img[ii, jj]
            if (vr, vg, vb) not in lookup_table:
                fr, ir = math.modf(vr * (grid_density - 1))
                fg, ig = math.modf(vg * (grid_density - 1))
                fb, ib = math.modf(vb * (grid_density - 1))
                ir = int(ir)
                ig = int(ig)
                ib = int(ib)
                neighbors = \
                    grid[ir, ig, ib], \
                    grid[ir, ig, ib + 1], \
                    grid[ir, ig + 1, ib], \
                    grid[ir, ig + 1, ib + 1], \
                    grid[ir + 1, ig, ib], \
                    grid[ir + 1, ig, ib + 1], \
                    grid[ir + 1, ig + 1, ib], \
                    grid[ir + 1, ig + 1, ib + 1]
                t0 = (1. - fb) * neighbors[0] + fb * neighbors[1]
                t1 = (1. - fb) * neighbors[2] + fb * neighbors[3]
                t2 = (1. - fg) * t0 + fg * t1

                t3 = (1. - fb) * neighbors[4] + fb * neighbors[5]
                t4 = (1. - fb) * neighbors[6] + fb * neighbors[7]
                t5 = (1. - fg) * t3 + fg * t4

                final = (1. - fr) * t2 + fr * t5
                lookup_table[(vr, vg, vb)] = final
            else:
                final = lookup_table[(vr, vg, vb)]
            out_img[ii, jj] = final
    return out_img


def compute_transport(in_img, ref_img, in_mask=None, ref_mask=None,
                      n_dims=None, n_out_dims=3, rescale=False,
                      n_regularization_copies=3, regularization_noise_std=0.1, n_iterations=20, alpha=0.5,
                      grid_density=16, n_bins=100, converge_threshold=0., smooth_window=None):
    """
    Compute high-dimensional mass transport (histogram transfer) among two images with potential
    additional metadata acting as additional image layers to guide the color transfer
    :param in_img: Input image (can be more than 3 layers for high-dimensional transport)
    :param ref_img: Reference image (cam be more than 3 layers)
    :param in_mask: Mask selecting which pixels in the input image affect the learned transport
    :param ref_mask: Mask selecting which pixels in the output image affect the learned transport
    :param n_dims: Number of dimensions in the transport (inferred if None)
    :param n_out_dims: Number of dimensions of pixel data to retain and output
    :param rescale: Whether to rescale the inputs to a smaller size
    :return: Output of transport (only first n_out_dims dimensions)
    """
    if in_img.dtype == np.uint8:
        in_img = in_img.astype(np.float32) / 255.
    else:
        in_img = in_img.astype(np.float32)
    if ref_img.dtype == np.uint8:
        ref_img = ref_img.astype(np.float32) / 255.
    else:
        ref_img = ref_img.astype(np.float32)

    if n_dims is None:
        n_dims = in_img.shape[-1]

    in_img = in_img[:, :, :n_dims]
    ref_img = ref_img[:, :, :n_dims]

    use_grid = False#n_dims == 3#
    # print('USE GRID', use_grid, 'NOISE', regularization_noise_std, 'BINS', n_bins, 'ITER', n_iterations)
    if use_grid:
        rgb_grid = make_rgb_grid(grid_density)
    else:
        rgb_grid = None

    orig_in_img = in_img.copy()[:, :, :n_out_dims]

    if rescale:
        rescale_factors = (calc_rescale_factor(in_img.shape[:2], max_long_axis_after_rescale=700),
                           calc_rescale_factor(ref_img.shape[:2], max_long_axis_after_rescale=700))
        print('Rescaling by {} {}'.format(*rescale_factors))
    else:
        rescale_factors = (1., 1.)

    if rescale and rescale_factors != (1., 1.):
        base_in_img = cv2.ximgproc.guidedFilter(orig_in_img, orig_in_img, radius=5, eps=0.2 * 0.2)
        residual_in_img = orig_in_img - base_in_img

        in_img = skimage.transform.rescale(in_img, (rescale_factors[0], rescale_factors[0], 1), order=3, mode='reflect', preserve_range=True)
        ref_img = skimage.transform.rescale(ref_img, (rescale_factors[1], rescale_factors[1], 1), order=3, mode='reflect',
                                            preserve_range=True)
        if in_mask is not None:
            in_mask = skimage.transform.rescale(in_mask, rescale_factors[0], order=3,
                                                mode='reflect',
                                                preserve_range=True)
        if ref_mask is not None:
            ref_mask = skimage.transform.rescale(ref_mask, rescale_factors[1], order=3,
                                                 mode='reflect',
                                                 preserve_range=True)

    samples_in = regularize_image(in_img, n_dimensions_to_regularize=n_out_dims)
    samples_ref = regularize_image(ref_img, n_dimensions_to_regularize=n_out_dims)

    if in_mask is not None:
        samples_in_mask = np.tile(np.expand_dims(in_mask, 0),
                                  (n_regularization_copies, 1, 1, 1)
                                  ).reshape(-1)
    else:
        samples_in_mask = np.ones_like(samples_in[:, 0]).reshape(-1)
    if ref_mask is not None:
        samples_ref_mask = np.tile(np.expand_dims(ref_mask, 0),
                                   (n_regularization_copies, 1, 1, 1)
                                   ).reshape(-1)
    else:
        samples_ref_mask = np.ones_like(samples_ref[:, 0]).reshape(-1)

    # widgets = [
    #     progressbar.Percentage(),
    #     progressbar.Bar(),
    #     progressbar.DynamicMessage('Distance'),
    # ]
    in_min = samples_in[samples_in_mask > 0.5].min(axis=0)
    in_max = samples_in[samples_in_mask > 0.5].max(axis=0)

    samples_in = (samples_in - in_min) / (in_max - in_min)
    samples_ref = (samples_ref - in_min) / (in_max - in_min)
    # with progressbar.ProgressBar(maxval=100, widgets=widgets) as bar:
    # for i_iter in bar(range(n_iterations)):
    for i_iter in range(n_iterations):
        samples_in, rgb_grid, distance = transport_single_iteration(i_iter,
                                                                    samples_in,
                                                                    samples_ref,
                                                                    samples_in_mask,
                                                                    samples_ref_mask,
                                                                    rgb_grid,
                                                                    alpha,
                                                                    n_bins,
                                                                    converge_threshold,
                                                                    smooth_window)
            # bar.update(i_iter, Distance=distance)

    samples_in = (samples_in * (in_max - in_min)) + in_min
    out_img = samples_in[:in_img.shape[0] * in_img.shape[1]].reshape(
        in_img.shape
    )[:, :, :n_out_dims]

    if use_grid:
        rgb_grid = rgb_grid.clip(0., 1.)
        out_img = interp_using_grid(orig_in_img,
                                    rgb_grid.reshape(grid_density, grid_density, grid_density, 3),
                                    grid_density) #todo was here

    else:
        if rescale == True and rescale_factors[0] != 1.:
            upsampled_out_img = skimage.transform.resize(
                out_img,
                orig_in_img.shape,
                order=3,
                mode='reflect',
                preserve_range=True
            ).astype(np.float32)
            smoothed_out_img = cv2.ximgproc.guidedFilter(upsampled_out_img, upsampled_out_img,
                                                         radius=5, eps=0.2 * 0.2)
            out_img = (smoothed_out_img + residual_in_img)

    out_img = out_img.clip(0.,1.)
    return out_img
