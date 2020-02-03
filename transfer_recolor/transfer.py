import os
from copy import deepcopy
import argparse

import cv2
from imageio import imread, imsave
import matplotlib.pyplot as plt
import numpy as np
from transfer_recolor.masstransport import compute_transport
from skimage.filters import unsharp_mask
from scipy.signal import convolve2d
from skimage import transform
from skimage.color import gray2rgb, rgb2gray, rgb2yiq, yiq2rgb, rgb2yuv, yuv2rgb, rgba2rgb
from scipy.ndimage.morphology import binary_erosion
# from color.color_detection import get_facemodel_skin, get_facemodel_luminance
# from orientation.orientation import get_angles_map


im_path = './input/from_gal/'
mask_path = './input/from_gal/'
patch_path = './input/reference/'
save_dir = './output'

# im_paths = ['/Users/gpatel/Documents/data/ORG/', '/Users/gpatel/Documents/data/friends_selfies/']
# mask_paths = ['/Users/gpatel/Documents/data/ORG/', '/Users/gpatel/Documents/data/friends_selfies/']
# INDEX = 0
# im_path = im_paths[INDEX]
# mask_path = mask_paths[INDEX]
# patch_path = '/Users/gpatel/Documents/data/hair_patches_2/ladshair/'
# save_dir = '/Users/gpatel/Documents/output/mass_transport_exp'


class HairColor:
    def __init__(self, name, rgb):
        self.name = name
        self.rgb = rgb


hair_colors = [HairColor('Auburn', [165, 42, 42]),
               HairColor('Brown', [106, 78, 66]),
               HairColor('GoldenBlonde', [229, 200, 168]),
               HairColor('LightRed', [181, 82, 57]),
               HairColor('AshBlonde', [222, 188, 153])]


def enhance_details(im, mask, w=0.001):
    X, Y = 0, 1
    mask = binary_mask(mask)
    angles, confidence = get_angles_map(im, mask)
    directions = np.zeros((im.shape[0], im.shape[1], 2))
    directions[..., X] = np.cos(angles)
    directions[..., Y] = np.sin(angles)

    enhanced = deepcopy(im) / 255.
    gray = rgb2gray(enhanced)
    # sobelx = cv2.Sobel(orientations, cv2.CV_64F, 1, 0, ksize=3) / 4.
    # sobely = cv2.Sobel(orientations, cv2.CV_64F, 0, 1, ksize=3) / 4.
    kernel = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
    x_der = convolve2d(directions[..., X], kernel, mode='same')
    # x_der = gaussian_filter(x_der, 3.)
    y_der = convolve2d(directions[..., Y], kernel.T, mode='same')
    # y_der = gaussian_filter(y_der, 3.)
    # normalize:
    norm = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    # directions = directions / np.repeat(norm[..., np.newaxis], 2, axis=-1)
    # x_der, y_der = x_der / norm, y_der / norm
    x_der = np.divide(x_der, norm, where=norm != 0)
    y_der = np.divide(y_der, norm, where=norm != 0)
    # todo normalize by norm
    mask_row_idx, mask_col_idx = np.where(mask)
    for c in range(im.shape[2]):
        # enhanced[mask_row_idx, mask_col_idx, c] -= (w * x_der * (1. - confidence))[mask_row_idx, mask_col_idx]
        # enhanced[mask_row_idx, mask_col_idx, c] -= (w * y_der * (1. - confidence))[mask_row_idx, mask_col_idx]

        enhanced[mask_row_idx, mask_col_idx, c] -= (w * x_der * (1. - confidence))[mask_row_idx, mask_col_idx]
        enhanced[mask_row_idx, mask_col_idx, c] -= (w * y_der * (1. - confidence))[mask_row_idx, mask_col_idx]
    # gray -= 0.01 * sobelx
    # gray -= 0.01 * sobely

    # enhanced -= np.min(enhanced)
    # enhanced = enhanced / np.max(enhanced)
    enhanced[mask_row_idx, mask_col_idx, :] = np.clip(enhanced[mask_row_idx, mask_col_idx, :], 0., 1.)
    # plt.imshow(np.hstack(
    #     (im / 255., gray2rgb(mask), gray2rgb(confidence), enhanced)))#, gray2rgb(gray), gray2rgb(rgb2gray(im)))))
    # plt.show()

    return enhanced * 255


def enhance_edges(im, mask, w=0.001):
    X, Y = 0, 1
    mask = binary_mask(mask)
    angles, confidence = get_angles_map(im, mask)
    directions = np.zeros((im.shape[0], im.shape[1], 2))
    directions[..., X] = np.cos(angles)
    directions[..., Y] = np.sin(angles)

    enhanced = deepcopy(im) / 255.
    gray = rgb2gray(enhanced)
    # sobelx = cv2.Sobel(orientations, cv2.CV_64F, 1, 0, ksize=3) / 4.
    # sobely = cv2.Sobel(orientations, cv2.CV_64F, 0, 1, ksize=3) / 4.
    kernel = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
    der = directions.copy()
    der[:, 1:, 0] -= der[:, :-1, 0]
    # y_der = directions[..., Y].copy()
    der[1:, :, 1] -= der[:-1, :, 1]
    # normalize:
    # norm = np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))
    # # directions = directions / np.repeat(norm[..., np.newaxis], 2, axis=-1)
    # # x_der, y_der = x_der / norm, y_der / norm
    # x_der = np.divide(x_der, norm, where=norm != 0)
    # y_der = np.divide(y_der, norm, where=norm != 0)
    # todo normalize by norm
    mask_row_idx, mask_col_idx = np.where(mask)
    for c in range(im.shape[2]):
        # enhanced[mask_row_idx, mask_col_idx, c] -= (w * x_der * (1. - confidence))[mask_row_idx, mask_col_idx]
        # enhanced[mask_row_idx, mask_col_idx, c] -= (w * y_der * (1. - confidence))[mask_row_idx, mask_col_idx]

        enhanced[mask_row_idx, mask_col_idx, c] += (w * der[..., X] * (1. - confidence))[mask_row_idx, mask_col_idx]
        enhanced[mask_row_idx, mask_col_idx, c] += (w * der[..., Y] * (1. - confidence))[mask_row_idx, mask_col_idx]
    # gray -= 0.01 * sobelx
    # gray -= 0.01 * sobely

    # enhanced -= np.min(enhanced)
    # enhanced = enhanced / np.max(enhanced)
    enhanced[mask_row_idx, mask_col_idx, :] = np.clip(enhanced[mask_row_idx, mask_col_idx, :], 0., 1.)
    # plt.imshow(np.hstack(
    #     (im / 255., gray2rgb(mask), gray2rgb(confidence), enhanced)))#, gray2rgb(gray), gray2rgb(rgb2gray(im)))))
    # plt.show()

    return enhanced * 255


def unsharp(im, mask, r=5, a=1):
    X, Y = 0, 1
    mask = binary_mask(mask)
    sharpened = (unsharp_mask(im, radius=r, amount=a) * 255).astype(np.uint8)
    # plt.imshow(np.hstack((im, sharpened)))
    # plt.show()
    return sharpened


def unsharp_hair(im, mask, r=5, a=1):
    X, Y = 0, 1
    mask = binary_mask(mask)
    where_hair = np.where(mask)
    sharpened = (unsharp_mask(im, radius=5, amount=1) * 255).astype(np.uint8)
    new_im = im.copy()
    new_im[where_hair[0], where_hair[1], :] = sharpened[where_hair[0], where_hair[1], :]
    # plt.imshow(np.hstack((im, sharpened)))
    # plt.show()
    return sharpened


def compare_enhancement(weights=[0.001, 0.01, 0.05, 0.1, 0.5]):
    save_at = save_dir + '/enhanced_' + im_path.split('/')[-2]
    if not os.path.isdir(save_at):
        os.mkdir(save_at)
    weight_names = [save_at + '/' + str(w).split('.')[-1] for w in weights]
    for weight_name in weight_names:
        if not os.path.isdir(weight_name):
            os.mkdir(weight_name)
    possibilities = list(
        filter(lambda name: not (name.endswith('hair.png') or name.endswith('.DS_Store')), list(os.listdir(im_path))))
    for name in possibilities:
        if name.endswith('_hair.png'):
            continue
        print('\n\n===> im name:', name)
        im = imread(im_path + name)
        suffix = name.split('.')[-1]
        mask_full_path = (mask_path + name).replace('.' + suffix, '_hair.png')
        if not os.path.isfile(mask_full_path):
            print('not os.path.isfile(mask_full_path):')
            return
        mask = imread(mask_full_path)
        comparison = [gray2rgb(mask), im]
        for i, w in enumerate(weights):
            enhanced_im = enhance_details(im, mask, w)
            comparison.append(enhanced_im)
            imsave(os.path.join(weight_names[i], name), enhanced_im)
        imsave(os.path.join(save_at, name), np.hstack(tuple(comparison)))


def compare_edge_enhancement(weights=[0.001, 0.01, 0.05, 0.1, 0.5]):
    save_at = save_dir + '/enhanced_edges_' + im_path.split('/')[-2]
    if not os.path.isdir(save_at):
        os.mkdir(save_at)
    weight_names = [save_at + '/' + str(w).split('.')[-1] for w in weights]
    for weight_name in weight_names:
        if not os.path.isdir(weight_name):
            os.mkdir(weight_name)
    possibilities = list(
        filter(lambda name: not (name.endswith('hair.png') or name.endswith('.DS_Store')), list(os.listdir(im_path))))
    for name in possibilities:
        if name.endswith('_hair.png'):
            continue
        print('\n\n===> im name:', name)
        im = imread(im_path + name)
        suffix = name.split('.')[-1]
        mask_full_path = (mask_path + name).replace('.' + suffix, '_hair.png')
        if not os.path.isfile(mask_full_path):
            return
        mask = imread(mask_full_path)
        comparison = [gray2rgb(mask), im]
        for i, w in enumerate(weights):
            enhanced_im = enhance_edges(im, mask, w)
            comparison.append(enhanced_im)
            imsave(os.path.join(weight_names[i], name), enhanced_im)
        imsave(os.path.join(save_at, name), np.hstack(tuple(comparison)))


def color_transfer(src_im, src_mask, dst_im, dst_mask):
    depth = 5

    before = deepcopy(src_im)

    # gaussian pyramid
    src_g_pyr = [src_im]
    dst_g_pyr = [dst_im]
    for i in range(depth):
        s = cv2.pyrDown(src_g_pyr[-1])
        print(s.shape, type(s))
        src_g_pyr.append(s)
        dst_g_pyr.append(cv2.pyrDown(dst_g_pyr[-1]))
    print(len(src_g_pyr))

    # laplacian pyramid
    src_l_pyr = [src_g_pyr[depth - 1]]
    dst_l_pyr = [dst_g_pyr[depth - 1]]
    print('base', src_g_pyr[-1].shape)
    for i in range(depth - 1, 0, -1):
        print('up', src_g_pyr[i].shape)
        g = cv2.pyrUp(src_g_pyr[i])
        src_l_pyr.append(cv2.subtract(src_g_pyr[i - 1], g))
        g = cv2.pyrUp(dst_g_pyr[i])
        dst_l_pyr.append(cv2.subtract(dst_g_pyr[i - 1], g))

    e_kernel = np.ones((15, 15), np.uint8)
    print('==>', len(src_g_pyr), len(src_l_pyr))
    # src_erosion = cv2.erode(src_mask, e_kernel, iterations=1)
    src_mask = cv2.resize(src_mask, src_l_pyr[0].shape)
    print('uni', len(src_mask[src_mask == 0]))
    dst_erosion = cv2.erode(dst_mask, e_kernel, iterations=1)
    dst_mask = cv2.resize(dst_erosion, dst_l_pyr[0].shape)
    src_color = src_l_pyr[0][np.where(src_mask)]
    dst_color = dst_l_pyr[0][np.where(dst_mask)]
    print('colors', src_color.shape, dst_color.shape)
    matched = transform.match_histograms(src_color, dst_color, multichannel=True)
    src_l_pyr[0][np.where(src_mask)] = matched
    plt.imshow(np.hstack((src_color[:50], dst_color[:50], matched[:50])))
    plt.show()

    # reconstruct
    re = src_l_pyr[0]
    for i in range(1, depth):
        re = cv2.pyrUp(re)
        print(re.shape, src_l_pyr[i].shape)
        re = cv2.add(re, src_l_pyr[i])

    plt.imshow(np.hstack((before, re, dst_im)))
    plt.show()


def binary_mask(mask, T=128):
    mask = deepcopy(mask)
    mask[mask < T] = 0
    mask[mask > 0] = 255
    return mask


def histogram_transform_1d(src_im, src_mask, dst_im, dst_mask):
    # mask matching shapes
    # dst_im = (imresize(dst_im, src_im.shape)*255).astype(np.uint8)
    # print('dst_mask', np.unique(dst_mask))
    # dst_mask = (imresize(dst_mask, src_mask.shape)*255).astype(np.uint8)
    # # print('src_mask', np.unique(src_mask))
    # print('dst_mask', np.unique(dst_mask))

    # binary mask
    src_mask_bin, dst_mask_bin = binary_mask(src_mask) / 255, binary_mask(dst_mask) / 255
    # print('src_mask_bin', np.unique(src_mask_bin))
    print('dst_mask_bin', np.unique(dst_mask_bin))

    p1 = src_im[np.where(src_mask_bin)]
    p2 = dst_im[np.where(dst_mask_bin)]
    print('p shapes', p1.shape, p2.shape)
    # print('p1', np.unique(p1))
    # print('p2', np.unique(p2))
    hs = []
    hs.append(np.histogram(p1, 256)[0])
    hs.append(np.histogram(p2, 256)[0])
    Hs = []
    Hs.append(np.cumsum(hs[0]))
    Hs.append(np.cumsum(hs[1]))

    L = np.zeros(256, dtype=np.uint8)
    ims = [p1, p2]
    t = 0
    for g in range(256):
        print('g', g, 't', t)
        while Hs[1][t] < Hs[0][g]:  # Hs[1] is cumsum of reference
            t += 1
            # print('t', t)
        L[g] = (t + g) // 2
    print('ims[0]', ims[0].shape)
    return L[ims[0]]


def hist_match(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask), binary_mask(dst_mask)
    src_color = src_im[np.where(src_mask_bin)]
    dst_color = dst_im[np.where(dst_mask_bin)]
    # print('colors', src_color.shape, dst_color.shape)
    matched = transform.match_histograms(src_color, dst_color, multichannel=True)
    painted_im = deepcopy(src_im)
    painted_im[np.where(src_mask_bin)] = matched
    # alpha = src_mask.astype(np.float32) / 255.
    alpha = np.zeros(src_mask.shape, dtype=np.float32) / 255.
    alpha[np.where(src_mask_bin)] = src_mask[np.where(src_mask_bin)]
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    # print(np.unique(painted_im), np.unique(before))

    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_bare(src_im, src_mask, dst_im, dst_mask=None, n_iterations=20, n_bins=50):

    if dst_mask is None:
        dst_mask = np.full((dst_im.shape[0], dst_im.shape[1]), 255.0)
    else:
        if len(dst_mask.shape) == 3:
            dst_mask = dst_mask[:, :, 0]
    if len(src_mask.shape) == 3:
        src_mask = src_mask[:, :, 0]

    src_mask_bin = binary_mask(src_mask).astype(np.float32) / 255.
    dst_mask_bin = binary_mask(dst_mask).astype(np.float32) / 255.

    if len(np.where(src_mask_bin)[0]) == 0:
        return None

    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin,
                                   rescale=True, n_iterations=n_iterations, n_bins=n_bins) * 255
    return painted_im


def transport_match_bare_noise05(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True,
                                   regularization_noise_std=0.5) * 255  #
    return painted_im

def transport_match_bare_bins50(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True,
                                   n_bins=50) * 255  #
    return painted_im

def transport_match_bare_bins200(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True,
                                   n_bins=200) * 255  #
    return painted_im

def transport_match_bare_iterations10(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True,
                                   n_iterations=10) * 255  #
    return painted_im

def transport_match_bare_iq(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    src_yiq, dst_yiq = (np.clip(rgb2yiq(src_im),0.0,1.0) * 255).astype(np.uint8), (np.clip(rgb2yiq(dst_im),0.0,1.0) *
                                                                                           255).astype(
        np.uint8)
    painted_im_iq = compute_transport(src_yiq[..., 1:], dst_yiq[..., 1:], src_mask_bin, dst_mask_bin, rescale=True) * \
                    255  #
    painted_im_yiq = src_yiq.copy()
    painted_im_yiq[..., 1:] = painted_im_iq
    painted_im = (np.clip(yiq2rgb(painted_im_yiq),0.0,1.0) * 255).astype(np.uint8)
    return painted_im


def transport_match_erode20(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    src_mask_bin = binary_erosion(src_mask_bin, structure=np.ones((20, 20))).astype(src_mask_bin.dtype)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_enhanced(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    enhanced_im = enhance_details(src_im, src_mask, 0.005)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return np.hstack((enhanced_im, res))


def transport_enhanced_edges(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    enhanced_im = enhance_edges(src_im, src_mask, 0.05)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return np.hstack((enhanced_im, res))


def transport_unsharpened(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    sharpened_im = unsharp(src_im, src_mask, 0.005)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return np.hstack((sharpened_im, res))


def transport_unsharpened_3_2(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    sharpened_im = unsharp(src_im, src_mask, 3, 2)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return np.hstack((sharpened_im, res))


def transport_unsharpened_hair(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    sharpened_im = unsharp_hair(src_im, src_mask, 0.005)
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return np.hstack((sharpened_im, res))


def transport_match_power_blend(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    alpha = np.power(alpha, 2.)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_smooth_step(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    alpha = (3. * np.power(alpha, 2.)) - (2. * np.power(alpha, 3.))
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_gamma12_after_d(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    painted_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #

    painted_im = painted_im.astype(np.float32) / 255.
    painted_im = np.power(painted_im, 1.2).clip(0., 1.) * 255
    painted_im = painted_im.astype(np.uint8)

    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_gamma105_ref(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.

    dst_for_transport = src_im.copy().astype(np.float32) / 255.
    dst_for_transport = np.power(dst_for_transport, 1.05).clip(0., 1.) * 255
    dst_for_transport = dst_for_transport.astype(np.uint8)
    painted_im = compute_transport(src_im, dst_for_transport, src_mask_bin, dst_mask_bin, rescale=True) * 255  #

    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_gamma085_before_d(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.

    im_for_transport = src_im.copy().astype(np.float32) / 255.
    im_for_transport = np.power(im_for_transport, .85) * 255
    im_for_transport = im_for_transport.astype(np.uint8)
    painted_im = compute_transport(im_for_transport, dst_im, src_mask_bin, dst_mask_bin, rescale=True) * 255  #

    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def overlay_blend(im, mask, a=.5):
    alpha = 1. - (a * mask)
    blend = deepcopy(im)
    blend[:, :, 2] = np.multiply(im[:, :, 2], alpha) + (255 * (1. - alpha))
    blend[:, :, 0] = im[:, :, 0] * 0.8  # + (255 * (1. - alpha))
    blend[:, :, 1] = im[:, :, 1] * 0.8  # + (255 * (1. - alpha))
    return blend.astype(np.uint8)


def blend_mask(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    paint = np.zeros(src_im.shape)
    paint[..., 2] = 1.
    mask = src_mask.astype(np.float32) / 255.
    # alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = overlay_blend(src_im, mask, 1.)
    return res


def transport_match_normalizeDstWhereHair(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    # non_hair_where = np.where(1 - src_mask_bin)
    hair_where = np.where(src_mask_bin)
    # normalized_dst = normalize_image(dst_im, src_im[non_hair_where[0], non_hair_where[1], :])
    normalized_dst = normalize_image(dst_im, src_im[hair_where[0], hair_where[1], :])
    painted_im = compute_transport(src_im, normalized_dst, src_mask_bin, dst_mask_bin, rescale=False) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res, normalized_dst


def transport_match_normalizeDst08(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    normalized_dst = normalize_image(dst_im, src_im, .8)
    painted_im = compute_transport(src_im, normalized_dst, src_mask_bin, dst_mask_bin, rescale=False) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_normalizeDst(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    normalized_dst = normalize_image(dst_im, src_im)
    painted_im = compute_transport(src_im, normalized_dst, src_mask_bin, dst_mask_bin, rescale=False) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_normalizeDstM0(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    normalized_dst = normalize_image(dst_im, src_im, orig_mean=0.)
    painted_im = compute_transport(src_im, normalized_dst, src_mask_bin, dst_mask_bin, rescale=False) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_normalizeDstM05(src_im, src_mask, dst_im, dst_mask):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    normalized_dst = normalize_image(dst_im, src_im, orig_mean=.5)
    painted_im = compute_transport(src_im, normalized_dst, src_mask_bin, dst_mask_bin, rescale=False) * 255  #
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(painted_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)
    return res


def transport_match_with_gray(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst to match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    # print('src im',np.unique(src_im))
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1)
    # print('gray values', new_dst_gray_values)

    # now make 4 channels images - extra channel for gray values
    new_src_im = np.zeros((src_im.shape[0], src_im.shape[1], 4), dtype=np.uint8)
    new_src_im[..., :3] = src_im
    new_src_im[..., 3] = gray_src_im[..., 0] * 255
    new_dst_im = np.zeros((dst_im.shape[0], dst_im.shape[1], 4), dtype=np.uint8)
    new_dst_im[..., :3] = dst_im
    new_dst_im[..., 3] = new_dst_gray_values[..., 0] * 255

    # now match the src to the dst
    transformed_src = compute_transport(new_src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    transformed_src = transformed_src[..., :3]
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res


def transport_2way_meanLUT(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, new_dst_im


def transport_2way_fixLUT(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    not_detected = []
    should_have = []
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        l = r = v
        while possible_values.shape[0] == 0:
            if l > 0:
                where_before = np.where(gray_dst_im == l - 1)
                possible_values = np.concatenate((possible_values, dst_im[where_before[0], where_before[1], :]))
                l -= 1
            if r < 255:
                where_after = np.where(gray_dst_im == r + 1)
                possible_values = np.concatenate((possible_values, dst_im[where_after[0], where_after[1], :]))
                r += 1

        num_possibilities = possible_values.shape[0]
        assert num_possibilities > 0
        #     not_detected.append(v)
        #     rgb = [0,0,0]
        #     print(rgb)
        # else:
        #     rgb = possible_values[np.random.randint(0, num_possibilities)]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        if len(where_new[0]):
            should_have.append(v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, new_dst_im


def transport_2way_distLUT(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    not_detected = []
    should_have = []
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        l = r = v
        while possible_values.shape[0] == 0:
            if l > 0:
                where_before = np.where(gray_dst_im == l - 1)
                possible_values = np.concatenate((possible_values, dst_im[where_before[0], where_before[1], :]))
                l -= 1
            if r < 255:
                where_after = np.where(gray_dst_im == r + 1)
                possible_values = np.concatenate((possible_values, dst_im[where_after[0], where_after[1], :]))
                r += 1

        num_possibilities = possible_values.shape[0]
        assert num_possibilities > 0
        #     not_detected.append(v)
        #     rgb = [0,0,0]
        #     print(rgb)
        # else:
        #     rgb = possible_values[np.random.randint(0, num_possibilities)]
        # rgb = np.mean(possible_values, axis=0)
        rgb = possible_values[np.random.randint(0, num_possibilities)]
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        if len(where_new[0]):
            should_have.append(v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, new_dst_im


def transport_match_grayLUT(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    # print('src im',np.unique(src_im))
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, np.ones_like(src_mask_bin),
                                            rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res


def transport_match_grayLUTHalfHist(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    # first, make gray values of dst to match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    # print('src im',np.unique((gray_dst_im[...,0]*255).astype(np.uint8).copy()))
    new_dst_gray_values = histogram_transform_1d((gray_src_im[..., 0] * 255).astype(np.uint8).copy(), src_mask,
                                                 (gray_dst_im[..., 0] * 255).astype(np.uint8).copy(), dst_mask)
    new_dst_gray_values = new_dst_gray_values[..., np.newaxis]
    print('new', new_dst_gray_values.shape, np.unique(new_dst_gray_values))

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    # gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res


# , rgb2colorSpace, colorSpace2rgb

def transport_2way_meanYIQ(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    return transport_2way_meanChroma(src_im, src_mask, dst_im, dst_mask, rgb2yiq, yiq2rgb, save_stats)


def transport_2way_meanYUV(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    return transport_2way_meanChroma(src_im, src_mask, dst_im, dst_mask, rgb2yuv, yuv2rgb, save_stats)


def transport_2way_meanChroma(src_im, src_mask, dst_im, dst_mask, rgb2colorSpace, colorSpace2rgb, save_stats=''):
    # first, make gray values of dst to match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    chroma_src_im = rgb2colorSpace(src_im)
    chroma_dst_im = rgb2colorSpace(dst_im)
    y_src_im = chroma_src_im[..., 0][..., np.newaxis]
    y_dst_im = chroma_dst_im[..., 0][..., np.newaxis]

    # print('src im',np.unique(src_im))
    new_dst_y_values = compute_transport(y_dst_im, y_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                         n_dims=1, n_out_dims=1)  # * 255
    new_dst_chroma_values = chroma_dst_im.copy()
    new_dst_chroma_values[..., 0] = new_dst_y_values[..., 0]
    new_dst_im = colorSpace2rgb(new_dst_chroma_values).clip(0., 1.) * 255  # *255#.astype(np.uint8)
    new_dst_im = new_dst_im.astype(np.uint8)

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, new_dst_im


def transport_match_grayLUTshift(src_im, src_mask, dst_im, dst_mask, s=1., save_stats=''):
    # first, make gray values of dst to match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    # print('src im',np.unique(src_im))
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    gray_diff = np.max(gray_dst_im) - np.max(new_dst_gray_values)
    if gray_diff > 0.:
        new_dst_gray_values += int(s * gray_diff)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, new_dst_im


def transport_match_grayLUTavg(src_im, src_mask, dst_im, dst_mask, a=.5, save_stats=''):
    # first, make gray values of dst to match the src
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src_im = rgb2gray(src_im)[..., np.newaxis]
    gray_dst_im = rgb2gray(dst_im)[..., np.newaxis]
    # print('src im',np.unique(src_im))
    new_dst_gray_values = compute_transport(gray_dst_im, gray_src_im, dst_mask_bin, src_mask_bin, rescale=False,
                                            n_dims=1, n_out_dims=1) * 255
    new_dst_gray_values = new_dst_gray_values.astype(np.uint8)

    new_dst_gray_values = (a * new_dst_gray_values) + ((1. - a) * new_dst_gray_values)

    # using the original as LUT, convert to new RGB dst image
    new_dst_im = np.zeros_like(dst_im)
    gray_dst_im = (gray_dst_im * 255).astype(np.uint8)
    lut = {}
    for v in range(256):
        where_orig = np.where(gray_dst_im == v)
        possible_values = dst_im[where_orig[0], where_orig[1], :]
        rgb = np.mean(possible_values, axis=0)
        lut[v] = rgb
        where_new = np.where(new_dst_gray_values == v)
        new_dst_im[where_new[0], where_new[1], :] = rgb

    # now match the src to the dst
    transformed_src = compute_transport(src_im, new_dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                        n_dims=4, n_out_dims=4) * 255
    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_src, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res


def transport_match_yuv(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    return transport_match_chroma(src_im, src_mask, dst_im, dst_mask, rgb2yuv, yuv2rgb, save_stats=save_stats)


def transport_match_yiq(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    return transport_match_chroma(src_im, src_mask, dst_im, dst_mask, rgb2yiq, yiq2rgb, save_stats=save_stats)


def transport_match_chroma(src_im, src_mask, dst_im, dst_mask, rgb2colorSpace, colorSpace2rgb, save_stats=''):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.

    src_im_colorSpace = rgb2colorSpace(src_im)
    dst_im_colorSpace = rgb2colorSpace(dst_im)
    src_im_chroma = src_im_colorSpace[..., 1:]
    dst_im_chroma = dst_im_colorSpace[..., 1:]

    transformed_im_chroma = compute_transport(src_im_chroma, dst_im_chroma, src_mask_bin, dst_mask_bin, rescale=False,
                                              n_dims=2, n_out_dims=2)
    transformed_im = np.zeros_like(src_im_colorSpace)
    transformed_im[..., 0] = src_im_colorSpace[..., 0]
    transformed_im[..., 1:] = transformed_im_chroma
    transformed_im = colorSpace2rgb(transformed_im).clip(0., 1.)
    transformed_im *= 255

    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res


def transport_match_shift(src_im, src_mask, dst_im, dst_mask, save_stats=''):
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.

    src_mean = np.mean(rgb2gray(src_im))
    dst_mean = np.mean(rgb2gray(dst_im))
    diff = (dst_mean - src_mean) * 0.333
    dst_im = dst_im.copy() - diff

    transformed_im = compute_transport(src_im, dst_im, src_mask_bin, dst_mask_bin, rescale=False,
                                       n_dims=1, n_out_dims=1)
    transformed_im *= 255

    alpha = src_mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)
    res = np.multiply(transformed_im, alpha).astype(np.uint8) + np.multiply(src_im, 1. - alpha).astype(np.uint8)

    if save_stats:
        compare_hist([src_im, res, dst_im], [src_mask_bin, src_mask_bin, dst_mask_bin],
                     ['source', 'result', 'destination'], save_stats)

    return res, dst_im


def stack_images(src_mask, src_im, res, dst_im, dst_mask):
    height = int(max(src_im.shape[0], dst_im.shape[0]))
    width = int(max(src_im.shape[1], dst_im.shape[1]))
    stack = np.zeros((height, width * 5, 3))
    for i, im in enumerate([src_mask, src_im, res, dst_im, dst_mask]):
        stack[:im.shape[0], i * width:(i * width) + im.shape[1], :] = im
    return stack


def stack_image_patch(src_mask, src_im, res, dst_im):
    dst_im = cv2.resize(dst_im, src_im.shape)
    return np.hstack((src_mask, src_im, res, dst_im))


def match(src_im, src_mask, dst_im, dst_mask, save_path='', method=hist_match, patch=True):
    res = method(src_im, src_mask, dst_im, dst_mask, save_path[:-4] + '_stats.png')
    # stitcher = cv2.Stitcher_create()
    # (status, stitched) = stitcher.stitch([])
    if save_path:
        # if not os.path.isdir(save_path):
        #     os.mkdir(save_path)
        # print(np.unique(src_im))
        # print(np.unique(res))
        # print(src_im.shape, res.shape, dst_im.shape)
        if patch:
            imsave(save_path, stack_image_patch(gray2rgb(src_mask), src_im, res, dst_im))
        else:
            imsave(save_path, stack_images(gray2rgb(src_mask), src_im, res, dst_im, gray2rgb(dst_mask)))
        # imsave(save_path+'/source.png', src_im)
        # imsave(save_path+'/mask.png', src_mask_bin)
    else:
        plt.imshow(np.hstack((gray2rgb(src_mask), src_im, res, dst_im, gray2rgb(dst_mask))))
        plt.show()


def compare_histograms(src_im, src_mask, dst_im, dst_mask, save_path):
    # hair regions only
    src_mask_bin, dst_mask_bin = binary_mask(src_mask).astype(np.float32) / 255., binary_mask(dst_mask).astype(
        np.float32) / 255.
    gray_src, gray_dst = rgb2gray(src_im), rgb2gray(dst_im)
    src_values, dst_values = gray_src[np.where(src_mask_bin)], gray_dst[np.where(dst_mask_bin)]
    src_hist, src_bins = np.histogram(src_values, 256)
    dst_hist, dst_bins = np.histogram(dst_values, 256)
    # src_im, src_mask, dst_im, dst_mask
    compare_hist([src_im, dst_im], [src_mask_bin, dst_mask_bin],
                 ['source', 'destination'], save_path)


def save_match(src_im, src_mask, dst_im, dst_mask, method, save_path='', stats=False, new_patches=False,
               new_patches_path=''):
    if stats:
        method(src_im, src_mask, dst_im, dst_mask, save_path)
    else:
        if new_patches:
            res, new_patch = method(src_im, src_mask, dst_im, dst_mask)
            imsave(new_patches_path, new_patch)
        else:
            res = method(src_im, src_mask, dst_im, dst_mask)
        imsave(save_path, res)


def compare_match(src_im, src_mask, dst_im, dst_mask, save_path='', methods=[hist_match], patch=True):
    results = []
    for method in methods:
        res = method(src_im, src_mask, dst_im, dst_mask)
        results.append(res)
    results = np.hstack(tuple(results))
    if save_path:
        if patch:
            imsave(save_path, stack_image_patch(gray2rgb(src_mask), src_im, results, dst_im))
        else:
            imsave(save_path, stack_images(gray2rgb(src_mask), src_im, results, dst_im, gray2rgb(dst_mask)))
    else:
        plt.imshow(np.hstack((gray2rgb(src_mask), src_im, results, dst_im, gray2rgb(dst_mask))))
        plt.show()


def single_image_recolor(name, color_name, save_dir=save_dir):
    im = imread(im_path + name)
    mask = imread(mask_path + name)
    # im = enhance_details(im, binary_mask(mask), 0.01) * 255
    color_im = imread(im_path + color_name)
    color_mask = imread(mask_path + color_name)
    match(im, mask, color_im, color_mask, save_dir + '/' +
          name.split('.')[0] + '__' + color_name.split('.')[0] + '.png', transport_match)


def patch_image_recolor_enhance(name, patch, save_dir=save_dir):
    im = imread(im_path + name)
    mask = imread(mask_path + name)
    enhanced = enhance_details(im, mask, w=0.1)
    patch_im = imread(patch_path + patch)
    patch_mask = np.full((patch_im.shape[0], patch_im.shape[1]), 255)
    # print(np.max(mask), np.max(patch_mask))
    match(im, mask, patch_im, patch_mask, save_dir + '/patch_' +
          name.split('.')[0] + '__' + patch.split('.')[0] + '.png', transport_match)
    match(enhanced, mask, patch_im, patch_mask, save_dir + '/patch_' +
          name.split('.')[0] + '__' + patch.split('.')[0] + '_rem_nder_dg5_1.png', transport_match)


def compare_hist(images, masks, names, save_path, show_images=True):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # assumes no more images than colors

    gray_images = []
    for im in images:
        gray_images.append(rgb2gray(im) * 255)

    if show_images:
        fig, axs = plt.subplots(2, len(images), tight_layout=True)
    else:
        fig, axs = plt.subplots(1, 2, tight_layout=True)

    fig.subplots_adjust(top=0.9)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[1] = 3.0
    plt.rcParams["figure.figsize"] = fig_size

    for i in range(len(images)):
        im = images[i]
        gray_im = gray_images[i]
        mask = masks[i]
        # print(i, gray_im.shape, mask.shape)
        im_values = gray_im[np.where(mask)]  # .astype(np.float32)
        # axs[0][i].hist(im_values, bins=256, normed=True)
        hist, bins = np.histogram(im_values, bins=256, range=(0, 256), density=False)  # , density=True)
        # assert np.abs(np.sum(hist) - 1.0) < 0.000000001
        if show_images:
            a = axs[0]
        else:
            a = axs
        a[0].plot(bins[:-1], hist, color=colors[i], label=names[i], alpha=0.7)
        a[1].plot(bins[:-1], np.cumsum(hist), color=colors[i], label=names[i], alpha=0.7)
        a[0].set_title('Histogram')
        a[1].set_title('Cumulative Distribution')
        for j in range(2):
            a[j].set_xlim(bins[0], bins[-2])
            # print(np.histogram(gray_im, 256)[0].shape)#, np.unique(np.histogram(gray_im, 256)[0]))
            a[j].grid(True)
            a[j].legend(loc=0)  # , prop={'size': 6})
        if show_images:
            axs[1][i].imshow(np.hstack((im / 255., gray2rgb(gray_im))))
            axs[1][i].get_xaxis().set_visible(False)
            axs[1][i].get_yaxis().set_visible(False)
            axs[1][i].set_title(names[i])
    # src_values /= np.max(src_values)

    # target_values = gray_target_im[np.where(target_mask)]
    # target_values /= np.max(target_values)
    # axs[0][1].hist(target_values, bins=256, color='r')
    # axs[0][1].grid(True)
    # axs[1][1].imshow(np.hstack((target_im/255., gray2rgb(gray_target_im))))

    # plt.suptitle('Grayscale Histograms')

    plt.savefig(save_path)


def patch_image_recolor(name, patch, save_dir, method, stats, new_patches):
    im = imread(im_path + name)
    if len(im.shape) != 3:
        print('image shape is invalid:', im.shape)
        return
    if im.shape[2] == 4:
        im = (rgba2rgb(im) * 255).astype(np.uint8)
        print('Image wag RGBa')
    suffix = name.split('.')[-1]
    mask_full_path = (mask_path + name).replace('.' + suffix, '_hair.png')
    if not os.path.isfile(mask_full_path):
        return
    mask = imread(mask_full_path)
    # mask = imread(mask_path + name)
    patch_im = imread(patch_path + patch)
    # patch_mask = np.full((patch_im.shape[0], patch_im.shape[1]), 255)
    # resize patch
    patch_im = cv2.resize(patch_im, im.shape)
    patch_mask = np.full((patch_im.shape[0], patch_im.shape[1]), 255)
    # compare_hist([im, patch_im], [mask, patch_mask], ['source', 'destination], save_dir + '/patch_' +
    #              name.split('.')[0] + '_' + patch.split('.')[0] + 'stats.png')
    # compare_match(im, mask, patch_im, patch_mask, save_dir + '/patch_' +
    #               name.split('.')[0] + '__' + patch.split('.')[0] + '.png',
    #               [method])  # transport_match)
    save_name = 'patch_' + name.split('.')[0] + '__' + patch.split('.')[0] + '.png'
    if os.path.isfile(os.path.join(save_dir, save_name)):
        print(save_name, 'already exists')
        return
    new_patches_dir = ''
    if new_patches:
        new_patches_dir = save_dir + '_generated_patches'
        if not os.path.isdir(new_patches_dir):
            os.mkdir(new_patches_dir)

    save_match(im, mask, patch_im, patch_mask, method,
               save_path=save_dir + '/' + save_name, stats=stats,
               new_patches=new_patches, new_patches_path=new_patches_dir + '/' + save_name)


def patch_image_recolor_use_facemodel(name, patch, save_dir, method, stats, new_patches, skin, luminance):
    im = imread(im_path + name)
    if len(im.shape) != 3:
        print('image shape is weird:', im.shape)
        return
    if im.shape[2] == 4:
        im = (rgba2rgb(im) * 255).astype(np.uint8)
        print('Image wag RGBa')
    suffix = name.split('.')[-1]
    mask_full_path = (mask_path + name).replace('.' + suffix, '_hair.png')
    if not os.path.isfile(mask_full_path):
        return
    mask = imread(mask_full_path)
    patch_im = imread(patch_path + patch)
    patch_im = cv2.resize(patch_im, im.shape)
    patch_mask = np.full((patch_im.shape[0], patch_im.shape[1]), 255)
    save_name = 'patch_' + name.split('.')[0] + '__' + patch.split('.')[0] + '.png'
    if os.path.isfile(os.path.join(save_dir, save_name)):
        print(save_name, 'already exists')
        return
    new_patches_dir = ''
    if new_patches:
        new_patches_dir = save_dir + '_generated_patches'
        if not os.path.isdir(new_patches_dir):
            os.mkdir(new_patches_dir)


    # strip image from luminance
    im = np.multiply(im.astype(np.float)/255., 1./luminance)
    im = np.clip(im, 0.0, 1.0)*255
    im = im.astype(np.uint8)

    save_match(im, mask, patch_im, patch_mask, method,
               save_path=save_dir + '/' + save_name, stats=stats,
               new_patches=new_patches, new_patches_path=new_patches_dir + '/' + save_name)

def remove(name='stats', dir=save_dir):  # '/Users/gpatel/Documents/output/mass_transport'):
    # for dir in ['/Users/gpatel/Documents/output/mass_transport', im_path, mask_path]:
    files = os.listdir(dir)
    for file in files:
        if name in file:
            os.remove(dir + '/' + file)


def image2image_recolor():
    src_num = ref_num = 12
    possibilities = list(os.listdir(im_path))
    src_choice = np.random.choice(range(len(possibilities)), src_num, replace=False)
    ref_choice = np.random.choice(range(len(possibilities)), ref_num, replace=False)
    for i in src_choice:
        src_name = possibilities[i]
        for j in ref_choice:
            ref_name = possibilities[j]
            single_image_recolor(src_name, ref_name)


def patch2image_recolor(method, stats=False, new_patches=False):
    print('METHOD:', method.__name__)
    save_at = save_dir + '/' + method.__name__ + '_' + patch_path.split('/')[-2]
    if not os.path.isdir(save_at):
        os.mkdir(save_at)
    if new_patches and not os.path.isdir(save_at + '_generated_patches'):
        os.mkdir(save_at + '_generated_patches')
    src_num = 50
    possibilities = list(
        filter(lambda name: not (name.endswith('hair.png') or name.endswith('.DS_Store')), list(os.listdir(im_path))))
    if len(possibilities) <= src_num:
        src_choice = range(len(possibilities))
    else:
        src_choice = np.random.choice(range(len(possibilities)), src_num, replace=False)
    patches = os.listdir(patch_path)
    for i in src_choice:
        src_name = possibilities[i]
        if src_name.endswith('_hair.png'):
            continue
        if src_name.endswith('stats.png'):
            os.remove(src_name)
            print('removed', src_name)
        print('\n\n===> im name:', src_name)
        for patch in patches:
            if not (patch.endswith('.png') or patch.endswith('.jpg')):
                continue
            print('\npatch:', patch)
            patch_image_recolor(src_name, patch, save_at, method, stats, new_patches)

def patch2image_recolor_use_facemodel(method, facemodel_path, stats=False, new_patches=False):
    print('METHOD:', method.__name__)
    save_at = save_dir + '/' + method.__name__ + '_facemodel_usage_' + patch_path.split('/')[-2]
    if not os.path.isdir(save_at):
        os.mkdir(save_at)
    if new_patches and not os.path.isdir(save_at + '_generated_patches'):
        os.mkdir(save_at + '_generated_patches')
    src_num = 50
    possibilities = list(
        filter(lambda name: not (name.endswith('hair.png') or name.endswith('.DS_Store')), list(os.listdir(im_path))))
    if len(possibilities) <= src_num:
        src_choice = range(len(possibilities))
    else:
        src_choice = np.random.choice(range(len(possibilities)), src_num, replace=False)
    patches = os.listdir(patch_path)
    for i in src_choice:
        src_name = possibilities[i]
        if src_name.endswith('_hair.png'):
            continue
        if src_name.endswith('stats.png'):
            os.remove(src_name)
            print('removed', src_name)
        print('\n\n===> im name:', src_name)
        prefix = src_name.split('.')[0]
        skin_color, luminance = get_facemodel_skin(prefix, facemodel_path), get_facemodel_luminance(prefix, facemodel_path)
        for patch in patches:
            if not (patch.endswith('.png') or patch.endswith('.jpg')):
                continue
            print('\npatch:', patch)
            patch_image_recolor_use_facemodel(src_name, patch, save_at, method, stats, new_patches, skin_color, luminance)

def single_manual_image_recolor(name, color, save_dir):
    im = imread(im_path + name)
    suffix = name.split('.')[-1]
    mask_full_path = (mask_path + name).replace('.' + suffix, '_hair.png')
    if not os.path.isfile(mask_full_path):
        return
    mask = imread(mask_full_path)
    gray = gray2rgb(rgb2gray(im))
    # gamma correction
    gray = np.power(gray, 0.5) * 255
    gray = gray.astype(np.uint8)
    color_im = np.zeros_like(im)
    color_im[:, :, :] = color.rgb

    alpha = mask.astype(np.float32) / 255.
    alpha = np.repeat(alpha[..., np.newaxis], 3, axis=-1)

    painted = np.multiply(color_im, 0.5 * alpha).astype(np.uint8) + np.multiply(gray, 1. - 0.5 * alpha).astype(np.uint8)
    imsave(save_dir + '/patch_' + name.split('.')[0] + '__' + color.name + '_basis.png', painted)

    res = np.multiply(painted, alpha).astype(np.uint8) + np.multiply(im, 1. - alpha).astype(np.uint8)
    imsave(save_dir + '/patch_' + name.split('.')[0] + '__' + color.name + '.png', res)


def manual_images_recolor():
    save_at = save_dir + '/' + 'manual'
    if not os.path.isdir(save_at):
        os.mkdir(save_at)
    src_num = 50
    possibilities = list(
        filter(lambda name: not (name.endswith('hair.png') or name.endswith('.DS_Store')), list(os.listdir(im_path))))
    if len(possibilities) <= src_num:
        src_choice = range(len(possibilities))
    else:
        src_choice = np.random.choice(range(len(possibilities)), src_num, replace=False)
    for i in src_choice:
        src_name = possibilities[i]
        if src_name.endswith('_hair.png'):
            continue
        if src_name.endswith('stats.png'):
            os.remove(src_name)
            print('removed', src_name)
        print('\n\n===> im name:', src_name)
        # gray hair
        for color in hair_colors:
            print('\nhair color:', color.name)
            single_manual_image_recolor(src_name, color, save_at)


def get_dr(image):
    min_val, max_val = np.min(image), np.max(image)
    dr = np.log2(max_val - min_val)
    return min_val, max_val, max_val - min_val, dr


def get_rms(image):
    # values = image[np.where(mask)[0], np.where(mask)[1], :]
    values = rgb2gray(image)
    return np.sqrt(np.std(values))


def get_basic_stat(image):
    # gray = rgb2gray(image)
    return np.mean(image), np.std(image)


def normalize_image(src_im, dst_im, factor=1., orig_mean=1.):
    '''normalize src im to have same gray stats as dst image'''
    gray_src, gray_dst = rgb2gray(src_im), rgb2gray(dst_im)
    mu_src, mu_dst = np.mean(gray_src), np.mean(gray_dst)
    s_src, s_dst = np.std(gray_src), np.std(gray_dst)
    # print(np.unique(src_im))
    x = (src_im.copy().astype(np.float32) / 255.)
    print('before:', np.mean(x), np.std(x), np.min(x), np.max(x))
    print('current std', np.std(x), 'wanted std', np.std(dst_im.copy().astype(np.float32) / 255.))
    # new_src = (((x - mu_src) / s_src) + mu_dst) * mu_dst
    new_src = (orig_mean * mu_src) + ((1. - orig_mean) * mu_dst) + ((x - mu_src) * (factor * (s_dst / s_src)))

    print('new src', np.min(new_src), np.max(new_src))
    # new_src = new_src.clip(0., 1.).astype(np.uint8)
    print('after1:', np.mean(new_src), np.std(new_src))
    new_src = np.clip(new_src, 0., 1.)
    print('after2:', np.mean(new_src), np.std(new_src))
    new_src = new_src * 255
    new_src = new_src.astype(np.uint8)
    print('after3:', np.mean(new_src), np.std(new_src))
    return new_src


def data_stats():
    patches = list(filter(lambda name: name.endswith('.png') or name.endswith('.jpg'), list(os.listdir(patch_path))))
    hairs = list(filter(lambda name: not name.endswith('hair.png'), list(os.listdir(im_path))))
    for patch in patches:
        patch_im = imread(patch_path + patch)
        print(patch, get_dr(patch_im), get_rms(patch_im), get_basic_stat(patch_im))
    for hair in hairs:
        suffix = hair.split('.')[-1]
        mask_path = (im_path + hair).replace('.' + suffix, '_hair.png')
        if not os.path.isfile(mask_path):
            continue
        hair_im = imread(im_path + hair)
        mask = imread(mask_path)
        hair_only = hair_im[np.where(mask >= 0.5)]
        print(hair, get_dr(hair_im), get_dr(hair_only), get_rms(hair_im), get_basic_stat(hair_im))


if __name__ == '__main__':
    # patch2image_recolor(transport_match_bare)

    src_im = imageio

    transport_match_bare(src_im, src_mask, dst_im, dst_mask)

