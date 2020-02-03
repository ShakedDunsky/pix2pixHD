import imageio
import numpy as np
import coremltools
import argparse
import os
import matplotlib.pyplot as plt

from transfer_recolor.transfer import transport_match_bare
from transfer_recolor.utils import name_from_path, imsave


def apply_mask(im, recolored_full_im, alpha):

    if recolored_full_im is None:
        return None

    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
    if np.max(alpha) > 1:
        alpha = alpha / 255

    recolored_only_hair = np.multiply(1 - alpha, im) + np.multiply(alpha, recolored_full_im)
    recolored_only_hair = recolored_only_hair.astype(np.uint8)

    return recolored_only_hair


def recolor_im(im, hair_mask, dst_im=None):

    if dst_im is None:
        dst_im = np.array(imageio.imread('../research-hair/input/reference/blonde.jpg'))

    recolored_full_im = transport_match_bare(im, hair_mask, dst_im, n_iterations=10)

    out_im = apply_mask(im, recolored_full_im, hair_mask)
    return out_im


def enhance_brightening(orig_im, new_im, factor=1.5):
    # RGB space

    orig_im = orig_im.astype(np.float)
    new_im = new_im.astype(np.float)

    enhanced = orig_im + factor * (new_im - orig_im)
    enhanced = np.clip(enhanced, 0, 255)
    return enhanced.astype(np.uint8)


if __name__ == '__main__':
    im_idx = '10846'

    im_path = 'datasets/hair/test_A/' + im_idx + '.jpg'
    hair_mask_path = 'datasets/hair/test_mask/' + im_idx + '.png'
    # ref_path = 'input/reference/dark_brown.jpg'

    im = np.array(imageio.imread(im_path))
    hair_mask = np.array(imageio.imread(hair_mask_path))

    rec = recolor_im(im, hair_mask)

    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(rec)
    # plt.title('rec')
    # plt.figure()
    # plt.imshow(im)
    # plt.title('im')
    # plt.figure()
    # plt.imshow(hair_mask)
    # plt.title('hair_mask')
