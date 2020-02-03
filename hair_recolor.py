import numpy as np
import sys


def apply_mask(orig_im, recolored_im, hair_mask):
    alpha = np.array(hair_mask, dtype=float)
    if len(alpha.shape) == 2:
        alpha = alpha[:, :, np.newaxis]
    alpha = alpha / np.max(alpha)
    print(hair_mask.shape)
    recolored_only_hair = np.multiply(1 - alpha, orig_im) + np.multiply(alpha, recolored_im)
    recolored_only_hair = recolored_only_hair.astype(np.uint8)

    return recolored_only_hair


if __name__ == '__main__':
    orig_im = sys.argv[1]
    new_im = sys.argv[2]
    hair_mask = sys.argv[3]
