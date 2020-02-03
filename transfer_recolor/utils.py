
import os
import glob
import numpy as np
import imageio
from PIL import Image


def resize_image_to_squre(image, size):
    m1 = max(image.size[0], image.size[1])

    padd_0 = max(image.size[1] - image.size[0],0)
    padd_1 = max(image.size[0] - image.size[1],0)

    image_square = np.asarray(image)
    image3 = np.zeros((m1, m1, 3)).astype(np.uint8)
    for ch in range(3):
        image3[:, :, ch] = np.pad(
            image_square[:, :, ch], [(0, padd_1), (0, padd_0)], mode='constant', constant_values=0
        )
    image_square = Image.fromarray(image3)

    new_size = np.array([size, size]).astype(int)
    image2 = image_square.resize(new_size, Image.ANTIALIAS)

    return image2, padd_0, padd_1


def name_from_path(path):
    name = path.split('/')
    name = name[-1]
    return name.split('.')[0]


def imsave(path, im=None):
    if (im is None) and (path[-1] != '/'):
        path = path + '/'
    dir_name = path.split('/')
    dir_name = '/'.join(dir_name[:-1])
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    if im is not None:
        if np.max(im) <= 1:
            im = im * 255
        im = np.array(im, dtype=np.uint8)
        imageio.imsave(path, im)


# data shape = [n, 3]
def pca(data):
    data = data - np.mean(data, axis=0)
    mat = np.transpose(data) @ data
    w, v = np.linalg.eig(mat)
    return w, v


def create_im_list(root_dir, out_filename):
#create_im_list('/Volumes/MyPassport/hair/hair256_all/', '/Volumes/MyPassport/hair/hair256_all/train_im_256.txt')
    if root_dir[-1] != '/':
        root_dir = root_dir + '/'
    out_f = open(out_filename, 'w+')
    # for filename in glob.iglob(root_dir + '**/train/face/*.png', recursive=True):
    for filename in glob.iglob(root_dir + '*.jpg', recursive=True):
        # print(filename)
        out_f.write(filename + '\n')