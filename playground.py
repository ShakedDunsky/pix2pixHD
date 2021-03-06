from PIL import Image
import glob
import imageio
import numpy as np
import os

from util import util

a_path = 'datasets/cityscapes/train_A/00005.jpg'
b_path = 'datasets/cityscapes/train_B/00005.jpg'


# def list_files_in_dir(root_dir, out_filename, recursive=True):
#     out_f = open(out_filename, 'w+')
#     for filename in glob.iglob(root_dir + '**/*.jpg', recursive=recursive):
#         out_f.write(filename + '\n')


# list_files_in_dir('../research-hair/output/blonde_hair_rocolored/blonde/', 'tar_list.txt')
def list_files_in_dir(root_dir, out_filename, suffix='jpg', recursive=True):
    out_f = open(out_filename, 'w+')
    for i, filename in enumerate(glob.iglob(root_dir + '**/*.' + suffix, recursive=recursive)):
        filename = filename.split('/')
        a_filename = filename[4:]
        a_filename = '/'.join(a_filename)
        b_filename = 'dark_brown/' + filename[5]
        out_f.write(a_filename + '\n' + b_filename + '\n')


if __name__ == '__main__':

    epoch = '120'

    im_list_filename = '../research-hair/filepath_list/test_celebA_brown.txt'
    for im_path in [line.rstrip('\n') for line in open(im_list_filename)]:
        name = (im_path.split('/')[-1]).split('.')[0]
        panel_path = './results/celebA_full/test_' + epoch + '/images/' + name + '_panel.jpg'
        if not os.path.exists(panel_path):
            print(panel_path)
            continue

        panel = np.array(imageio.imread(panel_path))

        hist_path = '../research-hair/CelebA-HQ/synth_blonde/blonde2/' + name + '.jpg'
        hist = np.array(imageio.imread(hist_path))
        out_path = './results/celebA_full/test_' + epoch + '/full_panel_out_2/' + name + '.jpg'
        out = np.concatenate([panel, hist], axis=1)
        util.save_image(out, out_path)
