from PIL import Image
import glob

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

