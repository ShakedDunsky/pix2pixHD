import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer, get_out_panel
from util import html
import torch
import ntpath
import numpy as np
import imageio
from skimage.transform import resize


from transfer_recolor.recolor import apply_mask, enhance_brightening, recolor_im

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

dst_im = np.array(imageio.imread('../research-hair/input/reference/blonde.jpg'))
dst_im_out = resize(dst_im, (1024, 1024))

print('len dataset', len(dataset))

for i, data in enumerate(dataset):

    # panel_path = visualizer.save_panel(webpage, None, data['path'], None, save=False)
    # if os.path.exists(panel_path):
    #     continue

    # if i >= opt.how_many:   # shouldn't happen
    #     break

    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
    if opt.export_onnx:
        print("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        generated = model.inference(data['label'], data['inst'], data['image'])

    img_path = data['path']
    print('process image %d... %s' % (i, img_path))

    orig_im = util.tensor2label(data['label'][0], opt.label_nc)
    mask = util.tensor2label(data['mask'][0], opt.label_nc)
    synthesized_im = util.tensor2im(generated.data[0])
    masked_im = apply_mask(orig_im, synthesized_im, mask)
    enhanced = enhance_brightening(orig_im, masked_im)
    enhanced1 = enhance_brightening(orig_im, masked_im, factor=1.75)
    enhanced2 = enhance_brightening(orig_im, masked_im, factor=2)

    recolored = recolor_im(orig_im, mask, dst_im)
    recolored_from_synth = recolor_im(synthesized_im, mask, dst_im)

    # visuals = OrderedDict([('input_label', orig_im),
    #                        ('synthesized_image', synthesized_im),
    #                        ('masked_image', masked_im),
    #                        ('enhanced', enhanced)
    #                        ])

    # visualizer.save_images(webpage, visuals, img_path)

    visualizer.save_panel(webpage,
                          [orig_im, synthesized_im, masked_im, enhanced, enhanced1, enhanced2,
                              recolored, recolored_from_synth, dst_im_out],
                          img_path,
                          'panel')

# webpage.save()
