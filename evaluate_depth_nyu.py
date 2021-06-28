from __future__ import absolute_import, division, print_function

import os, sys
import cv2
sys.path.append(os.getcwd())
import numpy as np

import torch
import torch.nn.functional as F
import datasets
import networks
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    lg10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, lg10, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    # MIN_DEPTH = 1e-3
    # MAX_DEPTH = 80
    
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    encoder = networks.ResnetEncoder(opt.num_layers, False)
        
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    
    if opt.png:
        img_ext = '.png'
    else:
        img_ext = '.jpg'
        
    thisH, thisW = encoder_dict['height'], encoder_dict['width']
    
    filenames = readlines('./splits/nyuv2/test_files.txt')
    dataset = datasets.NYUTestDataset(
            opt.data_path,
            filenames,
            thisH, thisW,
    )
    dataloader = DataLoader(
            dataset, 1, shuffle=False, 
            num_workers=opt.num_workers
    )
    
    pred_disps = []
    filename_list = []
    input_color_list = []

    with torch.no_grad():
        gt_depths = list()
        for ind, (data, gt_depth, _, _, _, _, _) in enumerate(dataloader):
            input_color = data.cuda() # [0, 1]
            input_color_list.append(input_color)
            
            output = depth_decoder(encoder(input_color))

            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if opt.post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

            pred_disps.append(pred_disp)
            gt_depths.append(gt_depth.data.numpy()[0,0])
            # filename_list.append(filename)

    pred_disps = np.concatenate(pred_disps)

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < opt.min_depth] = opt.min_depth
        pred_depth[pred_depth > opt.max_depth] = opt.max_depth

        error = compute_errors(gt_depth, pred_depth)
        
        # if error[0] > 0.3:
        #     input_color = input_color_list[i].permute(0, 2, 3, 1).cpu().numpy()
        #     input_color = Image.fromarray(np.uint8(255.0 * input_color[0]))
        #     output_filename = filename_list[i][0].split('/')[-1].split('.')[0]
            
        #     output_rgb_filename = os.path.join('fail_cases', output_filename+".jpg")
        #     input_color.save(output_rgb_filename)

        #     output_d_filename = os.path.join('fail_cases', output_filename+"_d.jpg")
        #     plt.imshow(pred_disp, cmap="plasma")
        #     plt.axis('off')
        #     plt.savefig(output_d_filename, transparent=True)
        #     plt.close()

        errors.append(error)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        #print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print(("{: 8.3f}\t" * 8).format(*mean_errors.tolist()))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
