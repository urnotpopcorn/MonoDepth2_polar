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


from utils import *
from kitti_utils import *
from layers import *

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
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, use_polar=True, use_computing_polar_phi=opt.use_computing_polar_phi)

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
    
    pred_dists = []
    pred_phis = []
    # pred_depths = []
    # pred_phis = []

    with torch.no_grad():
        gt_depths = list()
        for ind, (data, gt_depth, _, _, K, _) in enumerate(dataloader):
            input_color = data.cuda() # [0, 1]
            output = depth_decoder(encoder(input_color))

            _, dist = disp_to_depth(output[("inv_dist", 0)], opt.min_depth, opt.max_depth)
            
            if opt.use_computing_polar_phi:
                meshgrid = np.meshgrid(range(opt.width), range(opt.height), indexing='xy')                   
                meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
                meshgrid = nn.Parameter(torch.from_numpy(meshgrid),
                                            requires_grad=False).cuda() # [2, 192, 640]
                                            
                img_x = meshgrid[0] - opt.width * 0.5
                img_y = meshgrid[1] - opt.height * 0.5
                camera_fx = K[0][0][0] # 0.58 * opt.width # fx
                camera_fy = K[0][1][1] # 1.92 * opt.height # fy
                # # phi = torch.atan(camera_f / torch.sqrt(img_x.pow(2)+img_y.pow(2)))
                # tan_phi = torch.sqrt(img_x.pow(2)/np.power(camera_fx, 2)+img_y.pow(2)/np.power(camera_fy, 2))
                tan_phi = torch.where(img_x > 0, \
                        torch.sqrt(img_x.pow(2)/np.power(camera_fx, 2)+img_y.pow(2)/np.power(camera_fy, 2)), \
                        -torch.sqrt(img_x.pow(2)/np.power(camera_fx, 2)+img_y.pow(2)/np.power(camera_fy, 2)))
                ori_phi = torch.atan(tan_phi)
            else:
                ori_phi = np.pi * (output[("ori_phi", 0)] - 0.5)
            
            pred_dists.append(dist.cpu()[:, 0].numpy())
            pred_phis.append(ori_phi.cpu()[:, 0].numpy())
            gt_depths.append(gt_depth.data.numpy()[0,0])

    pred_dists = np.concatenate(pred_dists)
    pred_phis = np.concatenate(pred_phis)

    errors = []
    ratios = []

    for i in range(pred_dists.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_dist = pred_dists[i]
        pred_dist = cv2.resize(pred_dist, (gt_width, gt_height))

        pred_phi = pred_phis[i]
        if not opt.use_computing_polar_phi:
            pred_phi = cv2.resize(pred_phi, (gt_width, gt_height))

        pred_depth = pred_dist * np.cos(pred_phi)

        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < opt.min_depth] = opt.min_depth
        pred_depth[pred_depth > opt.max_depth] = opt.max_depth

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        #print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print(("{: 8.3f}\t" * 8).format(*mean_errors.tolist()))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
