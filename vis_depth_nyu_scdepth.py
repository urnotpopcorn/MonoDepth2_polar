import argparse
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from path import Path

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

from utils import *
from kitti_utils import *
from layers import *


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torchvision import models, transforms
from torchvision.utils import make_grid
from PIL import Image

def mkdir_if_not_exists(path):
    """Make a directory if it does not exist.
    Args:
        path: directory to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, log10, rmse, a1, a2, a3


def depth_visualizer(data):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_depth = 1 / (data + 1e-6)
    vmax = np.percentile(inv_depth, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_depth.min(), vmax=vmax)
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    vis_data = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return vis_data


def depth_pair_visualizer(pred, gt):
    """
    Args:
        data (HxW): depth data
    Returns:
        vis_data (HxWx3): depth visualization (RGB)
    """

    inv_pred = 1 / (pred + 1e-6)
    inv_gt = 1 / (gt + 1e-6)

    vmax = np.percentile(inv_gt, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_gt.min(), vmax=vmax)
    # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)
    
    normalizer = mpl.colors.Normalize(vmin=inv_pred.min(), vmax=inv_pred.max())
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)

    return vis_pred, vis_gt


class DepthEvalEigen():
    def __init__(self, opt):
        self.min_depth = opt.min_depth
        self.opt = opt
        self.max_depth = opt.max_depth
        load_weights_folder = opt.load_weights_folder

        epoch = load_weights_folder.split('weights_')[1]
        model = load_weights_folder.split('/')[1]
        # output/vis_nyuv2/${MODEL}
        self.vis_dir = os.path.join('output', 'vis_nyuv2', model, 'weight_'+epoch)

    def main(self):
        # pred_depths = []

        """ Get result """
        # Read precomputed result
        print("load pred depth...")
        # pred_depths = np.load(os.path.join(args.pred_depth))

        """ Evaluation """
        print("load gt depth...")
        gt_depth = 'dataset/NYUv2_rectified/test/depth.npy'
        gt_depths = np.load(gt_depth)
        pred_disps, filename_list, encoder, decoder = self.predict_disps(self.opt)
        pred_depths = self.compute_depth(gt_depths, pred_disps, filename_list, eval_mono=True)

        """ Save result """
        # create folder for visualization result
        if self.vis_dir:
            save_folder = Path(self.vis_dir)/'vis_depth'
            mkdir_if_not_exists(save_folder)

            # image_paths = sorted(Path(args.img_dir).files('*.png'))
            image_paths = sorted(Path('dataset/NYUv2_rectified/test/color').files('*.png'))

            for i in tqdm(range(len(pred_depths))):
                # if os.path.basename(image_paths[i]) != '0188.png':
                #    continue

                # reading image
                img = cv2.imread(image_paths[i], 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # h, w, _ = img.shape
                pred = pred_depths[i]
                h, w = pred.shape 

                cat_img = 0
                cat_img = np.zeros((h, 3*w, 3))
                cat_img[:, :w] = cv2.resize(img, (w,h))
                gt = gt_depths[i]
                vis_pred, vis_gt = depth_pair_visualizer(pred, gt)
                cat_img[:, w:2*w] = vis_pred
                cat_img[:, 2*w:3*w] = cv2.resize(vis_gt, (w,h))

                # save image
                cat_img = cat_img.astype(np.uint8)
                # png_path = os.path.join(save_folder, "{:04}.png".format(i))
                png_path = os.path.join(save_folder, os.path.basename(image_paths[i]))
                cv2.imwrite(png_path, cv2.cvtColor(cat_img, cv2.COLOR_RGB2BGR))

                if os.path.basename(image_paths[i]) == '0188.png':
                    def save_img(tensor, name):
                        tensor = tensor.permute((1, 0, 2, 3))
                        im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
                        im = (im.data.cpu().numpy() * 255.).astype(np.uint8)
                        Image.fromarray(im).save(name + '.jpg')
                        
                    tf = transforms.Compose([transforms.Resize((256, 320)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
                    
                    img = Image.fromarray(img)
                    img = tf(img)
                    img = img.unsqueeze(0).cuda()
                    f1 = encoder.encoder.conv1(img)  # [1, 64, 112, 112]
                    save_img(f1, 'conv1')


    def predict_disps(self, opt):
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
            for ind, (data, gt_depth, _, _, _, _, filename) in enumerate(dataloader):
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
                filename_list.append(filename)

        pred_disps = np.concatenate(pred_disps)
        return pred_disps, filename_list, encoder, depth_decoder

    def compute_depth(self, gt_depths, pred_disps, filename_list=None, eval_mono=True):
        """evaluate depth result
        Args:
            gt_depths (NxHxW): gt depths
            pred_disps (NxHxW): predicted disps
            split (str): data split for evaluation
                - depth_eigen
            eval_mono (bool): use median scaling if True
        """
        errors = []
        ratios = []
        resized_pred_depths = []

        print("==> Evaluating depth result...")
        for i in tqdm(range(pred_disps.shape[0])):
            if pred_disps[i].mean() != -1:
                gt_depth = gt_depths[i]
                gt_height, gt_width = gt_depth.shape[:2]

                # resizing prediction (based on inverse depth)
                pred_inv_depth = pred_disps[i]
                pred_inv_depth = cv2.resize(pred_inv_depth, (gt_width, gt_height))
                pred_depth = 1 / (pred_inv_depth + 1e-6)

                mask = np.logical_and(gt_depth > self.min_depth, gt_depth < self.max_depth)
                val_pred_depth = pred_depth[mask]
                val_gt_depth = gt_depth[mask]

                # median scaling is used for monocular evaluation
                ratio = 1
                if eval_mono:
                    ratio = np.median(val_gt_depth) / np.median(val_pred_depth)
                    ratios.append(ratio)
                    val_pred_depth *= ratio
                    # val_pred_depth *= 31.289

                resized_pred_depths.append(pred_depth * ratio)

                val_pred_depth[val_pred_depth < self.min_depth] = self.min_depth
                val_pred_depth[val_pred_depth > self.max_depth] = self.max_depth

                cur_error = compute_depth_errors(val_gt_depth, val_pred_depth)
                errors.append(cur_error)
                
                if cur_error[0] > 0.4:
                    print(filename_list[i])

        if eval_mono:
            ratios = np.array(ratios)
            med = np.median(ratios)
            print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
            print(" Scaling ratios | mean: {:0.3f} +- std: {:0.3f}".format(np.mean(ratios), np.std(ratios)))

        mean_errors = np.array(errors).mean(0)
        print("\n  " + ("{:>8} | " * 6).format("abs_rel", "log10", "rmse", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 6).format(*mean_errors.tolist()) + "\\\\")

        return resized_pred_depths

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        # print('---------',self.submodule._modules.items())
        #for name, module in self.submodule._modules.items():
        for m in self.submodule.modules():
            x = m(x)
            if isinstance(m, nn.Conv2d):
                outputs.append(x)

            '''
            if "fc" in name:
                x = x.view(x.size(0), -1)
            # print(module)
            x = module(x)

            # print('name', name, self.extracted_layers)
            if name in self.extracted_layers:
                outputs.append(x)
            '''

        return outputs

    
if __name__ == '__main__':
    options = MonodepthOptions()
    eval = DepthEvalEigen(options.parse())
    eval.main()
