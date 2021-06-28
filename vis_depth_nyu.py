from PIL import Image
import os
import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity
    
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

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def view_idx(rgb_img, pred_depth, gt_depth, ins_img, sem_img, vis_save_path, error):
    plt.figure(figsize=(12,12))
    plt.subplot(4,2,1)
    plt.imshow(rgb_img)
    plt.title("RGB")

    plt.subplot(4,2,2)
    y, x = np.where(gt_depth > 0)
    d = gt_depth[gt_depth > 0]
    xyd = np.stack((x,y,d)).T
    gt_depth = lin_interp(gt_depth.shape, xyd)
    # gt = lin_interp（input_image.size，image）
    plt.imshow(1.0/gt_depth, cmap = "plasma")
    plt.title("Depth GT")

    plt.subplot(4,2,3)
    plt.imshow(1.0/pred_depth[0], cmap = "plasma")
    plt.title("FeatDepth_{:.3f}_{:.3f}".format(error[0][0], error[0][1]))
    
    plt.subplot(4,2,4)
    plt.imshow(1.0/pred_depth[1], cmap = "plasma")
    plt.title("Stage1_{:.3f}_{:.3f}".format(error[1][0], error[1][1]))

    plt.subplot(4,2,5)
    plt.imshow(pred_depth[0]-gt_depth, cmap = "RdGy")
    plt.title("ErrorMap_FeatDepth")
    
    plt.subplot(4,2,6)
    plt.imshow(pred_depth[1]-gt_depth, cmap = "RdGy")
    plt.title("ErrorMap_Stage1")
    
    t = (error[1][0] - error[0][0]) / error[1][0]
    if t > 0.2:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Worse/High')
    elif t > 0.1:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Worse/Medium')
    elif t > 0:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Worse/Low')
    elif t < -0.2:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Better/High')
    elif t < -0.1:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Better/Medium')
    elif t < -0:
        vis_save_path = vis_save_path.replace('FeatDepth', 'FeatDepth/Better/Low')
    make_dir(os.path.dirname(vis_save_path))
    # print(vis_save_path)
    '''
    plt.subplot(4,2,3)
    plt.imshow(pred[0], cmap = "plasma")
    plt.title("M_416x128_no_SIG_3_depth_median_scale")

    plt.subplot(4,2,4)
    plt.imshow(pred[1], cmap = "plasma")
    plt.title("M_416x128_SIG_104_depth_median_scale")

    plt.subplot(4,2,5)
    plt.imshow(pred[2])
    plt.title("M_416x128_no_SIG_3_disp_original")

    plt.subplot(4,2,6)
    plt.imshow(pred[3])
    plt.title("M_416x128_SIG_104_disp_original")
    '''

    plt.subplot(4,2,7)
    plt.imshow(ins_img)
    plt.title("SIG_ins_img")

    plt.subplot(4,2,8)
    plt.imshow(sem_img)
    plt.title("SIG_sem_img")

    
    plt.show()
    plt.savefig(vis_save_path)
    plt.close()

def get_image_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, ".png") # '0000000268.jpg'

    # '/userhome/34/h3567721/projects/Depth/monodepth2/kitti_data/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000268.jpg'
    # image_path = os.path.join(ori_kitti_data, folder, "image_0{}/data".format(side_map[side]), f_str)
    image_path = os.path.join(folder, "image_0{}/data".format(side_map[side]), f_str)

    return image_path

def get_ins_sem_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, ".npy") # '0000000268.jpg'

    # '/userhome/34/h3567721/projects/Depth/monodepth2/kitti_data/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000268.jpg'
    # image_path = os.path.join(ori_kitti_data, folder, "image_0{}/data".format(side_map[side]), f_str)
    image_path = os.path.join(folder, "image_0{}/data".format(side_map[side]), f_str)

    return image_path

def get_color(image_path):
    return Image.open(image_path).convert('RGB')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def recale_depth(pred_disp, gt_depth):
    gt_height, gt_width = gt_depth.shape[:2]

    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    pred_depth = pred_disp
    pred_depth = 1 / pred_disp

    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    
    pred_depth_mask = pred_depth[mask]
    gt_depth_mask = gt_depth[mask]
    
    # ratio = np.median(pred_depth_mask) / np.median(gt_depth_mask)
    ratio = np.median(gt_depth_mask) / np.median(pred_depth_mask)
    pred_depth *= ratio
    pred_depth_mask *= ratio

    # print(np.mean(pred_depth_mask))

    '''
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    ratio = np.median(gt_depth) / np.median(pred_depth)
    pred_depth *= ratio
    '''

    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    return pred_depth, gt_depth_mask, pred_depth_mask

if __name__ == "__main__":
    src_f_test_path = "splits/eigen/test_files.txt"

    with open(src_f_test_path, "r") as fd:
        src_f_test = fd.read().splitlines()

    side_map = {
        'l': "2",
        "r": "3"
    }

    rgb_path = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/dataset/raw_data"
    gt_path = "/home/xzwu/xzwu/Code/MonoDepth2_splits/eigen/gt_depths.npz"
    
    # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    save_dir = "output/vis_depth/FeatDepth"
    make_dir(save_dir)

    #M_416x128_no_SIG_3 = np.load("/local/xjqi/monodepth-project/tmp/M_416x128_bs_12_worker_4/models/weights_15/disps_eigen_split.npy")
    #M_416x128_SIG_104 = np.load("/local/xjqi/monodepth-project/tmp/M_416x128_SIG_bs_12_worker_12/models/weights_13/disps_eigen_split.npy")
    FeatDepth = np.load("/home/xzwu/xzwu/Code/FeatDepth/log/mono_fm_from_scratch_new/epoch_9_sc/disp_pred.npy")
    # FeatDepth = np.load("/home/xzwu/xzwu/Code/FeatDepth/log/mono_fm_from_scratch/epoch_9/disp_pred.npy")
    FeatDepth_stage1 = np.load("/home/xzwu/xzwu/Code/FeatDepth_stage1/log/mono_fm_stage1/disp_pred.npy")
    # FeatDepth_stage1 = np.load("/home/xzwu/xzwu/Code/FeatDepth_stage1/log/mono_fm_stage1_rpose/epoch_11/disp_pred.npy")
    
    sem_dir = "dataset/kitti_data_sem_eigen_zhou/test"
    ins_dir = "dataset/kitti_data_ins_eigen_zhou/test"

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    for i, sig_file in enumerate(src_f_test):
        # print(sig_file)
        # 2011_10_03/2011_10_03_drive_0034_sync 3247 l
        line = sig_file.split(" ")
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        image_path = get_image_path(folder, frame_index, side)
        
        if '2011_09_26_drive_0101_sync' not in image_path or '556' not in image_path :
            continue

        rgb_img = get_color(os.path.join(rgb_path, image_path))
        sem_ins_path = get_ins_sem_path(folder, frame_index, side)
        ins_img = np.load(os.path.join(ins_dir, sem_ins_path))[:,:,0]
        sem_img = np.load(os.path.join(sem_dir, sem_ins_path))[:,:,0]
        
        gt_depth = gt_depths[i]
        '''
        pred_3_disp = M_416x128_no_SIG_3[i]
        pred_104_disp = M_416x128_SIG_104[i]
        
        pred_3_depth = recale_depth(pred_3_disp, gt_depth)
        pred_104_depth = recale_depth(pred_104_disp, gt_depth)

        pred = [pred_3_depth, pred_104_depth, pred_3_disp, pred_104_disp]
        '''
        pred_disp_FeatDepth = FeatDepth[i]
        
        pred_depth_FeatDepth, gt_depth_mask, pred_depth_mask = recale_depth(pred_disp_FeatDepth, gt_depth)
        
        # d1 = pred_depth_FeatDepth[150:200, 575:625]
        # d2 = gt_depth[150:200, 575:625]
        # d2 = d2[d2>0]
        # print(d1.max(), d1.min(), d2.max(), d2.min())
        
        error_FeatDepth = compute_errors(gt_depth_mask, pred_depth_mask)
        
        pred_disp_FeatDepth_stage1 = FeatDepth_stage1[i]
        pred_depth_FeatDepth_stage1, gt_depth_mask, pred_depth_mask = recale_depth(pred_disp_FeatDepth_stage1, gt_depth)
        
        error_FeatDepth_stage1 = compute_errors(gt_depth_mask, pred_depth_mask)

        pred = [pred_depth_FeatDepth, pred_depth_FeatDepth_stage1]
        error = [error_FeatDepth, error_FeatDepth_stage1]

        save_name = folder.split("/")[1] + "_" + str(frame_index) + "_" + side
        vis_save_path = os.path.join(save_dir, save_name + ".jpg")

        view_idx(rgb_img, pred, gt_depth, ins_img, sem_img, vis_save_path, error)
