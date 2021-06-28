import os
import torch
from imageio import imread
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import networks

from utils import *
from kitti_utils import *
from layers import *

#import models
#from inverse_warp import pose_vec2mat


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=320, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=5)
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument("--load_weights_folder", type=str)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def read_imgs(data_root, scene, seq_length=5, step=1):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
    
    pose_list = np.array(np.load(os.path.join(data_root, 'pose', scene+'.npy'))).astype(np.float64)
    
    # construct 5-snippet sequences
    img_dir = os.path.join(data_root, 'colmap', scene, 'images')
    img_list = sorted([os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)])

    if len(pose_list) != len(img_list):
        print(len(pose_list), len(img_list))
        return None, None, None

    if len(img_list) > 2 * demi_length:
        tgt_indices = np.arange(demi_length, len(img_list) - demi_length).reshape(-1, 1)
        snippet_indices_list = shift_range + tgt_indices
    else:
        snippet_indices_list = [np.arange(len(img_list))]

    for snippet_indices in snippet_indices_list:
        imgs = [imread(img_list[i]).astype(np.float32) for i in snippet_indices]
        poses = np.stack([pose_list[i] for i in snippet_indices])
        
        first_pose = poses[0]
        poses[:,:,-1] -= first_pose[:,-1]
        # tgt: (5, 3, 4) (3, 4) (5, 3, 4)
        # cur: (5, 4, 4) (4, 4)
        compensated_poses = np.linalg.inv(first_pose[:3, :3]) @ poses[:, :3, :]
        print(poses)
        print(compensated_poses)
        im_sequences.append(imgs)
        # scene_list.append(scenes)
        poses_sequences.append(compensated_poses)
        indices_sequences.append(snippet_indices)
        
    return im_sequences, poses_sequences, indices_sequences

@torch.no_grad()
def main():
    args = parser.parse_args()
    seq_length = 5
    '''
    from nyuv2_eval.pose_evaluation_utils import test_framework_NYUv2 as test_framework
    weights = torch.load(args.pretrained_posenet)
    pose_net = models.PoseResNet(18, False).to(device)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    '''

    pose_encoder_path = os.path.join(args.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(args.load_weights_folder, "pose.pth")
    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    '''
    dataset_dir = Path(args.dataset_dir)
    framework = test_framework(dataset_dir, args.sequences, seq_length)
    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    '''
    dataset_dir = os.path.join(args.dataset_dir, 'colmap')
    scene_list = os.listdir(dataset_dir)
    ATE_list = list()
    RE_list = list()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    '''
    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']
    '''
    for j, scene in enumerate(tqdm(scene_list)):
        if scene != 'bathroom_0018':
            continue
        im_sequences, poses_sequences, indices_sequences = read_imgs(args.dataset_dir, scene)
        if not im_sequences:
            print(scene)
            continue

        for img_idx, imgs in enumerate(im_sequences):
            try:
                h, w, _ = imgs[0].shape

                if (not args.no_resize) and (h != args.img_height or w != args.img_width):
                    imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

                imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

                squence_imgs = []
                for i, img in enumerate(imgs):
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = ((img/255 - 0.45)/0.225).to(device)
                    squence_imgs.append(img)

                global_pose = np.eye(4)
                poses = []
                poses.append(global_pose[0:3, :])
                real_seq_length = len(squence_imgs)

                for iter in range(real_seq_length - 1):
                    # pose = pose_net(squence_imgs[iter], squence_imgs[iter + 1])
                    # [1, 3, 256, 320]) 
                    all_color_aug = torch.cat([squence_imgs[iter], squence_imgs[iter + 1]], 1)
                    features = [pose_encoder(all_color_aug)]
                    axisangle, translation = pose_decoder(features)

                    # pose, torch.Size([1, 6]
                    pose_mat = transformation_from_parameters(axisangle[:, 0], translation[:, 0]).squeeze().cpu().numpy()
                    # pose_mat = pred_pose.squeeze(0).cpu().numpy()
                    # pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
                    global_pose = global_pose @  np.linalg.inv(pose_mat)
                    poses.append(global_pose[0:3, :])

                final_poses = np.stack(poses, axis=0)  # [5, 3, 4], [N, 3, 4]
                final_poses_gt = poses_sequences[img_idx] # [N, 4, 4]
                ATE, RE = compute_pose_error(final_poses_gt, final_poses)
                '''
                if scene == 'bathroom_0016':
                    print(final_poses)
                    print()
                    print(final_poses_gt)
                    print()
                    print(RE)
                '''

                ATE_list.append(ATE)
                RE_list.append(RE)

                '''
                if args.output_dir is not None:
                    predictions_array[j] = final_poses
                '''
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(e)
                
        '''
        ATE, RE = compute_pose_error(poses_gt, final_poses)
        errors[j] = ATE, RE
        '''
    ATE_list = np.array(ATE_list)
    RE_list = np.array(RE_list)

    print(ATE_list.mean(), ATE_list.std())
    print(RE_list.mean(), RE_list.std())
    
    '''
    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)
    '''

def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2+1e-6)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        pose = pose_net(tgt_img, ref_img).unsqueeze(1)
        poses.append(pose)
    poses = torch.cat(poses, 1)
    return poses


if __name__ == '__main__':
    main()
