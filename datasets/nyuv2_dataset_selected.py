import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
from imageio import imread
from path import Path
import random
import os
import cv2

from PIL import Image  # using pillow-simd for increased speed
CROP = 0 # alread cropped

def load_as_float(path):
    return imread(path).astype(np.float32)

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.array(img.convert('RGB'))
            h, w, c = img.shape
            return img

class NYUv2SelectedDataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        data_path/scene_1/images/image0.jpg
        data_path/scene_1/images/image1.jpg
        ..
    
    Pose:
        data_path/computed_poses/poses_bounds.npy
    """

    # def __init__(self, root,
    #             filenames, height, width, frame_idxs, num_scales,
    #             is_train=True, sequence_length=3, transform=None, skip_frames=1):

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 sequence_length=3, skip_frames=1,
                 seq=None):

        super(NYUv2SelectedDataset, self).__init__()
        self.full_res_shape = (640-CROP*2, 480-CROP*2) 
        self.data_path = data_path # Path(data_path)
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        # scene_list_path = self.data_path/'train.txt' if is_train else self.data_path/'val.txt'
        # self.scenes = [self.data_path/folder[:-1] for folder in open(scene_list_path)]
        # scene_list_path = os.listdir(os.path.join(self.data_path, seq, "images"))
        # base_dir = Path(os.path.join(self.data_path, seq, "images"))
        # self.scenes = [base_dir/scene_path for scene_path in scene_list_path]
        self.base_dir = os.path.join(self.data_path, seq, "images")
        # self.scenes = [os.path.join(base_dir, scene_path) for scene_path in scene_list_path]
        self.skip_frames = skip_frames
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = False #self.check_depth()
        self.poses_bounds = np.load(os.path.join(self.data_path, seq, 'computed_poses', 'poses_bounds.npy')) # N x 17
        
        # self.crawl_folders(sequence_length)
        self.crawl_folders()

        
    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                # import pdb; pdb.set_trace()
                for i in range(self.num_scales):                   
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    
    def get_intrinsics(self, i):
        intrinsics = np.zeros([3, 3], dtype=np.float32)
        intrinsics[0][0] = self.poses_bounds[i][14] / self.full_res_shape[0] # focal
        intrinsics[1][1] = self.poses_bounds[i][14] / self.full_res_shape[1] # focal
        intrinsics[0][2] = 0.5 # cx
        intrinsics[1][2] = 0.5 # cy
        intrinsics[2][2] = 1
        return intrinsics
    
    def get_poses(self, rescale=True):
        ''' read abs pose and transform to rel pose

        Output:
        gt_local_poses: [N=1, 4, 4]
        '''

        gt_global_poses = np.copy(self.poses_bounds[:, :15]).reshape(-1, 3, 5)[:, :, :4] # [N, 3, 4]
        if rescale == True:
            gt_global_poses[:, 0, 3] #/= 1000.0
            gt_global_poses[:, 1, 3] #/= 1000.0
            gt_global_poses[:, 2, 3] #/= 1000.0
            
            gt_global_poses[:, 0, 3] *= 0.0
            gt_global_poses[:, 1, 3] *= 0.0
            gt_global_poses[:, 2, 3] *= 0.0
            

        gt_global_poses = np.concatenate(
            (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)

        gt_global_poses[:, 3, 3] = 1
        gt_xyzs = gt_global_poses[:, :3, 3]

        gt_local_poses = []
        for i in range(1, len(gt_global_poses)):
            gt_local_poses.append(
                np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))
        
        gt_local_poses = np.array(gt_local_poses).astype(np.float32)

        return gt_local_poses
        
    def crawl_folders(self):
        pair_set = []
        # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
        # imgs = sorted(scene.files('*.jpg'))
        imgs = sorted([os.path.join(self.base_dir, img) for img in os.listdir(self.base_dir) if '.jpg' in img])
        poses = self.get_poses() # [N-1, 4, 4]
        
        # for i in range(0, len(imgs)-1, 2):
        for i in range(1, len(imgs)-1, 1):
            intrinsic = self.get_intrinsics(i)
            sample = {'intrinsics': intrinsic, 'tgt': imgs[i], 'ref_imgs': [imgs[i-1], imgs[i+1]], 'poses': [poses[i-1], poses[i]]}
            pair_set.append(sample)

        # random.shuffle(pair_set)
        self.samples = pair_set

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = False # self.is_train and random.random() > 0.5

        sample = self.samples[index]

        # for ind, i in enumerate(self.shifts): # -1, 1
        for ind, i in enumerate(self.frame_idxs[1:]): # -1, 1
            inputs[("color", i, -1)] = self.get_color(sample['ref_imgs'][ind], do_flip)
            inputs[("pose", i, 0)] = torch.from_numpy(sample['poses'][ind])
        
        inputs[("color", 0, -1)] = self.get_color(sample['tgt'], do_flip)
        
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # K = self.K.copy()
            K = np.copy(sample['intrinsics'])

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            # K[0, :] = K[0, :] // (2 ** scale)
            # K[1, :] = K[1, :] // (2 ** scale)

            inv_K = np.linalg.pinv(K)
            
            # 3x3 -> 4x4
            row = np.array([[0, 0, 0, 1]], dtype=np.float32)
            col = np.array([[0], [0], [0]], dtype=np.float32)
            K = np.concatenate((K, col), axis=1)
            K = np.concatenate((K, row), axis=0)
            inv_K = np.concatenate((inv_K, col), axis=1)
            inv_K = np.concatenate((inv_K, row), axis=0)
            
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
            
        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            if not i in set([0, -2, -1, 1, 2]):
                continue

            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
        return inputs

    def get_color(self, fp, do_flip):
        color = load_as_float(fp)

        if do_flip:
            color = cv2.flip(color, 1)
            
        h, w, c = color.shape
        color = color[CROP:h-CROP, CROP:w-CROP, :]

        # return Image.fromarray(color)
        return Image.fromarray(np.uint8(color))

    def __len__(self):
        return len(self.samples)
