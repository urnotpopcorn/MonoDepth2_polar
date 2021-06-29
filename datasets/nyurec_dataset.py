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
CROP = 16

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

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    norm = np.array(h5f['norm'])
    norm = np.transpose(norm, (1,2,0))
    valid_mask = np.array(h5f['mask'])

    return rgb, depth, norm, valid_mask

class NYURecDataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
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
                 img_ext=None, opt=None, mode='train',
                 sequence_length=3, skip_frames=1):

        super(NYURecDataset, self).__init__()
        self.full_res_shape = (640-CROP*2, 480-CROP*2) 
        self.data_path = Path(data_path)
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        scene_list_path = self.data_path/'train.txt' if is_train else self.data_path/'val.txt'
        self.scenes = [self.data_path/folder[:-1] for folder in open(scene_list_path)]
        # self.transform = transform
        self.skip_frames = skip_frames
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        if self.is_train: self.loader = pil_loader
        else: self.loader = h5_loader
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
    
    def crawl_folders(self):
        pair_set = []
        for scene in self.scenes:
            try:
                # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
                imgs = sorted(scene.files('*.jpg'))
                intrinsics = sorted(scene.files('*.txt'))
                
                # for i in range(0, len(imgs)-1, 2):
                for i in range(1, len(imgs)-1, 2):
                    intrinsic = np.genfromtxt(intrinsics[int(i/2)]).astype(np.float32).reshape((3, 3))
                    sample = {'intrinsics': intrinsic, 'tgt': imgs[i], 'ref_imgs': [imgs[i-1], imgs[i+1]]}
                    pair_set.append(sample)
            except Exception as e:
                print(e)
                print(scene)
                input()

        random.shuffle(pair_set)
        self.samples = pair_set

    def crawl_folders_seq(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        self.shifts = list(range(-demi_length * self.skip_frames, demi_length * self.skip_frames + 1, self.skip_frames))
        self.shifts.pop(demi_length)
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            # imgs = sorted(scene.files('*.jpg'))
            imgs = sorted([img_file for img_file in scene.files('*.jpg') if '_1' not in img_file])
            
            if len(imgs) < sequence_length:
                continue
            if not os.path.exists(scene/'000000_cam.txt') and not os.path.exists(scene/'cam.txt'):
                continue

            if os.path.exists(scene/'cam.txt'):
                intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))

            for i in range(demi_length * self.skip_frames, len(imgs)-demi_length * self.skip_frames):
                cam_file = os.path.basename(imgs[i]).split('_')[0]
                if os.path.exists(scene/'cam.txt'):
                    pass
                elif os.path.exists(os.path.join(scene, cam_file+'_cam.txt')):
                    intrinsics = np.genfromtxt(os.path.join(scene, cam_file+'_cam.txt')).astype(np.float32).reshape((3, 3))
                else:
                    print('intrinsics invalid: ', os.path.join(scene, cam_file+'_cam.txt'))
                
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in self.shifts: # -1, 1
                    sample['ref_imgs'].append(imgs[i+j])
                    
                sequence_set.append(sample)
            
        random.shuffle(sequence_set)
        if len(sequence_set) == 0:
            print(scene)
            input()
            
        self.samples = sequence_set
    
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        sample = self.samples[index]
        '''
        for ind, i in enumerate(self.frame_idxs):
            # self.frame_idxs: 0, -1, 1, sample: t-1, t+1
            if not i in set([0, -2, -1, 1, 2]):
                continue
            inputs[("color", i, -1)] = self.get_color(sample['ref_imgs'][ind], do_flip)
        '''
        # for ind, i in enumerate(self.shifts): # -1, 1
        for ind, i in enumerate(self.frame_idxs[1:]): # -1, 1
            inputs[("color", i, -1)] = self.get_color(sample['ref_imgs'][ind], do_flip)
        
        inputs[("color", 0, -1)] = self.get_color(sample['tgt'], do_flip)
        
        # img = np.array(inputs[("color", 0, -1)])
        # img = load_as_float(sample['tgt'])
        # ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            # K = self.K.copy()
            K = np.copy(sample['intrinsics'])

            # K[0, :] *= self.width // (2 ** scale)
            # K[1, :] *= self.height // (2 ** scale))
            K[0, :] = K[0, :] // (2 ** scale)
            K[1, :] = K[1, :] // (2 ** scale)

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
        '''
        if fp in self.img_cache:
            color = self.img_cache[fp]
        else:
            color = self.loader(fp)
            if not(self.debug):
                color = self._undistort(color)
            self.img_cache[fp] = color
        '''
        color = load_as_float(fp)

        if do_flip:
            color = cv2.flip(color, 1)
            
        h, w, c = color.shape
        color = color[CROP:h-CROP, CROP:w-CROP, :]

        # return Image.fromarray(color)
        return Image.fromarray(np.uint8(color))

    def __len__(self):
        return len(self.samples)
