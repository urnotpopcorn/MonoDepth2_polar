# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import lpips

from PIL import Image  # using pillow-simd for increased speed

# nyuv2
from multiprocessing import Manager
# Init, get rid of slow io
manager = Manager()
shared_dict = manager.dict()

# def disp_to_depth(disp, min_depth, max_depth):
#     depth = 1.0 / disp
#     return disp, depth

from find_matches_sfm import find_correspondence_points


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        if False: #self.opt.use_pose_gt:
            self.use_pose_net = False
        else:
            self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        if self.opt.use_polar:
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales, use_polar=True, use_computing_polar_phi=self.opt.use_computing_polar_phi)
        else:
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)

        if self.opt.depth_decoder_normal_init:
            def weight_init(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
            self.models["depth"].apply(weight_init)

        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask: # no
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "nyuv2": datasets.NYUDataset,
                         "nyuv2rec": datasets.NYURecDataset,
                         "nyuv2_selected": datasets.NYUv2SelectedDataset
                         }

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.dataset == 'nyuv2':
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, shared_dict=shared_dict) # nyuv2
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, shared_dict=shared_dict) # nyuv2
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
            self.val_iter = iter(self.val_loader)
        elif self.opt.dataset == 'nyuv2rec':
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
        elif self.opt.dataset == 'nyuv2_selected':
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, seq='computer_lab_0002')
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
        else:
            train_dataset = self.dataset(
                self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
            val_dataset = self.dataset(
                self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
            self.train_loader = DataLoader(
                train_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
            self.val_loader = DataLoader(
                val_dataset, self.opt.batch_size, True,
                num_workers=self.opt.num_workers, pin_memory=False, drop_last=True)
            self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        # print("There are {:d} training items and {:d} validation items\n".format(
        #    len(train_dataset), len(val_dataset)))

        self.save_opts()
        
        # compute mask
        if self.opt.rotation_constraint:
            self.tgt_mask = dict()
            for scale in self.opt.scales:
                cur_width = self.opt.width // (2 ** scale)
                cur_height = self.opt.height // (2 ** scale)

                # crop = np.array([0.1 * cur_height, 0.9 * cur_height,
                #                 0.1 * cur_width,  0.9 * cur_width]).astype(np.int32)
                # tgt_mask = np.zeros((self.opt.batch_size, 1, cur_height, cur_width))
                # tgt_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1
                # self.tgt_mask[scale] = torch.from_numpy(tgt_mask).float().cuda()  
                self.tgt_mask[scale] = torch.ones((self.opt.batch_size, 1, cur_height, cur_width)).cuda()  
    
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        # print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if batch_idx % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses)

            # if early_phase or late_phase:
            # if batch_idx % self.opt.log_frequency == 0:
            if self.step % self.opt.log_frequency == 0:
                # self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)
        
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        if self.opt.use_polar:
            self.generate_images_pred_polar(inputs, outputs)
        elif self.opt.use_z_theta:
            self.generate_images_pred_z_theta(inputs, outputs)
        else:
            self.generate_images_pred(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    
                    if self.opt.rotation_constraint:
                        # outputs[("cam_T_cam_rot", 0, f_i)] = self.rotation_transformation_from_parameters(
                        #     axisangle[:, 0], invert=(f_i < 0))
                        if self.opt.random_rot:
                            outputs[("cam_T_cam_rot", 0, f_i)] = transformation_from_parameters(
                                0.5 * torch.rand(self.opt.batch_size, 1, 3).cuda(), torch.zeros_like(translation[:, 0]), invert=(f_i < 0))
                        else:
                            outputs[("cam_T_cam_rot", 0, f_i)] = transformation_from_parameters(
                                axisangle[:, 0], torch.zeros_like(translation[:, 0]), invert=(f_i < 0))
                    
        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def rotation_transformation_from_parameters(self, axisangle, invert=False):
        """Convert the network's (axisangle) output into a 4x4 matrix
        """
        R = rot_from_axisangle(axisangle)
        # t = translation.clone()

        if invert:
            R = R.transpose(1, 2)
            # t *= -1

        # T = get_translation_matrix(t)

        # if invert:
        #     M = torch.matmul(R, T)
        # else:
        #     M = torch.matmul(T, R)

        # return M
        return R

    def compute_phi(self, inputs, scale, cur_height, cur_width):
        meshgrid = np.meshgrid(range(cur_width), range(cur_height), indexing='xy')                   
        meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
        meshgrid = nn.Parameter(torch.from_numpy(meshgrid),
                                    requires_grad=False).cuda() # [2, 192, 640]
                                    
        img_x = meshgrid[0] - cur_width * 0.5
        img_y = meshgrid[1] - cur_height * 0.5
        img_x = img_x.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
        img_y = img_y.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
        camera_fx = 1.0 / inputs[("K", scale)][:, 0:1, 0:1].repeat(1, cur_height, cur_width).unsqueeze(1) # fx
        camera_fy = 1.0 / inputs[("K", scale)][:, 1:2, 1:2].repeat(1, cur_height, cur_width).unsqueeze(1) # fy
        img_x_camera_fx = img_x * camera_fx
        img_y_camera_fy = img_y * camera_fy

        # tan_phi = torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy)
        tan_phi = torch.where(img_x > 0, \
                torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy), \
                -torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy))
        phi = torch.atan(tan_phi)
        return phi

    def generate_images_pred_z_theta(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            # compute disp
            disp = outputs[("disp", scale)]
            source_scale = 0
            disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)    
            
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            
            # compute phi
            # if self.opt.use_computing_polar_phi:
            phi = self.compute_phi(inputs, source_scale, self.opt.height, self.opt.width)
            
            # compute depth
            outputs[("depth", 0, scale)] = depth
            dist = depth / torch.cos(phi)
            outputs[("dist", scale)] = dist
            outputs[("phi", 0, scale)] = phi
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                img_pred = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                outputs[("color", frame_id, scale)] = F.interpolate(
                    img_pred, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                # if self.opt.use_polar and self.opt.rotation_constraint and (self.opt.with_multi_scale or scale == 0):
                if self.opt.rotation_constraint and (self.opt.with_multi_scale or scale == 0):
                    # compute correspondence (only rotation)
                    T_rot = outputs[("cam_T_cam_rot", 0, frame_id)]
                    pix_coords_rot = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T_rot)

                    # forward warp from tgt system to intermediate system
                    img_rot = self.forward_warp(inputs[("color", 0, source_scale)], pix_coords_rot)
                    img_rot_mask = self.forward_warp(self.tgt_mask[source_scale], pix_coords_rot)
                    img_rot = img_rot * img_rot_mask
                    outputs[("img_rot", frame_id, scale)] = img_rot

                    # predict inv_dist in intermediate system 
                    outputs_rot = \
                        self.models["depth"](self.models["encoder"](img_rot))
                    
                    # inv_dist_rot = outputs_rot[("inv_dist", source_scale)]
                    disp_rot = outputs_rot[("disp", source_scale)]
                    disp_rot = F.interpolate(
                        disp_rot, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    _, depth_rot = disp_to_depth(disp_rot, self.opt.min_depth, self.opt.max_depth)
                    phi = self.compute_phi(inputs, source_scale, self.opt.height, self.opt.width)
                    dist_rot = depth_rot / torch.cos(phi)
                    
                    # backward warp inv_dist from intermediate system to tgt system
                    outputs[("dist_rot", frame_id, scale)] = F.grid_sample(
                        dist_rot,
                        pix_coords_rot,
                        padding_mode="border")
                    outputs[("img_rot_mask", frame_id, scale)] = F.grid_sample(
                        img_rot_mask,
                        pix_coords_rot) # without padding
                    outputs[("total_mask", frame_id, scale)] = outputs[("img_rot_mask", frame_id, scale)] * self.tgt_mask[source_scale]
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
   
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if self.opt.use_pose_gt:
                    T = inputs[("pose", frame_id, 0)]

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    T2 = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0])
                    T = torch.bmm(T2, T)

                    if frame_id < 0:
                        T = torch.inverse(T)
                else:        
                    if frame_id == "s":
                        T = inputs["stereo_T"]
                    else:
                        T = outputs[("cam_T_cam", 0, frame_id)]

                    # from the authors of https://arxiv.org/abs/1712.00175
                    if self.opt.pose_model_type == "posecnn":

                        axisangle = outputs[("axisangle", 0, frame_id)]
                        translation = outputs[("translation", 0, frame_id)]

                        inv_depth = 1 / depth
                        mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                        T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                    
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def forward_warp(self, tgt_img, pix_coords_rot):
        """Generate the warped (reprojected) color images using forward warping.
        
        Args:
            tgt_img: image used to warp, [b, c, self.opt.height, self.opt.width]
            pix_coords_rot: correspondence between images, [b, self.opt.height, self.opt.width, 2]

        Returns:
            img_rot: the warping image, [b, c, self.opt.height, self.opt.width]
        """
        _, num_channels, cur_height, cur_width = tgt_img.shape
        tgt_img = tgt_img.view(-1, num_channels, cur_height*cur_width) # b, c, h*w
        img_rot = torch.zeros_like(tgt_img) 
        index_tgt_x = (pix_coords_rot[..., 0] / 2.0 + 0.5) * (cur_width - 1)
        index_tgt_y = (pix_coords_rot[..., 1] / 2.0 + 0.5) * (cur_height - 1)
        index_tgt = index_tgt_y.long() * cur_width + index_tgt_x.long()

        index_tgt = index_tgt.view(-1, cur_height*cur_width).detach()
        index_tgt = torch.clamp(index_tgt, 0, cur_height*cur_width-1)
        
        for channel in range(num_channels):
            img_rot[:, channel, :] = img_rot[:, channel, :].scatter(1, index_tgt, tgt_img[:, channel, :])
            # img_rot[:, 1, :] = img_rot[:, 1, :].scatter(1, index_tgt, tgt_img[:, 1, :])
            # img_rot[:, 2, :] = img_rot[:, 2, :].scatter(1, index_tgt, tgt_img[:, 2, :]) # b, 122880
        
        img_rot = img_rot.view(-1, num_channels, cur_height, cur_width)
        tgt_img = tgt_img.view(-1, num_channels, cur_height, cur_width)

        img_rot = F.interpolate(
            img_rot, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        
        return img_rot
        
    def generate_images_pred_polar(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            source_scale = 0
            cur_width = self.opt.width // (2 ** source_scale)
            cur_height = self.opt.height // (2 ** source_scale)

            # compute dist
            inv_dist = outputs[("inv_dist", scale)]
            inv_dist = F.interpolate(
                inv_dist, [cur_height, cur_width], mode="bilinear", align_corners=False)
            _, dist = disp_to_depth(inv_dist, self.opt.min_depth, self.opt.max_depth)
            
            # compute phi
            if self.opt.use_computing_polar_phi:
                meshgrid = np.meshgrid(range(cur_width), range(cur_height), indexing='xy')                   
                meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
                meshgrid = nn.Parameter(torch.from_numpy(meshgrid),
                                            requires_grad=False).cuda() # [2, 192, 640]
                                            
                img_x = meshgrid[0] - cur_width * 0.5
                img_y = meshgrid[1] - cur_height * 0.5
                img_x = img_x.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
                img_y = img_y.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
                camera_fx = 1.0 / inputs[("K", source_scale)][:, 0:1, 0:1].repeat(1, cur_height, cur_width).unsqueeze(1) # fx
                camera_fy = 1.0 / inputs[("K", source_scale)][:, 1:2, 1:2].repeat(1, cur_height, cur_width).unsqueeze(1) # fy
                img_x_camera_fx = img_x * camera_fx
                img_y_camera_fy = img_y * camera_fy

                # tan_phi = torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy)
                tan_phi = torch.where(img_x > 0, \
                        torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy), \
                        -torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy))
                phi = torch.atan(tan_phi)
            else:
                ori_phi = np.pi * (outputs[("ori_phi", scale)] - 0.5) # [0, 1] -> [-pi/2.0, pi/2.0] instead of [0, 2*pi]
                phi = F.interpolate(
                    ori_phi, [cur_height, cur_width], mode="bilinear", align_corners=False)

            # compute depth
            depth = dist * torch.cos(phi)
            outputs[("depth", 0, scale)] = depth
            outputs[("dist", scale)] = dist
            outputs[("phi", 0, scale)] = phi
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                img_pred = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")
                outputs[("color", frame_id, scale)] = F.interpolate(
                    img_pred, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                if self.opt.rotation_constraint and (self.opt.with_multi_scale or scale == 0):
                    # compute correspondence (only rotation)
                    T_rot = outputs[("cam_T_cam_rot", 0, frame_id)]
                    pix_coords_rot = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T_rot)

                    # forward warp from tgt system to intermediate system
                    img_rot = self.forward_warp(inputs[("color", 0, source_scale)], pix_coords_rot)
                    img_rot_mask = self.forward_warp(self.tgt_mask[source_scale], pix_coords_rot)
                    img_rot = img_rot * img_rot_mask
                    outputs[("img_rot", frame_id, scale)] = img_rot

                    # predict inv_dist in intermediate system 
                    outputs_rot = \
                        self.models["depth"](self.models["encoder"](img_rot))
                    inv_dist_rot = outputs_rot[("inv_dist", source_scale)]
                    inv_dist_rot = F.interpolate(
                        inv_dist_rot, [cur_height, cur_width], mode="bilinear", align_corners=False)

                    # backward warp inv_dist from intermediate system to tgt system
                    outputs[("inv_dist_rot", frame_id, scale)] = F.grid_sample(
                        inv_dist_rot,
                        pix_coords_rot,
                        padding_mode="border")
                    outputs[("img_rot_mask", frame_id, scale)] = F.grid_sample(
                        img_rot_mask,
                        pix_coords_rot) # without padding
                    outputs[("total_mask", frame_id, scale)] = outputs[("img_rot_mask", frame_id, scale)] * self.tgt_mask[source_scale]
                    
                    # from inv_dist to dist
                    _, dist_rot = disp_to_depth(outputs[("inv_dist_rot", frame_id, scale)], self.opt.min_depth, self.opt.max_depth)
                    outputs[("dist_rot", frame_id, scale)] = dist_rot
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
   
    def generate_images_pred_polar_halfsc(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            # compute dist
            inv_dist = outputs[("inv_dist", scale)]
            inv_dist = F.interpolate(
                inv_dist, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            
            _, dist = disp_to_depth(inv_dist, self.opt.min_depth, self.opt.max_depth)

            source_scale = 0
            # compute phi
            if self.opt.use_computing_polar_phi:
                # meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
                # self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
                # self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                #                             requires_grad=False)

                # self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                #                         requires_grad=False)

                # self.pix_coords = torch.unsqueeze(torch.stack(
                #     [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
                # self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
                # self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                #                             requires_grad=False)
                meshgrid = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')                   
                meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
                meshgrid = nn.Parameter(torch.from_numpy(meshgrid),
                                            requires_grad=False).cuda() # [2, 192, 640]
                                            
                img_x = meshgrid[0] - self.opt.width * 0.5
                img_y = meshgrid[1] - self.opt.height * 0.5
                img_x = img_x.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
                img_y = img_y.repeat(self.opt.batch_size, 1, 1).unsqueeze(1)
                camera_fx = 1.0 / inputs[("K", source_scale)][:, 0:1, 0:1].repeat(1, self.opt.height, self.opt.width).unsqueeze(1) # fx
                camera_fy = 1.0 / inputs[("K", source_scale)][:, 1:2, 1:2].repeat(1, self.opt.height, self.opt.width).unsqueeze(1) # fy
                img_x_camera_fx = img_x * camera_fx
                img_y_camera_fy = img_y * camera_fy

                tan_phi = torch.sqrt(img_x_camera_fx*img_x_camera_fx + img_y_camera_fy*img_y_camera_fy)
                # tan_phi = torch.sqrt(img_x.pow(2).float()/camera_fx.pow(2)+img_y.pow(2).float()/camera_fy.pow(2))
                phi = torch.atan(tan_phi)
            else:
                ori_phi = np.pi * (outputs[("ori_phi", scale)] - 0.5) # [0, 1] -> [-pi/2.0, pi/2.0] instead of [0, 2*pi]
                phi = F.interpolate(
                    ori_phi, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            # compute depth
            depth = dist * torch.cos(phi)
            outputs[("depth", 0, scale)] = depth
            outputs[("dist", scale)] = dist
            outputs[("phi", 0, scale)] = phi
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                # if self.opt.use_polar and self.opt.rotation_constraint and scale == 0:
                if self.opt.rotation_constraint and (self.opt.with_multi_scale or scale == 0):
                    T_rot = outputs[("cam_T_cam_rot", 0, frame_id)]
                    pix_coords_rot = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T_rot)

                    # axisangle = outputs[("axisangle", 0, frame_id)]
                    # thres = torch.ones_like(axisangle)
                    # thres[:, :, :, 1] *= 0
                    # # 0: y
                    # # 1: x
                    # # 2: nothing
                    # thres[:, :, :, 2] *= 0
                    # axisangle = axisangle * thres
                    # T_test = self.rotation_transformation_from_parameters(
                    #             axisangle[:, 0].detach(), invert=(frame_id < 0))
                    # pix_coords_rot = self.project_3d[source_scale](
                    #     cam_points, inputs[("K", source_scale)], T_test)
                    
                    # stage1_polar_rot3_nyuv2, stage1_polar_rot_kitti3
                    tgt_img = inputs[("color", 0, source_scale)].view(-1, 3, self.opt.height*self.opt.width)
                    img_rot = torch.zeros_like(tgt_img) # b, 3, h*w
                    index_tgt_x = (pix_coords_rot[..., 0] / 2.0 + 0.5) * (self.opt.width - 1)
                    index_tgt_y = (pix_coords_rot[..., 1] / 2.0 + 0.5) * (self.opt.height - 1)
                    index_tgt = index_tgt_y.long() * self.opt.width + index_tgt_x.long()

                    index_tgt = index_tgt.view(-1, self.opt.height*self.opt.width).detach()
                    index_tgt = torch.clamp(index_tgt, 0, self.opt.height*self.opt.width-1)

                    img_rot[:, 0, :] = img_rot[:, 0, :].scatter(1, index_tgt, tgt_img[:, 0, :])
                    img_rot[:, 1, :] = img_rot[:, 1, :].scatter(1, index_tgt, tgt_img[:, 1, :])
                    img_rot[:, 2, :] = img_rot[:, 2, :].scatter(1, index_tgt, tgt_img[:, 2, :]) # b, 122880
                    img_rot = img_rot.view(-1, 3, self.opt.height, self.opt.width)
                    '''
                    img_rot = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_rot,
                            padding_mode="border") # stage1_polar_rot2_nyuv2, stage1_polar_rot_kitti2
                    '''
                    outputs[("img_rot", frame_id, scale)] = img_rot
                    outputs_rot = \
                        self.models["depth"](self.models["encoder"](img_rot))
                    outputs[("inv_dist_rot", frame_id, scale)] = F.grid_sample(
                        outputs_rot[("inv_dist", source_scale)],
                        pix_coords_rot,
                        padding_mode="border")

                    _, dist_rot = disp_to_depth(outputs[("inv_dist_rot", frame_id, scale)], self.opt.min_depth, self.opt.max_depth)
                    outputs[("dist_rot", frame_id, scale)] = dist_rot
                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    

    def compute_rotation_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        return l1_loss

    def extract_flow(self, pix_coords):
        new_pix_coords = pix_coords.clone()
        # [-1, 1] -> [0, 1] -> [0, w], [b, h, w, 2]
        new_pix_coords = new_pix_coords / 2.0 + 0.5
        new_pix_coords[:, :, :, 0] *= (new_pix_coords.shape[2]-1) # w
        new_pix_coords[:, :, :, 1] *= (new_pix_coords.shape[1]-1) # h

        xx, yy = np.meshgrid(np.arange(0, self.opt.width), np.arange(0, self.opt.height))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        cur_pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(self.opt.batch_size,1,1,1).float().to(self.device) # [b,2,h,w]
        cur_pix_coords = cur_pix_coords.permute(0, 2, 3, 1) # [b,h,w,2]

        flow_pred = new_pix_coords - cur_pix_coords

        return flow_pred

    def compute_flow_acc(self, dense_flow, sparse_flow):
        # sparse_flow: (2, N)
        index1_x = np.clip(sparse_flow[0], 0, self.opt.width-1).astype(int)
        index1_y = np.clip(sparse_flow[1], 0, self.opt.height-1).astype(int)
        index1 = index1_y * cur_width + index1_x
        index1 = torch.from_numpy(index1).cuda().squeeze().to(dtype=torch.int64)
                        

        index_x = 

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            if self.opt.use_polar:
                disp = outputs[("inv_dist", scale)]
            else:
                disp = outputs[("disp", scale)]

            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))
                        
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)
                outputs[("photometric_loss_map", scale)] = to_optimise

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
            
            to_optimise_mean = to_optimise.mean()
            loss += to_optimise_mean
            losses["loss/photo_loss_{}".format(scale)] = to_optimise_mean

            mean_disp = disp.mean(2, True).mean(3, True)
            # norm_disp = disp / (mean_disp + 1e-7)
            norm_disp = disp / (mean_disp + 1e-3)
            smooth_loss = get_smooth_loss(norm_disp, color)
            
            if torch.lt(mean_disp, 1e-6*torch.ones_like(mean_disp)).any():
                print(disp)
                input()

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            losses["loss/smooth_loss_{}".format(scale)] = smooth_loss / (2 ** scale)

            # new loss
            flag_compute_rotation_loss = self.opt.rotation_constraint and (self.opt.with_multi_scale or scale == 0)

            '''
            if flag_compute_rotation_loss:
                cur_width = self.opt.width // (2 ** scale)
                cur_height = self.opt.height // (2 ** scale)

                inv_dist = outputs[("inv_dist", scale)]
                inv_dist = F.interpolate(
                    inv_dist, [cur_height, cur_width], mode="bilinear", align_corners=False)

                rotation_loss = 0
                for frame_id in self.opt.frame_ids[1:]:
                    inv_dist_rot = outputs[("inv_dist_rot", frame_id, scale)]
                    inv_dist_rot = F.interpolate(
                        inv_dist_rot, [cur_height, cur_width], mode="bilinear", align_corners=False)

                    rotation_loss += torch.sum(self.tgt_mask[scale] * self.compute_rotation_loss(inv_dist_rot, inv_dist), (1,2,3)) / torch.sum(self.tgt_mask[scale], (1,2,3))                    
                    # cur_mask = outputs[("img_rot_mask", frame_id, scale)] * self.tgt_mask[scale]
                    # rotation_loss += (torch.sum(cur_mask * self.compute_rotation_loss(inv_dist_rot, inv_dist), (1,2,3)) / torch.sum(cur_mask, (1,2,3)))
                
                rotation_loss = (rotation_loss / len(self.opt.frame_ids[1:])).mean()
                loss += (rotation_loss * self.opt.rotation_loss_weights)
                losses["loss/rotation_loss_{}".format(scale)] = rotation_loss
            '''
            
            if flag_compute_rotation_loss:
                dist = outputs[("dist", scale)]
                dist = F.interpolate(
                    dist, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                rotation_loss = 0
                for frame_id in self.opt.frame_ids[1:]:
                    dist_rot = outputs[("dist_rot", frame_id, scale)]
                    dist_rot = F.interpolate(
                        dist_rot, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                    rotation_loss += torch.sum(self.tgt_mask[source_scale] * self.compute_rotation_loss(dist_rot, dist), (1,2,3)) * 1.0 / torch.sum(self.tgt_mask[source_scale], (1,2,3))
                rotation_loss = (rotation_loss / len(self.opt.frame_ids[1:])).mean()
                loss += rotation_loss * self.opt.rotation_loss_weights
                losses["loss/rotation_loss_{}".format(scale)] = rotation_loss
            
            if self.opt.use_lpips_loss:
                lpips_losses = []
                loss_fn_alex = lpips.LPIPS(net='alex', verbose=False).cuda()
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    lpips_losses.append(loss_fn_alex(pred, target))
                lpips_losses = torch.cat(lpips_losses, 1)
                lpips_loss = lpips_losses
            
                identity_lpips_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_lpips_losses.append(
                        loss_fn_alex(pred, target))
                        
                identity_lpips_losses = torch.cat(identity_lpips_losses, 1)
                identity_lpips_loss = identity_lpips_losses

                # add random numbers to break ties
                identity_lpips_loss += torch.randn(
                    identity_lpips_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_lpips_loss, lpips_loss), dim=1)
                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                lpips_loss = to_optimise.mean() * 1e-2
                outputs["lpips_loss/{}".format(scale)] = lpips_loss
                loss += self.opt.lpips_weight * lpips_loss

            if self.opt.use_geo_loss and scale == 0:
                cur_width = self.opt.width // (2 ** scale)
                cur_height = self.opt.height // (2 ** scale)

                geo_loss = 0.0 # 
                geo_loss_cnt = 0
                depth_tgt = outputs[("depth", 0, scale)]
                
                depth_tgt = F.interpolate(
                            depth_tgt, [cur_height, cur_width], mode="bilinear", align_corners=False)
                cam_points_tgt = self.backproject_depth[scale](
                    depth_tgt, inputs[("inv_K", scale)])
                    
                for frame_id in self.opt.frame_ids[1:]:
                    features_src = self.models["encoder"](inputs["color_aug", frame_id, 0])
                    outputs_src = self.models["depth"](features_src)
                    disp_src = outputs_src[("disp", scale)]
                    _, depth_src = disp_to_depth(disp_src, self.opt.min_depth, self.opt.max_depth)
                    cam_points_src = self.backproject_depth[scale](
                        depth_src, inputs[("inv_K", scale)])
                    
                    source = outputs[("color", frame_id, scale)]
                    for batch_idx in range(source.shape[0]):
                        img1 = source[batch_idx,:].permute(1,2,0).cpu().detach().numpy() * 255.0
                        img2 = target[batch_idx,:].permute(1,2,0).cpu().detach().numpy() * 255.0
                        pts1, pts2, img3 = find_correspondence_points(img1.astype(np.uint8), img2.astype(np.uint8)) # [2 n]
                        if pts1 is None:
                            continue
                        
                        optical_flow = self.extract_flow(outputs[("sample", frame_id, scale)])
                        optical_flow2 = pts2-pts1
                        acc = self.compute_flow_acc(optical_flow, optical_flow2)

                        if ("corr", frame_id, scale) not in outputs:
                            outputs[("corr", frame_id, scale)] = list()
                        img3 = torch.from_numpy(img3).cuda().permute(2,0,1)
                        outputs[("corr", frame_id, scale)].append(img3)

                        # compute related index
                        pts1 = np.array(pts1)
                        pts2 = np.array(pts2)
                        index1_x = np.clip(pts1[0], 0, cur_width-1).astype(int)
                        index1_y = np.clip(pts1[1], 0, cur_height-1).astype(int)
                        index1 = index1_y * cur_width + index1_x
                        index1 = torch.from_numpy(index1).cuda().squeeze().to(dtype=torch.int64)
                        index2_x = np.clip(pts2[0], 0, cur_width-1).astype(int)
                        index2_y = np.clip(pts2[1], 0, cur_height-1).astype(int)
                        index2 = index2_y * cur_width + index2_x
                        index2 = torch.from_numpy(index2).cuda().squeeze().to(dtype=torch.int64)
                        
                        # randomly select point-pairs for distance calculation
                        match_num = index1.nelement()
                        idx = torch.randperm(match_num)
                        index1_after = index1[idx]
                        index2_after = index2[idx]
                        
                        # compute distance
                        '''
                        cam_points_tgt_select = torch.cat([
                                    torch.gather(cam_points_tgt[batch_idx, 0], 0, index1).unsqueeze(0),
                                    torch.gather(cam_points_tgt[batch_idx, 1], 0, index1).unsqueeze(0),
                                    torch.gather(cam_points_tgt[batch_idx, 2], 0, index1).unsqueeze(0),
                                    torch.gather(cam_points_tgt[batch_idx, 3], 0, index1).unsqueeze(0)], dim=0)
                        cam_points_src_select = torch.cat([
                                    torch.gather(cam_points_src[batch_idx, 0], 0, index2).unsqueeze(0),
                                    torch.gather(cam_points_src[batch_idx, 1], 0, index2).unsqueeze(0),
                                    torch.gather(cam_points_src[batch_idx, 2], 0, index2).unsqueeze(0),
                                    torch.gather(cam_points_src[batch_idx, 3], 0, index2).unsqueeze(0)], dim=0)
                        cam_points_tgt_select = torch.cat([
                                    torch.gather(cam_points_tgt[batch_idx, 0], 0, index1_after).unsqueeze(1),
                                    torch.gather(cam_points_tgt[batch_idx, 1], 0, index1_after).unsqueeze(1),
                                    torch.gather(cam_points_tgt[batch_idx, 2], 0, index1_after).unsqueeze(1)], dim=1)
                        cam_points_src_select = torch.cat([
                                    torch.gather(cam_points_src[batch_idx, 0], 0, index2_after).unsqueeze(1),
                                    torch.gather(cam_points_src[batch_idx, 1], 0, index2_after).unsqueeze(1),
                                    torch.gather(cam_points_src[batch_idx, 2], 0, index2_after).unsqueeze(1)], dim=1)
                        '''
                        cam_points_tgt_select = cam_points_tgt[batch_idx, :, index1_after]
                        cam_points_src_select = cam_points_src[batch_idx, :, index2_after]
                        point_pair_num = match_num//2
                        tgt_distance = torch.sqrt(
                                            torch.sum(
                                                torch.pow(
                                                    torch.abs(
                                                        cam_points_tgt_select[:, :point_pair_num] - cam_points_tgt_select[:, point_pair_num:2*point_pair_num]
                                                    ), 
                                                    2
                                                ),
                                                dim=1
                                            )
                                        )
                        src_distance = torch.sqrt(
                                            torch.sum(
                                                torch.pow(
                                                    torch.abs(
                                                        cam_points_src_select[:, :point_pair_num] - cam_points_src_select[:, point_pair_num:2*point_pair_num]
                                                    ), 
                                                    2
                                                ),
                                                dim=1
                                            )
                                        )
                        # cur_geo_loss = torch.abs(cam_points_tgt_select-cam_points_src_select)
                        # cur_batch_geo_loss += torch.sum(cur_geo_loss) / cur_geo_loss.shape[-1]
                        geo_loss += torch.mean(torch.abs(tgt_distance-src_distance))
                        geo_loss_cnt += 1
                
                if geo_loss_cnt == 0:
                    print('geo loss cnt == 0')
                    print(smooth_loss)
                    # losses["loss/geo_loss_{}".format(scale)] = torch.tensor(0.0).cuda()
                else:
                    geo_loss = geo_loss / (geo_loss_cnt + 1e-6)
                    loss += geo_loss * self.opt.geo_loss_weights
                    losses["loss/geo_loss_{}".format(scale)] = geo_loss

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        '''
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        '''

        loss = losses["loss"].cpu().data
        photo_loss = losses["loss/photo_loss_0"].cpu().data
        smooth_loss = losses["loss/smooth_loss_0"].cpu().data
        
        if "loss/geo_loss_0" in losses:
            geo_loss = losses["loss/geo_loss_0"].cpu().data
        else:
            geo_loss = 0

        print_string = "epoch {:>3} | batch {:>6}" + \
            " | loss: {:.5f} | photo: {:.5f} | smooth: {:.5f} | geo: {:.5f}"
        print(print_string.format(self.epoch, batch_idx, loss, photo_loss, smooth_loss, geo_loss))

    def vis_depth(self, tensor, max_value=None):
        from matplotlib import cm
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap

        def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
            # Construct the list colormap, with interpolated values for higer resolution
            # For a linear segmented colormap, you can just specify the number of point in
            # cm.get_cmap(name, lutsize) with the parameter lutsize
            x = np.linspace(0, 1, low_res_cmap.N)
            low_res = low_res_cmap(x)
            new_x = np.linspace(0, max_value, resolution)
            high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                                for i in range(low_res.shape[1])], axis=1)
            return ListedColormap(high_res)

        tensor = tensor.detach().cpu()
        if max_value is None:
            max_value = tensor.max().item()
        
        colormap = high_res_colormap(cm.get_cmap('magma'))
        norm_array = tensor.squeeze().numpy()/max_value
        array = colormap(norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)
        return array 

    def log(self, mode, inputs, outputs, losses, add_image=False, vis_special=True):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        if self.opt.dataset == 'nyuv2_selected':
            min_images = self.opt.batch_size
        else:
            min_images = 2

        for j in range(min(min_images, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                        writer.add_image(
                            "photometric_loss_map_{}_{}/{}".format(frame_id, s, j),
                            outputs[("photometric_loss_map", s)][j].data.unsqueeze(0), self.step)
                        
                        if self.opt.rotation_constraint and (self.opt.with_multi_scale or s == 0):
                            writer.add_image(
                                "img_rot_{}/{}".format(s, j),
                                outputs[("img_rot", frame_id, s)][j], self.step)
                            writer.add_image(
                                "dist_rot_{}/{}".format(s, j),
                                normalize_image(outputs[("dist_rot", frame_id, s)][j]), self.step)
                            writer.add_image(
                                "total_mask_{}/{}".format(s, j),
                                outputs[("total_mask", frame_id, s)][j], self.step)

                        if self.opt.use_geo_loss and s == 0:
                            if ("corr", frame_id, s) in outputs:
                                writer.add_image(
                                    "corr_{}/{}".format(s, j),
                                    outputs[("corr", frame_id, s)][j], self.step)        

                if self.opt.use_polar:
                    writer.add_image(
                        "inv_dist_{}/{}".format(s, j),
                        normalize_image(outputs[("inv_dist", s)][j]), self.step)
                    writer.add_image(
                        "phi_{}/{}".format(s, j),
                        normalize_image(torch.cos(outputs[("phi", 0, s)][j])), self.step)
                    writer.add_image(
                        "depth_{}/{}".format(s, j),
                        normalize_image(outputs[("depth", 0, s)][j]), self.step)
                elif self.opt.use_z_theta:
                    writer.add_image(
                        "dist_{}/{}".format(s, j),
                        normalize_image(outputs[("dist", s)][j]), self.step)
                    writer.add_image(
                        "phi_{}/{}".format(s, j),
                        normalize_image(torch.cos(outputs[("phi", 0, s)][j])), self.step)
                    writer.add_image(
                        "depth_{}/{}".format(s, j),
                        normalize_image(outputs[("depth", 0, s)][j]), self.step)
                else:
                    '''
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)
                    '''
                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)

                # if self.opt.predictive_mask:
                #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                #         writer.add_image(
                #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                #             self.step)

                # elif not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

        if vis_special == True:
            def preprocess(input_image_dir):
                input_image = pil_loader(input_image_dir)
                resize = transforms.Resize((self.opt.height, self.opt.width),
                                                interpolation=Image.ANTIALIAS)
                input_image = resize(input_image)
                to_tensor = transforms.ToTensor()
                input_image = to_tensor(input_image).cuda()
                input_image = input_image.unsqueeze(0)
                return input_image
            
            def predict(input_image):
                features = self.models["encoder"](input_image)
                outputs_special = self.models["depth"](features)
                return outputs_special

            input_image_dir = 'dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00162.jpg'
            input_image = preprocess(input_image_dir)
            outputs_special = predict(input_image)
            
            if self.step == 0:
                writer.add_image(
                        "color_special_{}/{}".format(0, 0),
                        input_image[0], self.step)

            writer.add_image(
                        "disp_special_{}/{}".format(0, 0),
                        normalize_image(outputs_special[("disp", 0)][0]), self.step)
                        
            input_image_dir = 'dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00163.jpg'
            input_image = preprocess(input_image_dir)
            outputs_special = predict(input_image)

            if self.step == 0:
                writer.add_image(
                        "color_special_{}/{}".format(0, 1),
                        input_image[0], self.step)

            writer.add_image(
                        "disp_special_{}/{}".format(0, 1),
                        normalize_image(outputs_special[("disp", 0)][0]), self.step)

            input_image_dir = 'dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00191.jpg'
            input_image = preprocess(input_image_dir)
            outputs_special = predict(input_image)
            if self.step == 0:
                writer.add_image(
                        "color_special_{}/{}".format(0, 2),
                        input_image[0], self.step)

            writer.add_image(
                        "disp_special_{}/{}".format(0, 2),
                        normalize_image(outputs_special[("disp", 0)][0]), self.step)
            
            input_image_dir = 'dataset/NYUv2/NYUv2_labeled/processed_data/rgb/00192.jpg'
            input_image = preprocess(input_image_dir)
            outputs_special = predict(input_image)
            if self.step == 0:
                writer.add_image(
                        "color_special_{}/{}".format(0, 3),
                        input_image[0], self.step)
            writer.add_image(
                        "disp_special_{}/{}".format(0, 3),
                        normalize_image(outputs_special[("disp", 0)][0]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
