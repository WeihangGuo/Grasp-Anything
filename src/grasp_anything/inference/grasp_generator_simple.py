import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera, RGBCamera, AbstractCamera
from hardware.device import get_device
from grasp_anything.inference.post_process import post_process_output
from grasp_anything.utils.data.camera_data import CameraData
from grasp_anything.utils.dataset_processing.grasp import detect_grasps
from grasp_anything.utils.visualisation.plot import plot_grasp

# This class is a simplified version of the GraspGenerator class. It does not translate the grasp pose to the robot. 
class GraspGeneratorSimple:
    def __init__(self, saved_model_path, camera:AbstractCamera, visualize=False):
        self.saved_model_path = saved_model_path
        self.camera = camera
        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None
        if isinstance(camera, RGBCamera):
            self.is_depth = False
        else:
            self.is_depth = True
        self.cam_data = CameraData(include_depth=self.is_depth, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        # self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        # self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')

        # homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        # self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        # self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        # self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        # Very hacky. Need replacement
        # It is needed because the path is changed from relative to absolute
        import sys
        import os
        this_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(this_dir)
        sys.path.append(parent_dir)
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path, weights_only=False)
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)

        # Get grasp position from model output
        if self.is_depth:
            pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] * self.cam_depth_scale - 0.04
        else: 
            pos_z = -1 # No depth information available
        pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
        pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)

        if pos_z == 0:
            return

        target = np.asarray([pos_x, pos_y, pos_z])
        target.shape = (3, 1)
        print('target: ', target)

        # Convert camera to robot coordinates
        # camera2robot = self.cam_pose
        # target_position = np.dot(camera2robot[0:3, 0:3], target) + camera2robot[0:3, 3:]
        # target_position = target_position[0:3, 0]

        # Convert camera to robot angle
        # angle = np.asarray([0, 0, grasps[0].angle])
        # angle.shape = (3, 1)
        # target_angle = np.dot(camera2robot[0:3, 0:3], angle)

        # Concatenate grasp pose with grasp angle
        # grasp_pose = np.append(target_position, target_angle[2])

        # print('grasp_pose: ', grasp_pose)

        # np.save(self.grasp_pose, grasp_pose)

        if self.fig:
            plot_grasp(fig=self.fig, rgb_img=self.cam_data.get_rgb(rgb, False), grasps=grasps, save=True)

    def run(self):
        while True:
            self.generate()
