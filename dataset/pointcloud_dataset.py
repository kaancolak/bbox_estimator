import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

from models.provider import *

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8  # one cluster for each type
NUM_OBJECT_POINT = 512


class PointCloudDataset(Dataset):
    def __init__(self, data_dir, classes, num_points=512, min_points=10):
        self.data_dir = data_dir
        self.classes = classes
        self.num_points = num_points

        db_info = self.data_dir + "nuscenes_dbinfos_train.pkl"
        with open(db_info, 'rb') as f:
            data = pickle.load(f)

        self.files = []
        self.bounding_boxes = []
        self.labels = []
        for c in classes:
            self.files += [d['path'] for d in data[c] if d['num_points_in_gt'] > min_points]
            self.bounding_boxes += [d['box3d_lidar'] for d in data[c] if d['num_points_in_gt'] > min_points]
            self.labels += [d['name'] for d in data[c] if d['num_points_in_gt'] > min_points]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        point_path = self.data_dir + self.files[idx]
        with open(point_path, 'rb') as f:
            obj_points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)
        obj_points = self.pad_or_sample_points(obj_points, self.num_points)
        obj_points[:, :3] += self.bounding_boxes[idx][:3]  # normalized to real position

        bounding_box = self.bounding_boxes[idx]

        one_hot_vector = torch.zeros(len(self.classes))
        ind = self.classes.index(self.labels[idx])
        one_hot_vector[ind] = 1.
        box_center = bounding_box[:3]
        box_size = bounding_box[3:6]
        size_class, residual = size2class(box_size, "car")
        yaw = bounding_box[6]
        angle_class, angle_residual = angle2class(yaw,
                                                  NUM_HEADING_BIN)
        data = {
            'point_cloud': torch.tensor(obj_points, dtype=torch.float32),
            'one_hot': one_hot_vector,
            # 'box3d_center': torch.tensor(box_center, dtype=torch.float32),
            'box3d_center': box_center,
            'size_class': size_class,
            # 'size_residual': torch.tensor(residual, dtype=torch.float32),
            'size_residual': residual,
            'angle_class': torch.tensor(angle_class, dtype=torch.float32),
            'angle_residual': torch.tensor(angle_residual, dtype=torch.float32),
        }

        return data

    def pad_or_sample_points(self, points, num_points):
        # Padding using duplicate of the correct points, maybe alternative padding with zeros
        if len(points) > num_points:
            sampled_indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[sampled_indices]
        elif len(points) < num_points:
            pad_indices = np.random.choice(len(points), num_points - len(points), replace=True)
            pad_points = points[pad_indices]
            sampled_points = np.vstack((points, pad_points))
        else:
            sampled_points = points
        return sampled_points

    def size2class(bbox, type_name):
        ''' Convert 3D bounding box size to template class and residual.
        todo (rqi): support multiple size clusters per type.

        Input:
            size: numpy array of shape (3,) for (l,w,h)
            type_name: string
        Output:
            size_class: int scalar
            size_residual: numpy array of shape (3,)
        '''
        size_class = g_type2class[type_name]
        size_residual = bbox - g_type_mean_size[type_name]
        return size_class, size_residual

    def angle2class(angle, num_class):
        ''' Convert continuous angle to discrete class and residual.

        Input:
            angle: rad scalar, from 0-2pi (or -pi~pi), class center at
                0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            num_class: int scalar, number of classes N
        Output:
            class_id, int, among 0,1,...,N-1
            residual_angle: float, a number such that
                class*(2pi/N) + residual_angle = angle
        '''
        angle = angle % (2 * np.pi)
        assert (angle >= 0 and angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - \
                         (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

#
# #
# datadir = "/media/kaan/Extreme SSD/nuscenes/"
# # classes = ['car', 'truck', 'bus', 'trailer']
# # classes = ['car', 'pedestrian', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle']
# classes = ['car']
# dataset = PointCloudDataset(datadir, classes)
# from timeit import default_timer as timer
#
# start = timer()
# d = dataset.__getitem__(100)
# end = timer()
# print(end - start)
# start = timer()
# d = dataset.__getitem__(6)
# end = timer()
# print(end - start)
# start = timer()
# d = dataset.__getitem__(0)
# end = timer()
# print(end - start)
#
# print(dataset.__getitem__(100).keys())
#
# heading_angle = class2angle(d['angle_class'], d['angle_residual'], NUM_HEADING_BIN)
# box_size = class2size(d['size_class'], d['size_residual'])
#
# # corners_3d = get_3d_box(box_size, heading_angle, d['box3d_center'])
#
# from models.box_utils import center_to_corner_box3d, center_to_corner_box3d_torch, center_to_corner_box3d_numpy
#
# # print("np version: ")
# # print(d['box3d_center'])
# # print(box_size)
# # print(heading_angle)
# corner3d = center_to_corner_box3d_numpy(d['box3d_center'], box_size, heading_angle)
#
# print(corner3d)
# from models.box_utils import box3d_iou
#
# print("************")
# print("************")
# print("************")
# print("************")
# print(box3d_iou(corner3d, corner3d))
# # center = torch.from_numpy(d['box3d_center'])
# # center = torch.unsqueeze(center, 0)
# # size = torch.from_numpy(box_size)
# # size = torch.unsqueeze(size, 0)
# # # print(type(heading_angle))
# # # heading = torch.from_numpy(heading_angle)
# #
# centers = torch.tensor([
#     [4.441864,  11.292346,  -1.5824497],  # Center of the first box
# ], dtype=torch.float32)
#
# sizes = torch.tensor([
#     [4.66099977, 1.85500002, 1.76300001],  # Size of the first box (length, width, height)
# ], dtype=torch.float32)
#
# angles = torch.tensor([
#     -0.9111         # Rotation angle of the third box (in radians)
# ], dtype=torch.float32)


#
#
# # # torch.Size([3, 3])
# # # torch.Size([3, 3])
# # # torch.Size([3])
# #
# # print(centers.shape)
# # print(sizes.shape)
# # print(angles.shape)
# corner3d_torch = center_to_corner_box3d_torch(centers, sizes, angles)
# print(corner3d_torch)

# from visualization.vis_utils import center_to_corner_box3d
# from visualization.test import  draw_box
#
# import matplotlib.pyplot as plt
# obj_points = d['point_cloud']
# x = obj_points[:, 0]
# y = obj_points[:, 1]
# z = obj_points[:, 2]
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# draw_box(ax, d['box3d_center'], box_size, heading_angle)
# plt.show()
#
#
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
