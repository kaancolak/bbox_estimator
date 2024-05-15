import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


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
        self.labels =  []
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
        obj_points[:, :3] += self.bounding_boxes[idx][:3] # normalized to real position

        bounding_box = self.bounding_boxes[idx]
        one_hot_vector = torch.zeros(len(self.classes))
        ind = self.classes.index(self.labels[idx])
        one_hot_vector[ind] = 1.
        print(one_hot_vector)

        return torch.tensor(obj_points, dtype=torch.float32), one_hot_vector, torch.tensor(bounding_box, dtype=torch.float32)

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

#
# datadir = "/media/kaan/Extreme SSD/nuscenes/"
# classes = ['car', 'truck', 'bus', 'trailer']
# # classes = ['car', 'pedestrian', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle']
# dataset = PointCloudDataset(datadir, classes)
#
# print(len(dataset))
# print(dataset)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# for i, (points, labels) in enumerate(dataloader):
#     print(points.shape, labels.shape)
#     break
