import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader


# -----------------
# Global Constants
# -----------------

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 4  # 8 # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class = {'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3}

g_type_mean_size = {'car': np.array([4.6344314, 1.9600292, 1.7375569]),
                    'truck': np.array([6.936331, 2.5178623, 2.8506238]),
                    'bus': np.array([11.194943, 2.9501154, 3.4918275]),
                    'trailer': np.array([12.275775, 2.9231303, 3.87086])}

g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


class PointCloudDataset(Dataset):
    def __init__(self, data_dir, classes, num_points=512, min_points=10, train=True,
                 augment_data=False,
                 use_mirror=False,
                 use_shift=False):
        self.data_dir = data_dir
        self.classes = classes
        self.num_points = num_points

        self.augment_data = augment_data
        self.use_mirror = use_mirror
        self.use_shift = use_shift

        if train:
            db_info = self.data_dir + "dbinfos_train.pkl"
        else:
            db_info = self.data_dir + "dbinfos_val.pkl"

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

        # print(obj_points.shape)
        obj_points = self.pad_or_sample_points(obj_points, self.num_points)
        obj_points[:, :3] += self.bounding_boxes[idx][:3]  # normalized to real position

        bounding_box = self.bounding_boxes[idx]

        if self.augment_data:

            if self.use_mirror:

                rand_int = np.random.randint(0, 3)

                if rand_int == 0:
                    obj_points, bounding_box = self.mirror_point_cloud(obj_points, bounding_box, 'yz')
                elif rand_int == 1:
                    obj_points, bounding_box = self.mirror_point_cloud(obj_points, bounding_box, 'xz')

            if self.use_shift:
                obj_points, bounding_box = self.random_shift_point_cloud(obj_points, bounding_box, [20, 5])

        one_hot_vector = torch.zeros(len(self.classes))
        ind = self.classes.index(self.labels[idx])
        one_hot_vector[ind] = 1.
        box_center = bounding_box[:3]
        box_size = bounding_box[3:6]

        size_class, residual = self.size2class(box_size, self.labels[idx])
        yaw = bounding_box[6]
        angle_class, angle_residual = self.angle2class(yaw,
                                                  NUM_HEADING_BIN)
        data = {
            'point_cloud': torch.tensor(obj_points, dtype=torch.float32),
            'one_hot': one_hot_vector,
            'box3d_center': torch.tensor(box_center, dtype=torch.float32),
            'size_class': torch.tensor(size_class, dtype=torch.float32),
            'size_residual': torch.tensor(residual, dtype=torch.float32),
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

    def size2class(self, bbox, type_name):
        ''' Convert 3D bounding box size to template class and residual.
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

    def angle2class(self, angle, num_class):
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

    def mirror_point_cloud(self, point_cloud, label, plane):
        """
        Mirror the point cloud and label along the specified plane.
        """
        mirrored_point_cloud = np.copy(point_cloud)
        mirrored_label = np.copy(label)

        if plane == 'xz':
            mirrored_point_cloud[:, 1] = -mirrored_point_cloud[:, 1]
            mirrored_label[1] = -mirrored_label[1]  # Mirror y-coordinate of box center
        elif plane == 'yz':
            mirrored_point_cloud[:, 0] = -mirrored_point_cloud[:, 0]
            mirrored_label[0] = -mirrored_label[0]  # Mirror x-coordinate of box center

        # Adjust the rotation for the mirror
        mirrored_label[6] = -mirrored_label[6]

        return mirrored_point_cloud, mirrored_label

    def random_shift_point_cloud(self, point_cloud, label, shift_range):
        """
        Apply random shift to the point cloud and label within the specified range.
        """
        shift_x = np.random.uniform(-shift_range[0], shift_range[0])
        shift_y = np.random.uniform(-shift_range[1], shift_range[1])

        shifted_point_cloud = point_cloud + np.array([shift_x, shift_y, 0, 0, 0])
        shifted_label = label.copy()
        shifted_label[:2] += np.array([shift_x, shift_y])  # Shift the center of the bounding box

        return shifted_point_cloud, shifted_label
