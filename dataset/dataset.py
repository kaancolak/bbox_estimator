''' Provider class and helper functions for Frustum PointNets.
Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import _pickle as pickle
import sys
import os
import numpy as np

from utils.box_util import box3d_iou
from torch.utils.data import Dataset

# Global Constants
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 8
NUM_OBJECT_POINT = 512

g_type2class = {
    'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3,
    'Person_sitting': 4, 'Cyclist': 5, 'Tram': 6, 'Misc': 7
}
g_class2type = {g_type2class[t]: t for t in g_type2class}

g_type_mean_size = {
    'car': np.array([4.6344314, 1.9600292, 1.7375569]),
    'truck': np.array([6.936331, 2.5178623, 2.8506238]),
    'bus': np.array([11.194943, 2.9501154, 3.4918275]),
    'trailer': np.array([12.275775, 2.9231303, 3.87086]),
    'Person_sitting': np.array([0.80057803, 0.5983815, 1.27450867]),
    'Cyclist': np.array([1.76282397, 0.59706367, 1.73698127]),
    'Tram': np.array([16.17150617, 2.53246914, 3.53079012]),
    'Misc': np.array([3.64300781, 1.54298177, 1.92320313])
}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))

for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]

def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


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


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, classes, num_points=512, min_points=10, train=True,
                 use_rotate=False,
                 use_mirror=False,
                 use_shift=False):
        self.data_dir = data_dir
        self.classes = classes
        self.num_points = num_points

        self.use_rotate = use_rotate
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

        bounding_box = self.bounding_boxes[idx]
        if self.labels[idx] in point_path:
            with open(point_path, 'rb') as f:
                obj_points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)

            nuscenes_to_model = np.array([[1.00000000e+00, -8.13701808e-16, 3.42213790e-16],
                                          [3.42861654e-16, 7.96326711e-04, -9.99999683e-01],
                                          [8.13429036e-16, 9.99999683e-01, 7.96326711e-04]]
                                         )

            # if not (bounding_box[6] > 2* np.pi/12 and bounding_box[6] < 5*np.pi/12):
            #     None


            obj_points = obj_points[:, :3]
            obj_points[:, :3] += bounding_box[:3]  # normalized to real position

            obj_points[:, :3] = np.dot(nuscenes_to_model, obj_points[:, :3].T).T
            obj_points[:, 1] += bounding_box[5] / 2

            center_to_corner_box3d_numpy = self.center_to_corner_box3d_numpy(bounding_box[0:3], bounding_box[3:6],
                                                                             bounding_box[6])

            rotated_corners = np.dot(nuscenes_to_model, center_to_corner_box3d_numpy[:, :3].T).T

            center = np.mean(rotated_corners, axis=0)
            x, y, z = center

            # Calculate yaw angle
            edge_direction = rotated_corners[1] - rotated_corners[0]  # Direction of one edge of the box
            yaw = np.arctan2(edge_direction[0], edge_direction[2])

            bounding_box[0], bounding_box[1], bounding_box[2] = x, y, z
            bounding_box[6] = yaw

            bounding_box[1] -= 1.65
            obj_points[:, 1] -= 1.65


        else:
            with open(point_path, 'rb') as f:
                obj_points = np.fromfile(f, dtype=np.float32).reshape(-1, 3)

            baselink_to_model = np.array([[7.96326711e-04, - 9.99999683e-01, 5.55111512e-16],
                                          [7.96326458e-04, 6.34136230e-07, - 9.99999683e-01],
                                          [9.99999366e-01, 7.96326458e-04, 7.96326711e-04]])

            obj_points = obj_points[:, :3]
            obj_points[:, :3] += bounding_box[:3]  # normalized to real position

            obj_points[:, :3] = np.dot(baselink_to_model, obj_points[:, :3].T).T
            obj_points[:, 1] += bounding_box[5] / 2

            center_to_corner_box3d_numpy = self.center_to_corner_box3d_numpy(bounding_box[0:3], bounding_box[3:6],
                                                                             bounding_box[6])

            rotated_corners = np.dot(baselink_to_model, center_to_corner_box3d_numpy[:, :3].T).T

            center = np.mean(rotated_corners, axis=0)
            x, y, z = center

            # Calculate yaw angle
            edge_direction = rotated_corners[1] - rotated_corners[0]  # Direction of one edge of the box
            yaw = np.arctan2(edge_direction[0], edge_direction[2])

            bounding_box[0], bounding_box[1], bounding_box[2] = x, y, z
            bounding_box[6] = yaw



        #Shuffle points order rondomly
        shuffled_indices = np.random.permutation(len(obj_points))
        obj_points = obj_points[shuffled_indices]

        original_point_count = len(obj_points)
        obj_points = self.pad_or_sample_points(obj_points, self.num_points)


        if self.use_rotate:
            obj_points, bounding_box = self.rotate_point_cloud(obj_points, bounding_box)

        if self.use_mirror:
            if np.random.random() > 0.5:  # 50% chance flipping
                obj_points[:, 0] *= -1
                bounding_box[0] *= -1
                bounding_box[6] = np.pi - bounding_box[6]

        if self.use_shift:
            shifter = np.random.uniform(0.95, 1.05)
            obj_points[:, 0] *= shifter  # Shift the point cloud
            obj_points[:, 2] *= shifter  # Shift the point cloud
            bounding_box[0] *= shifter  # Shift the center of the bounding box
            bounding_box[2] *= shifter  # Shift the center of the bounding box


        one_hot_vector = np.zeros(len(self.classes))
        ind = self.classes.index(self.labels[idx])
        one_hot_vector[ind] = 1.

        box_center = bounding_box[:3]
        box_size = bounding_box[3:6]

        size_class, residual = size2class(box_size, self.labels[idx])

        yaw = bounding_box[6]
        angle_class, angle_residual = angle2class(yaw, NUM_HEADING_BIN)

        seg = 0
        rot_angle = 0
        return obj_points, seg, box_center, angle_class, angle_residual, \
            size_class, residual, rot_angle, one_hot_vector

    def center_to_corner_box3d_numpy(self, centers, sizes, angles, origin=(0.5, 0.5, 0.5)):

        is_batched = True
        if len(centers.shape) == 1:
            centers = np.expand_dims(centers, axis=0)
            sizes = np.expand_dims(sizes, axis=0)
            angles = np.expand_dims(angles, axis=0)
            is_batched = False

        N = centers.shape[0]

        l, w, h = sizes[:, 0], sizes[:, 1], sizes[:, 2]

        # Compute the shift for each dimension based on the origin
        ox, oy, oz = origin
        x_shift = l * (0.5 - ox)
        y_shift = w * (0.5 - oy)
        z_shift = h * (0.5 - oz)

        # Create corner points in the local box coordinate system
        x_corners = np.stack([0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l],
                             axis=1) - np.expand_dims(x_shift, axis=1)
        y_corners = np.stack([0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w, 0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w],
                             axis=1) - np.expand_dims(y_shift, axis=1)
        z_corners = np.stack([0.5 * h, 0.5 * h, 0.5 * h, 0.5 * h, -0.5 * h, -0.5 * h, -0.5 * h, -0.5 * h],
                             axis=1) - np.expand_dims(z_shift, axis=1)

        corners = np.stack((x_corners, y_corners, z_corners), axis=-1)  # shape (N, 8, 3)

        # Rotation matrix around z-axis
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        rotation_matrix = np.stack([
            np.stack([cos_angles, -sin_angles, np.zeros_like(cos_angles)], axis=1),
            np.stack([sin_angles, cos_angles, np.zeros_like(cos_angles)], axis=1),
            np.stack([np.zeros_like(cos_angles), np.zeros_like(cos_angles), np.ones_like(cos_angles)], axis=1),
        ], axis=1)  # shape (N, 3, 3)

        # Apply rotation to each corner
        rotated_corners = np.einsum('bij,bkj->bki', rotation_matrix, corners)  # shape (N, 8, 3)

        # Translate corners to the center
        corners_3d = rotated_corners + np.expand_dims(centers, axis=1)  # shape (N, 8, 3)

        if not is_batched:
            corners_3d = np.squeeze(corners_3d, axis=0)

        return corners_3d

    def pad_or_sample_points(self, points, num_points):
        # Padding using duplicate of the correct points, maybe alternative padding with zeros
        if len(points) > num_points:
            sampled_indices = np.random.choice(len(points), num_points, replace=False)
            sampled_points = points[sampled_indices]
        elif len(points) < num_points:
            # pad_indices = np.random.choice(len(points), num_points - len(points), replace=True)
            # pad_points = points[pad_indices]
            # sampled_points = np.vstack((points, pad_points))

            # Copy points iteratively up to num points
            num_copies = num_points // len(points)
            remainder = num_points % len(points)
            sampled_points = np.tile(points, (num_copies, 1))
            sampled_points = np.vstack((sampled_points, points[:remainder]))

        else:
            sampled_points = points
        return sampled_points

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

    def rotate_point_cloud(self, pc, label, min_angle=-np.pi / 2, max_angle=np.pi / 2):
        angle = np.random.uniform(min_angle, max_angle)

        cosval = np.cos(angle)
        sinval = np.sin(angle)
        rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
        pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))

        expanded_label = np.expand_dims(label, 0)
        expanded_label[:, [0, 2]] = np.dot(expanded_label[:, [0, 2]], np.transpose(rotmat))

        unsqu_label = expanded_label.squeeze()
        unsqu_label[6] = unsqu_label[6] - angle

        return pc, unsqu_label

    def corners_to_bounding_box(self, corners):
        """
        Convert 8 corners of a 3D bounding box to (x, y, z, width, length, height, yaw).

        :param corners: List or array of shape (8, 3) containing corner points of the bounding box.
        :return: List [x, y, z, width, length, height, yaw] representing the bounding box parameters.
        """
        # Compute center (x, y, z)
        center = np.mean(corners, axis=0)
        x, y, z = center

        # Determine dimensions (width, length, height)
        width = np.max(corners[:, 0]) - np.min(corners[:, 0])
        length = np.max(corners[:, 1]) - np.min(corners[:, 1])
        height = np.max(corners[:, 2]) - np.min(corners[:, 2])

        # Calculate yaw angle
        edge_direction = corners[1] - corners[0]  # Direction of one edge of the box
        yaw = np.arctan2(edge_direction[1], edge_direction[0])

        return [x, y, z, width, length, height, yaw]


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.
    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.
    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
                                          heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
                                      heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
        np.array(iou3d_list, dtype=np.float32)


def box3d_from_prediction(center_pred,
                          heading_logits, heading_residuals,
                          size_logits, size_residuals):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.
    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    boxes = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        box = {'center': center_pred[i],
               'heading_angle': heading_angle,
               'box_size': box_size}
        boxes.append(box)
    return boxes


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset(1024, split='val',
                             rotate_to_center=True, random_flip=True, random_shift=True)
    for i in range(len(dataset)):
        data = dataset[i]
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6], \
               'real_size:', g_type_mean_size[g_class2type[data[5]]] + data[6]))
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))

        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])
        ps = data[0]
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))
