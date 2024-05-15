import torch

print(torch.__version__)

import pickle
import numpy as np

from vis_utils import center_to_corner_box3d

dataset_path = "/media/kaan/Extreme SSD/nuscenes/"
db_info_path = dataset_path + "nuscenes_dbinfos_train.pkl"
with open(db_info_path, 'rb') as f:
    data = pickle.load(f)

print(data.keys())
# dict_keys(['traffic_cone', 'truck', 'car', 'pedestrian', 'barrier', 'construction_vehicle', 'motorcycle', 'bicycle', 'bus', 'trailer'])

print(len(data['truck']))  # 65262
print(len(data['car']))  # 339949
print(len(data['pedestrian']))  # 161928
print(len(data['motorcycle']))  # 8846
print(len(data['bicycle']))  # 8185
print(len(data['bus']))  # 12286
print(len(data['trailer']))  # 19202

test_idx = 5550

print(data['car'][test_idx])
# {'name': 'car', 'path': 'nuscenes_gt_database/0_car_3.bin',
#  'image_idx': 0, 'gt_idx': 3,
#  'box3d_lidar': array([9.21442, -5.5796075, -1.7985696, 4.25, 1.638, 1.44, 0.3584961, 5.52849, 1.1955143],
#                       dtype=float32),
#  'num_points_in_gt': 1672, 'difficulty': 0, 'group_id': 3}

filtered_car = [d for d in data['car'] if d['num_points_in_gt'] > 512]
print("Len filtered cars:" + str(len(filtered_car)))  # 339949

filtered_truck = [d for d in data['truck'] if d['num_points_in_gt'] > 512]
print("Len filtered truck:" + str(len(filtered_truck)))  # 339949

filtered_bus = [d for d in data['bus'] if d['num_points_in_gt'] > 512]
print("Len filtered bus:" + str(len(filtered_bus)))  # 339949


obj_point_path = dataset_path + data['car'][test_idx]['path']
print(obj_point_path)
with open(obj_point_path, 'rb') as f:
    obj_points = np.fromfile(f, dtype=np.float32).reshape(-1, 5)

print(obj_points.shape)

obj_points[:, :3] += data['car'][test_idx]['box3d_lidar'][:3]

bbox_center = data['car'][test_idx]['box3d_lidar'][:3]
bbox_size = data['car'][test_idx]['box3d_lidar'][3:6]
bbox_yaw = data['car'][test_idx]['box3d_lidar'][6]

import matplotlib.pyplot as plt

x = obj_points[:, 0]
y = obj_points[:, 1]
z = obj_points[:, 2]
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)


def draw_box(ax, c, s, yaw):
    as_np = np.array([c.tolist() + s.tolist() + [yaw]]).astype(np.float32)
    corners = center_to_corner_box3d(
        as_np[:, :3], as_np[:, 3:6], as_np[:, 6], origin=(0.5, 0.5, 0), axis=2)
    # Convert tensor to numpy
    corners = corners.numpy()
    corners = corners.squeeze(0)
    print(corners)

    # Verticals
    ax.plot3D(*zip(corners[0], corners[1]), color='b')
    ax.plot3D(*zip(corners[2], corners[3]), color='b')
    ax.plot3D(*zip(corners[4], corners[5]), color='b')
    ax.plot3D(*zip(corners[6], corners[7]), color='b')
    ax.plot3D(*zip(corners[0], corners[3]), color='b')
    ax.plot3D(*zip(corners[4], corners[7]), color='b')
    ax.plot3D(*zip(corners[1], corners[2]), color='b')
    ax.plot3D(*zip(corners[5], corners[6]), color='b')
    ax.plot3D(*zip(corners[0], corners[4]), color='b')
    ax.plot3D(*zip(corners[1], corners[5]), color='b')
    ax.plot3D(*zip(corners[2], corners[6]), color='b')
    ax.plot3D(*zip(corners[3], corners[7]), color='b')


draw_box(ax, bbox_center, bbox_size, bbox_yaw)

plt.show()
