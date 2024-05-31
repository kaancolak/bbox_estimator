import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.pointcloud_dataset import PointCloudDataset, NUM_HEADING_BIN, g_class2type, g_type_mean_size
from utils.box_utils import center_to_corner_box3d_numpy, class2angle, class2size

import pickle
import random


def draw_box(ax, center, size, yaw):
    """Draws a 3D bounding box on the provided axes."""
    bbox_params = np.array([center.tolist() + size.tolist() + [yaw]], dtype=np.float32)
    corners = center_to_corner_box3d_numpy(bbox_params[:, :3], bbox_params[:, 3:6], bbox_params[:, 6]).squeeze(0)

    # Plot the edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]
    for start, end in edges:
        ax.plot3D(*zip(corners[start], corners[end]), color='b')


def class2size_test(pred_cls, residual):
    """Inverse function to convert class and residual to actual size."""
    type_str = g_class2type[pred_cls.item()]
    mean_size = g_type_mean_size[type_str]
    return mean_size + residual.numpy()


def visualize_point_cloud_with_bbox(data, ax):
    """Visualizes the point cloud and bounding box for a single data point."""
    obj_points = data['point_cloud']
    x, y, z = obj_points[:, 0], obj_points[:, 1], obj_points[:, 2]
    ax.scatter(x, y, z)
    heading_angle = class2angle(data['angle_class'], data['angle_residual'], NUM_HEADING_BIN)
    box_size = class2size_test(data['size_class'], data['size_residual'])
    draw_box(ax, data['box3d_center'], box_size, heading_angle)

def visualize_point_cloud_with_bbox_center_head(data, ax):
    """Visualizes the point cloud and bounding box for a single data point."""
    obj_points = data['point_cloud']
    x, y, z = obj_points[:, 0], obj_points[:, 1], obj_points[:, 2]
    ax.scatter(x, y, z)

    sin_yaw_gt = data['sin_yaw']
    cos_yaw_gt = data['cos_yaw']
    yaw_angle_gt = np.arctan2(sin_yaw_gt, cos_yaw_gt)
    yaw_angle_wrapped_gt = np.arctan2(np.sin(yaw_angle_gt), np.cos(yaw_angle_gt))

    box_size = data['box_size']
    draw_box(ax, data['box3d_center'], box_size, yaw_angle_wrapped_gt)

def main():
    # Dataset and DataLoader initialization
    datadir = "/home/kaan/dataset_custom/"
    classes = ['car', 'truck', 'bus', 'trailer']
    dataset = PointCloudDataset(datadir, classes, min_points=10, train=True, augment_data=False)

    # Visualize a specific data point
    data_point = dataset[50]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    visualize_point_cloud_with_bbox_center_head(data_point, ax)
    # visualize_point_cloud_with_bbox(data_point, ax)

    plt.show()

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


def merge_dataset(path1, path2):
    train_orig_db_path = path1 + "dbinfos_train.pkl"
    with open(train_orig_db_path, 'rb') as f:
        db_infos_train_orig = pickle.load(f)

    test_orig_db_path = path1 + "dbinfos_val.pkl"
    with open(test_orig_db_path, 'rb') as f:
        db_infos_test_orig = pickle.load(f)

    train2_db_path = path2 + "dbinfos_train.pkl"
    with open(train2_db_path, 'rb') as f:
        db_infos2 = pickle.load(f)

    print(len(db_infos_train_orig['car']))
    print(len(db_infos_test_orig['car']))
    print(len(db_infos2['car']))

    print(db_infos2['car'][0])
    print(db_infos_train_orig['car'][0])

    for key in db_infos2:
        for i in range(len(db_infos2[key])):
            # Change all paths to the new path
            # gt_database/1704951684_825038000_1.bin to nuscenes_gt_database/0_car_3.bin
            db_infos2[key][i]['path'] = db_infos2[key][i]['path'].replace("gt_database", "nuscenes_gt_database")

    for key in db_infos2:
        shuffled_list = db_infos2[key]
        random.shuffle(shuffled_list)

        db_infos_train_orig[key].extend(shuffled_list[:int(len(shuffled_list) * 0.8)])
        db_infos_test_orig[key].extend(shuffled_list[int(len(shuffled_list) * 0.8):])

    print("After merge")
    print(len(db_infos_train_orig['car']))
    print(len(db_infos_test_orig['car']))

    # Save the new db_infos
    # with open(train_orig_db_path, 'wb') as f:
    #     pickle.dump(db_infos_train_orig, f)
    #
    # with open(test_orig_db_path, 'wb') as f:
    #     pickle.dump(db_infos_test_orig, f)


    with open("/home/kaan/datas/dbinfos_train.pkl", 'rb') as f:
        asdasd = pickle.load(f)
    print(len(asdasd['car']))

    # import os
    # try:
    #     os.mkdir(path)
    # except OSError as error:
    #     print(error)


if __name__ == "__main__":
    main()
    # merge_dataset("/home/kaan/dataset_custom/",
    #               "/home/kaan/dataset_hiratsuka/")
