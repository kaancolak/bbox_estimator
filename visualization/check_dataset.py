import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.pointcloud_dataset import PointCloudDataset, NUM_HEADING_BIN, g_class2type, g_type_mean_size
from utils.box_utils import center_to_corner_box3d_numpy, class2angle, class2size


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


def main():
    # Dataset and DataLoader initialization
    datadir = "/home/kaan/datas/"
    classes = ['car', 'truck', 'bus', 'trailer']
    dataset = PointCloudDataset(datadir, classes, min_points=4000, train=True, augment_data=True, use_mirror=True,
                                use_shift=True)

    # Visualize a specific data point
    data_point = dataset[120]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    visualize_point_cloud_with_bbox(data_point, ax)
    plt.show()

    # DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


if __name__ == "__main__":
    main()