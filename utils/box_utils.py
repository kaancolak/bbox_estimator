import numpy as np
from scipy.spatial import ConvexHull
import torch

from dataset.pointcloud_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from dataset.pointcloud_dataset import g_class2type, g_type_mean_size

def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU


    todo (rqi): add more description on corner points' orders.
    '''

    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])

    # inter, inter_area = convex_hull_intersection(rect1, rect2)
    # iou_2d = inter_area / (area1 + area2 - inter_area)
    int_area = intersection_area(rect1, rect2)
    iou_2d = int_area / (area1 + area2 - int_area)

    ymax = min(corners1[0, 2], corners2[0, 2])
    ymin = max(corners1[4, 2], corners2[4, 2])
    inter_vol = int_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)

    iou_2d = max(iou_2d, 0.0)
    iou_2d = min(iou_2d, 1.0)
    iou = max(iou, 0.0)
    iou = min(iou, 1.0)

    return iou, iou_2d


def center_to_corner_box3d_torch(centers, sizes, angles, origin=(0.5, 0.5, 1.)):
    """
    Convert center-based 3D box parameters to corner coordinates.

    Args:
        centers (torch.Tensor): Tensor of shape (N, 3) representing the centers of the boxes.
        sizes (torch.Tensor): Tensor of shape (N, 3) representing the sizes of the boxes (length, width, height).
        angles (torch.Tensor): Tensor of shape (N,) representing the rotation angles of the boxes around the z-axis.
        origin (tuple): Origin point for the boxes in the form of (ox, oy, oz) where each value is between 0 and 1.

    Returns:
        torch.Tensor: Tensor of shape (N, 8, 3) representing the corner coordinates of the boxes.
    """
    N = centers.shape[0]

    l, w, h = sizes[:, 0], sizes[:, 1], sizes[:, 2]

    # Compute the shift for each dimension based on the origin
    ox, oy, oz = origin
    x_shift = l * (0.5 - ox)
    y_shift = w * (0.5 - oy)
    z_shift = h * (0.5 - oz)

    # Create corner points in the local box coordinate system
    x_corners = torch.stack([0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l],
                            dim=1) - x_shift.unsqueeze(1)
    y_corners = torch.stack([0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w, 0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w],
                            dim=1) - y_shift.unsqueeze(1)
    z_corners = torch.stack([0.5 * h, 0.5 * h, 0.5 * h, 0.5 * h, -0.5 * h, -0.5 * h, -0.5 * h, -0.5 * h],
                            dim=1) - z_shift.unsqueeze(1)

    corners = torch.stack((x_corners, y_corners, z_corners), dim=-1)  # shape (N, 8, 3)

    # Rotation matrix around z-axis
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)

    rotation_matrix = torch.stack([
        torch.stack([cos_angles, -sin_angles, torch.zeros_like(cos_angles)], dim=1),
        torch.stack([sin_angles, cos_angles, torch.zeros_like(cos_angles)], dim=1),
        torch.stack([torch.zeros_like(cos_angles), torch.zeros_like(cos_angles), torch.ones_like(cos_angles)], dim=1),
    ], dim=1)  # shape (N, 3, 3)

    # Apply rotation to each corner
    rotated_corners = torch.einsum('bij,bkj->bki', rotation_matrix, corners)  # shape (N, 8, 3)

    # Translate corners to the center
    corners_3d = rotated_corners + centers.unsqueeze(1)  # shape (N, 8, 3)

    return corners_3d


def center_to_corner_box3d_numpy(centers, sizes, angles, origin=(0.5, 0.5, 1.)):
    """
    Convert center-based 3D box parameters to corner coordinates.

    Args:
        centers (np.ndarray): Array of shape (N, 3) or (3,) representing the centers of the boxes.
        sizes (np.ndarray): Array of shape (N, 3) or (3,) representing the sizes of the boxes (length, width, height).
        angles (np.ndarray): Array of shape (N,) or () representing the rotation angles of the boxes around the z-axis.
        origin (tuple): Origin point for the boxes in the form of (ox, oy, oz) where each value is between 0 and 1.

    Returns:
        np.ndarray: Array of shape (N, 8, 3) or (8, 3) representing the corner coordinates of the boxes.
    """
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


def shoelace_formula(x, y):
    """Compute the area of a polygon using the shoelace formula."""
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j]
        area -= x[j] * y[i]
    return abs(area) / 2.0


def intersection_area(corners1, corners2):
    """Compute the intersection area of two polygons."""
    x1, y1 = zip(*corners1)
    x2, y2 = zip(*corners2)

    # Compute area of each polygon
    area1 = shoelace_formula(x1, y1)
    area2 = shoelace_formula(x2, y2)

    # Find the intersection points
    intersection_x = x1 + x2
    intersection_y = y1 + y2

    # Compute area of intersection polygon
    intersection_area = shoelace_formula(intersection_x, intersection_y)

    return min(area1, area2, intersection_area)


def compute_box3d_iou(center_pred,
                      pred,
                      gt):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residual: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residual: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''



    sin_yaw_pred = pred['sin_yaw'].detach().cpu().numpy()
    cos_yaw_pred = pred['cos_yaw'].detach().cpu().numpy()
    size_pred = pred['sizes'].detach().cpu().numpy()
    center_pred = center_pred.detach().cpu().numpy()

    sin_yaw_gt = gt['sin_yaw'].detach().cpu().numpy()
    cos_yaw_gt = gt['cos_yaw'].detach().cpu().numpy()
    size_gt = gt['box_size'].detach().cpu().numpy()
    center_gt = gt['box3d_center'].detach().cpu().numpy()



    batch_size = sin_yaw_gt.shape[0]

    iou2d_list = []
    iou3d_list = []

    for i in range(batch_size):

        yaw_angle_gt = np.arctan2(sin_yaw_gt[i], cos_yaw_gt[i])
        yaw_angle_wrapped_gt = np.arctan2(np.sin(yaw_angle_gt), np.cos(yaw_angle_gt))


        corners_3d = center_to_corner_box3d_numpy(center_gt[i], size_gt[i], yaw_angle_wrapped_gt )


        yaw_angle_pred = np.arctan2(sin_yaw_pred[i], cos_yaw_pred[i])
        yaw_angle_wrapped_pred = np.arctan2(np.sin(yaw_angle_pred), np.cos(yaw_angle_pred))

        corners_3d_pred = center_to_corner_box3d_numpy(center_pred[i], size_pred[i], np.squeeze(yaw_angle_wrapped_pred))

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_pred)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)


    return np.array(iou2d_list, dtype=np.float32), np.array(iou3d_list, dtype=np.float32)


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
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    type_str = g_class2type[pred_cls]
    mean_size = g_type_mean_size[type_str]

    return mean_size + residual
