""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected by Charles R. Qi
Date: September 2017
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull
import torch


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def is_clockwise(p):
    x = p[:, 0]
    y = p[:, 1]
    return np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)) > 0


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


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two 2D bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def box2d_iou(box1, box2):
    ''' Compute 2D bounding box IoU.

    Input:
        box1: tuple of (xmin,ymin,xmax,ymax)
        box2: tuple of (xmin,ymin,xmax,ymax)
    Output:
        iou: 2D IoU scalar
    '''
    return get_iou({'x1': box1[0], 'y1': box1[1], 'x2': box1[2], 'y2': box1[3]}, \
                   {'x1': box2[0], 'y1': box2[1], 'x2': box2[2], 'y2': box2[3]})




def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 0.5, 0),
                           axis=2):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    if dims.ndim == 1:
        ndim = int(dims.shape[0])  #TODO: Check this
    else:
        ndim = int(dims.shape[1])
    # ndim = int(dims.shape[0])  # TODO: Check this
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def rotation_3d_in_axis(
        points,
        angles,
        axis: int = 0,
        return_mat: bool = False,
        clockwise: bool = False
):
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    # Convert angles to float
    angles = torch.as_tensor(angles, dtype=torch.float32)

    # Convert points to float
    points = torch.as_tensor(points, dtype=torch.float32)

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
           points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
                                               f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


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