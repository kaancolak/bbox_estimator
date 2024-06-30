import numpy as np
from scipy.spatial import ConvexHull
import torch

from dataset.pointcloud_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from dataset.pointcloud_dataset import g_class2type, g_type_mean_size
from scipy.spatial import ConvexHull


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
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def calculate_2d_iou(box1, box2):
    """
    Calculate 2D IoU for two bounding boxes.
    box1 and box2 are (4, 2) shaped arrays representing the corners of the 2D bounding boxes.
    """
    x11, y11 = np.min(box1[:, 0]), np.min(box1[:, 1])
    x12, y12 = np.max(box1[:, 0]), np.max(box1[:, 1])
    x21, y21 = np.min(box2[:, 0]), np.min(box2[:, 1])
    x22, y22 = np.max(box2[:, 0]), np.max(box2[:, 1])

    xi1, yi1 = max(x11, x21), max(y11, y21)
    xi2, yi2 = min(x12, x22), min(y12, y22)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def calculate_3d_iou(box1, box2):
    """
    Calculate 3D IoU for two bounding boxes.
    box1 and box2 are (8, 3) shaped arrays representing the corners of the 3D bounding boxes.
    """
    x11, y11, z11 = np.min(box1[:, 0]), np.min(box1[:, 1]), np.min(box1[:, 2])
    x12, y12, z12 = np.max(box1[:, 0]), np.max(box1[:, 1]), np.max(box1[:, 2])
    x21, y21, z21 = np.min(box2[:, 0]), np.min(box2[:, 1]), np.min(box2[:, 2])
    x22, y22, z22 = np.max(box2[:, 0]), np.max(box2[:, 1]), np.max(box2[:, 2])

    xi1, yi1, zi1 = max(x11, x21), max(y11, y21), max(z11, z21)
    xi2, yi2, zi2 = min(x12, x22), min(y12, y22), min(z12, z22)

    inter_volume = max(0, xi2 - xi1) * max(0, yi2 - yi1) * max(0, zi2 - zi1)
    box1_volume = (x12 - x11) * (y12 - y11) * (z12 - z11)
    box2_volume = (x22 - x21) * (y22 - y21) * (z22 - z21)

    union_volume = box1_volume + box2_volume - inter_volume

    iou = inter_volume / union_volume if union_volume != 0 else 0

    return iou


def calculate_ious(corners1, corners2):
    """
    Calculate both 2D and 3D IoU for given bounding boxes.
    corners1 and corners2 are (8, 3) shaped arrays representing the corners of the 3D bounding boxes.
    """
    # Project to 2D by taking only the first two coordinates
    corners1_2d = corners1[:, :2]
    corners2_2d = corners2[:, :2]

    iou_2d = calculate_2d_iou(corners1_2d, corners2_2d)
    iou_3d = calculate_3d_iou(corners1, corners2)

    return iou_3d, iou_2d


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

    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)

    ymax = min(corners1[0, 2], corners2[0, 2])
    ymin = max(corners1[4, 2], corners2[4, 2])

    inter_vol = inter_area * max(0.0, ymax - ymin)
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

    center_pred = center_pred.detach().cpu().numpy()

    heading_logits = pred['heading_scores'].detach().cpu().numpy()
    heading_residual = pred['heading_residual'].detach().cpu().numpy()
    size_logits = pred['size_scores'].detach().cpu().numpy()
    size_residual = pred['size_residual'].detach().cpu().numpy()

    center_label = gt['box3d_center'].detach().cpu().numpy()
    heading_class_label = gt['angle_class'].detach().cpu().numpy()
    heading_residual_label = gt['angle_residual'].detach().cpu().numpy()
    size_class_label = gt['size_class'].detach().cpu().numpy()
    size_residual_label = gt['size_residual'].detach().cpu().numpy()

    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residual[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residual[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])


        corners_3d = center_to_corner_box3d_numpy(center_pred[i], box_size, heading_angle, )

        heading_angle_label = class2angle(heading_class_label[i],
                                          heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i].item(), size_residual_label[i])

        corners_3d_label = center_to_corner_box3d_numpy(center_label[i], box_size_label,
                                                        np.squeeze(heading_angle_label), )

        # iou_3d, iou_2d = calculate_ious(corners_3d, corners_3d_label)
        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)

        # if i == 0:
        #     print("**box_size**")
        #     print(box_size)
        #     print("**heading_angle**")
        #     print(heading_angle)
        #     print("**box_size_label**")
        #     print(box_size_label)
        #     print("**heading_angle_label**")
        #     print(heading_angle_label)
        #
        #     print("iou_3d")
        #     print(iou_3d)
        #     print("iou_2d")
        #     print(iou_2d)
        #
        #     print("corners_3d")
        #     print(corners_3d)
        #     print("corners_3d_label")
        #     print(corners_3d_label)


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


def compute_box3d(center_pred,
                      heading_logits, heading_residual,
                      size_logits, size_residual):
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residual[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residual[i, size_class[i], :] \
                               for i in range(batch_size)])

    corners = []

    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])

        # corners_3d = center_to_corner_box3d_numpy(center_pred[i], box_size, heading_angle, )

        corners_3d = {
            'center': center_pred[i],
            'heading_angle': heading_angle,
            'size': box_size,
        }

        corners.append(corners_3d)



    return corners
