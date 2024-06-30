import numpy as np


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

    return iou_2d, iou_3d


import matplotlib.pyplot as plt
import numpy as np

# Coordinates for the first rectangle
rectangle1 = np.array([
    [-3.91443748, -13.00824393],
    [-2.95156231, -14.68045926],
    [-6.85104542, -16.92581331],
    [-7.81392059, -15.25359797],
    [-3.91443748, -13.00824393]  # Closing the rectangle
])

# Coordinates for the second rectangle
rectangle2 = np.array([
    [-4.53258039, -17.75032503],
    [-6.34152985, -17.66923421],
    [-6.16492937, -13.72968405],
    [-4.35597991, -13.81077487],
    [-4.53258039, -17.75032503]  # Closing the rectangle
])

corners_3d = np.array([[-3.91443748, -13.00824393, 1.98198793],
                       [-6.85104542, -16.92581331, 1.98198793],
                       [-2.95156231, -14.68045926, 1.98198793],
                       [-3.91443748, -13.00824393, 0.23495801],
                       [-6.85104542, -16.92581331, 0.23495801],
                       [-2.95156231, -14.68045926, 0.23495801],
                       [-7.81392059, -15.25359797, 1.98198793],
                       [-7.81392059, -15.25359797, 0.23495801]])
corners_3d_label = np.array(
    [[-4.53258039, - 17.75032503, 2.0917353],
     [-6.34152985, - 17.66923421, 2.0917353],
     [-6.16492937, - 13.72968405, 2.0917353],
     [-4.35597991, - 13.81077487, 2.0917353],
     [-4.53258039, - 17.75032503, 0.13287407],
     [-6.34152985, - 17.66923421, 0.13287407],
     [-6.16492937, - 13.72968405, 0.13287407],
     [-4.35597991, - 13.81077487, 0.13287407]])


def calculate_iou_2d(box1, box2):
    # Calculate intersection coordinates
    xmin_inter = max(box1[:, 0].min(), box2[:, 0].min())
    xmax_inter = min(box1[:, 0].max(), box2[:, 0].max())
    ymin_inter = max(box1[:, 1].min(), box2[:, 1].min())
    ymax_inter = min(box1[:, 1].max(), box2[:, 1].max())

    # Calculate intersection area
    if xmin_inter < xmax_inter and ymin_inter < ymax_inter:
        intersection_area = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
    else:
        intersection_area = 0.0

    # Calculate areas of each rectangle
    area_box1 = (box1[:, 0].max() - box1[:, 0].min()) * (box1[:, 1].max() - box1[:, 1].min())
    area_box2 = (box2[:, 0].max() - box2[:, 0].min()) * (box2[:, 1].max() - box2[:, 1].min())

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


# Calculate IoU for the given rectangles
iou_score = calculate_iou_2d(rectangle1, rectangle2)
print(f"IoU between the two rectangles: {iou_score:.4f}")

# Calculate IoU for the given 3D bounding boxes

from utils.box_utils import box3d_iou

iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
print(f"3D IoU between the two bounding boxes: {iou_3d:.4f}")
print(f"2D IoU between the two bounding boxes: {iou_2d:.4f}")

print("After refactoring : ")
print(corners_3d.shape)
iou_2d, iou_3d = calculate_ious(corners_3d, corners_3d_label)
print(f"2D IoU between the two rectangles: {iou_2d:.4f}")
print(f"3D IoU between the two rectangles: {iou_3d:.4f}")

# # Extract x and y coordinates for plotting
# x1, y1 = rectangle1[:, 0], rectangle1[:, 1]
# x2, y2 = rectangle2[:, 0], rectangle2[:, 1]
#
# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(x1, y1, marker='o', linestyle='-', color='b', label='Rectangle 1')
# plt.plot(x2, y2, marker='o', linestyle='-', color='r', label='Rectangle 2')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Plot of Rectangles')
# plt.legend()
# plt.grid(True)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
