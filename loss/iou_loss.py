import torch
import math


def compute_iou(boxes_pred, boxes_gt):
    """
    Compute Intersection over Union (IoU) between predicted bounding boxes and ground truth bounding boxes, considering yaw angle.

    Args:
    - boxes_pred (Tensor): Predicted bounding boxes of shape (B, 7) representing (center_x, center_y, center_z, width, height, length, yaw).
    - boxes_gt (Tensor): Ground truth bounding boxes of shape (B, 7) representing (center_x, center_y, center_z, width, height, length, yaw).

    Returns:
    - iou (Tensor): IoU values of shape (B,).
    """
    # Convert boxes from (center_x, center_y, center_z, width, height, length, yaw) to (min_x, min_y, min_z, max_x, max_y, max_z)
    box_pred_min = boxes_pred[:, :3] - boxes_pred[:, 3:6].clamp(min=0) / 2
    box_pred_max = boxes_pred[:, :3] + boxes_pred[:, 3:6].clamp(min=0) / 2
    box_gt_min = boxes_gt[:, :3] - boxes_gt[:, 3:6].clamp(min=0) / 2
    box_gt_max = boxes_gt[:, :3] + boxes_gt[:, 3:6].clamp(min=0) / 2

    # Rotate bounding boxes based on yaw angle
    rot_matrix_pred = torch.stack([
        torch.cos(boxes_pred[:, 6]), -torch.sin(boxes_pred[:, 6]),
        torch.sin(boxes_pred[:, 6]), torch.cos(boxes_pred[:, 6])
    ], dim=1).reshape(-1, 2, 2)
    rot_matrix_gt = torch.stack([
        torch.cos(boxes_gt[:, 6]), -torch.sin(boxes_gt[:, 6]),
        torch.sin(boxes_gt[:, 6]), torch.cos(boxes_gt[:, 6])
    ], dim=1).reshape(-1, 2, 2)
    corners_pred = torch.cat([
        torch.matmul(torch.stack([box_pred_min[:, :2], box_pred_max[:, :2]]), rot_matrix_pred.transpose(1, 2)),
        box_pred_min[:, 2:3].unsqueeze(1).repeat(1, 2),
        box_pred_max[:, 2:3].unsqueeze(1).repeat(1, 2)
    ], dim=1)
    corners_gt = torch.cat([
        torch.matmul(torch.stack([box_gt_min[:, :2], box_gt_max[:, :2]]), rot_matrix_gt.transpose(1, 2)),
        box_gt_min[:, 2:3].unsqueeze(1).repeat(1, 2),
        box_gt_max[:, 2:3].unsqueeze(1).repeat(1, 2)
    ], dim=1)

    # Compute intersection volume
    inter_min = torch.max(corners_pred[:, :3], corners_gt[:, :3])
    inter_max = torch.min(corners_pred[:, 3:], corners_gt[:, 3:])
    inter_size = torch.clamp(inter_max - inter_min, min=0)
    inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]

    # Compute union volume
    vol_pred = (boxes_pred[:, 3] * boxes_pred[:, 4] * boxes_pred[:, 5]).clamp(min=0)
    vol_gt = (boxes_gt[:, 3] * boxes_gt[:, 4] * boxes_gt[:, 5]).clamp(min=0)
    union_vol = vol_pred + vol_gt - inter_vol

    # Compute IoU
    iou = inter_vol / union_vol
    return iou


def compute_ciou_loss(boxes_pred, boxes_gt):
    """
    Compute Complete IoU (CIoU) loss between predicted bounding boxes and ground truth bounding boxes, considering yaw angle.

    Args:
    - boxes_pred (Tensor): Predicted bounding boxes of shape (B, 7) representing (center_x, center_y, center_z, width, height, length, yaw).
    - boxes_gt (Tensor): Ground truth bounding boxes of shape (B, 7) representing (center_x, center_y, center_z, width, height, length, yaw).

    Returns:
    - ciou_loss (Tensor): CIoU loss value.
    """
    # Compute IoU
    iou = compute_iou(boxes_pred, boxes_gt)

    # Compute distance between box centers
    center_pred = boxes_pred[:, :3]
    center_gt = boxes_gt[:, :3]
    dist_center = torch.norm(center_pred - center_gt, dim=1)

    # Compute enclosing bounding box diagonal
    diagonal_pred = torch.norm(boxes_pred[:, 3:6], dim=1)
    diagonal_gt = torch.norm(boxes_gt[:, 3:6], dim=1)
    diagonal = torch.maximum(diagonal_pred, diagonal_gt)

    # Compute aspect ratio term
    aspect_ratio_term = ((torch.atan(boxes_pred[:, 3] / boxes_pred[:, 5]) - torch.atan(
        boxes_gt[:, 3] / boxes_gt[:, 5])) ** 2) / (4 * (math.pi ** 2))

    # Compute CIoU loss
    ciou_loss = 1 - iou + (dist_center ** 2) / (diagonal ** 2) + aspect_ratio_term
    return ciou_loss.mean()  # Return mean CIoU loss over batch
