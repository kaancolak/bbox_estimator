import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_utils import center_to_corner_box3d_torch
from dataset.pointcloud_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from dataset.pointcloud_dataset import g_mean_size_arr


def parse_output_to_tensors(box_pred, stage1_center):
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]
    c = 3

    # heading
    heading_sin = box_pred[:, c:c + 1]
    c += 1
    heading_cos = box_pred[:, c:c + 1]
    c += 1

    # size
    sizes = box_pred[:, c:c + 3]



    return {
        'center_boxnet': center_boxnet,

        'sin_yaw': heading_sin,
        'cos_yaw': heading_cos,

        'sizes': sizes,
    }


def get_box3d_corners(center, yaw_angle_wrapped_pred, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda() \
                 + size_residual.cuda()
    sizes = mean_sizes + size_residual
    sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3) \
        .repeat(1, NUM_HEADING_BIN, 1, 1).float()

    centers = center.view(bs, 1, 1, 3).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)
    N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER



    corners_3d = center_to_corner_box3d_torch(centers.view(N, 3), sizes.view(N, 3), yaw_angle_wrapped_pred.view(N))

    return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)


def huber_loss(error, delta=1.0):  # (32,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear
    return torch.mean(losses)


class FrustumPointNetLoss(nn.Module):
    def __init__(self):
        super(FrustumPointNetLoss, self).__init__()

    def forward(self, center, stage1_center, pred_parsed, gt,
                corner_loss_weight=10.0, box_loss_weight=1.0):

        sin_yaw = pred_parsed['sin_yaw']
        cos_yaw = pred_parsed['cos_yaw']

        sizes_pred = pred_parsed['sizes']

        center_label = gt['box3d_center']

        sin_yaw_label = gt['sin_yaw']
        cos_yaw_label = gt['cos_yaw']

        sizes_label = gt['box_size']

        bs = center.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center - stage1_center, dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss

        criterion = nn.SmoothL1Loss()
        # loss_sin = criterion(torch.squeeze(sin_yaw), sin_yaw_label)
        # loss_cos = criterion(torch.squeeze(cos_yaw), cos_yaw_label)

        yaw_angle_pred = torch.atan2(sin_yaw, cos_yaw)
        yaw_angle_wrapped_pred = torch.atan2(torch.sin(yaw_angle_pred), torch.cos(yaw_angle_pred))

        yaw_angle_gt = torch.atan2(sin_yaw_label, cos_yaw_label)
        yaw_angle_wrapped_gt = torch.atan2(torch.sin(yaw_angle_gt), torch.cos(yaw_angle_gt))

        # Wrap the yaw angle difference to the range [-π, π]
        yaw_diff = yaw_angle_wrapped_pred - yaw_angle_wrapped_gt
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))

        # Loss for the direct yaw angle prediction
        loss_yaw = criterion(yaw_diff, torch.zeros_like(yaw_diff))

        # heading_loss_total = loss_sin + loss_cos + loss_yaw
        heading_loss_total = loss_yaw


        # Size loss
        size_loss = F.smooth_l1_loss(sizes_pred, torch.squeeze(sizes_label))

        # Corner Loss

        corners_3d_pred = center_to_corner_box3d_torch(center, sizes_pred, torch.squeeze(yaw_angle_wrapped_pred))

        corners_3d_gt = center_to_corner_box3d_torch(center_label, sizes_label, yaw_angle_wrapped_gt)
        corners_3d_gt_flip = center_to_corner_box3d_torch(center_label, sizes_label, yaw_angle_wrapped_gt + np.pi)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = box_loss_weight * (center_loss + \
                                        heading_loss_total * 20 +
                                        size_loss * 10 + \
                                        stage1_center_loss * 20 + \
                                        corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'heading_loss': heading_loss_total * 20,
            'size_loss': size_loss * 10,
            'stage1_center_loss': stage1_center_loss * 20,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }
        return losses
