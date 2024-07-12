import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from dataset.dataset import g_mean_size_arr
from utils.model_utils import get_box3d_corners, get_box3d_corners_helper

def huber_loss(error, delta=1.0):
    """
    Huber loss function.
    """
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic ** 2 + delta * linear

    return torch.mean(losses)

class BBoxEstimatorLoss(nn.Module):
    def __init__(self, return_all=False):
        super(BBoxEstimatorLoss, self).__init__()
        self.return_all = return_all

    def forward(self,
                center, center_label, stage1_center,
                heading_scores, heading_residuals_normalized, heading_residuals,
                heading_class_label, heading_residuals_label,
                size_scores, size_residuals_normalized, size_residuals,
                size_class_label, size_residuals_label,
                corner_loss_weight=10.0, box_loss_weight=1.0):
        """
        Compute the Frustum PointNet loss.

        Args:
            center: (bs, 3) torch.float32 - Predicted center of objects
            center_label: (bs, 3) - Ground truth center labels
            stage1_center: (bs, 3) torch.float32 - Predicted stage 1 center
            heading_scores: (bs, 12) torch.float32 - Scores for heading angle
            heading_residuals_normalized: (bs, 12) torch.float32 - Normalized heading residuals
            heading_residuals: (bs, 12) torch.float32 - Heading residuals
            heading_class_label: (bs,) - Ground truth heading class labels
            heading_residuals_label: (bs,) - Ground truth heading residuals
            size_scores: (bs, 8) torch.float32 - Scores for object sizes
            size_residuals_normalized: (bs, 8, 3) torch.float32 - Normalized size residuals
            size_residuals: (bs, 8, 3) torch.float32 - Size residuals
            size_class_label: (bs,) - Ground truth size class labels
            size_residuals_label: (bs, 3) - Ground truth size residuals
            corner_loss_weight: float scalar - Weight for corner loss
            box_loss_weight: float scalar - Weight for total box loss

        Returns:
            total_loss: torch.float32 - Total loss value
        """
        bs = center.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center - stage1_center, dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), heading_class_label.long())
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long().cpu()].cuda()
        heading_residuals_normalized_label = heading_residuals_label / (np.pi / NUM_HEADING_BIN)
        heading_residuals_normalized_dist = torch.sum(heading_residuals_normalized * hcls_onehot.float(), dim=1)
        heading_residuals_normalized_loss = huber_loss(heading_residuals_normalized_dist -
                                                       heading_residuals_normalized_label, delta=1.0)

        # Size Loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), size_class_label.long())
        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long().cpu()].cuda()
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)
        predicted_size_residuals_normalized_dist = torch.sum(size_residuals_normalized * scls_onehot_repeat.cuda(),
                                                             dim=1)
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda().view(1, NUM_SIZE_CLUSTER, 3)
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)
        size_residuals_label_normalized = size_residuals_label / mean_size_label.cuda()
        size_normalized_dist = torch.norm(size_residuals_label_normalized - predicted_size_residuals_normalized_dist,
                                          dim=1)
        size_residuals_normalized_loss = huber_loss(size_normalized_dist, delta=1.0)

        # Corner Loss
        corners_3d = get_box3d_corners(center, heading_residuals, size_residuals).cuda()
        gt_mask = hcls_onehot.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER).repeat(1, NUM_HEADING_BIN, 1)
        corners_3d_pred = torch.sum(
            gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1).float().cuda() * corners_3d,
            dim=[1, 2])
        heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()
        heading_label = heading_residuals_label.view(bs, 1) + heading_bin_centers.view(1, NUM_HEADING_BIN)
        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda()
        size_label = mean_sizes + size_residuals_label.view(bs, 1, 3)
        size_label = torch.sum(scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])
        corners_3d_gt = get_box3d_corners_helper(center_label, heading_label, size_label)
        corners_3d_gt_flip = get_box3d_corners_helper(center_label, heading_label + np.pi, size_label)
        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = box_loss_weight * (center_loss +
                                        heading_class_loss +
                                        size_class_loss +
                                        heading_residuals_normalized_loss * 20 +
                                        size_residuals_normalized_loss * 20 +
                                        stage1_center_loss +
                                        corner_loss_weight * corners_loss)

        if self.return_all:
            return (total_loss, box_loss_weight * center_loss,
                    box_loss_weight * heading_class_loss,
                    box_loss_weight * size_class_loss,
                    box_loss_weight * heading_residuals_normalized_loss * 20,
                    box_loss_weight * size_residuals_normalized_loss * 20,
                    box_loss_weight * stage1_center_loss,
                    box_loss_weight * corners_loss * corner_loss_weight)
        else:
            return total_loss
