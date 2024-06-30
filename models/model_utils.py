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
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residual_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residual = \
        heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]

    c += 4
    size_residual_normalized = \
        box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    c += 3 * NUM_SIZE_CLUSTER

    size_residual_normalized = \
        size_residual_normalized.view(bs, NUM_SIZE_CLUSTER, 3)

    size_residual = size_residual_normalized * \
                    torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()

    return {
        'center_boxnet': center_boxnet,
        'heading_scores': heading_scores,
        'heading_residual_normalized': heading_residual_normalized,
        'heading_residual': heading_residual,
        'size_scores': size_scores,
        'size_residual_normalized': size_residual_normalized,
        'size_residual': size_residual
    }

def parse_output_to_tensors_cpu(box_pred):
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]
    c = 3

    # heading
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residual_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residual = \
        heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]

    c += 4
    size_residual_normalized = \
        box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    c += 3 * NUM_SIZE_CLUSTER

    size_residual_normalized = \
        size_residual_normalized.view(bs, NUM_SIZE_CLUSTER, 3)

    size_residual = size_residual_normalized * \
                    torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1)

    return {
        'center_boxnet': center_boxnet,
        'heading_scores': heading_scores,
        'heading_residual_normalized': heading_residual_normalized,
        'heading_residual': heading_residual,
        'size_scores': size_scores,
        'size_residual_normalized': size_residual_normalized,
        'size_residual': size_residual
    }


def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy( \
        np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float()

    headings = heading_residual + heading_bin_centers.view(1, -1).cuda()

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda() \
                 + size_residual.cuda()
    sizes = mean_sizes + size_residual
    sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3) \
        .repeat(1, NUM_HEADING_BIN, 1, 1).float()
    headings = headings.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER)
    centers = center.view(bs, 1, 1, 3).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)
    N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER

    corners_3d = center_to_corner_box3d_torch(centers.view(N, 3), sizes.view(N, 3), headings.view(N))

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
        heading_scores = pred_parsed['heading_scores']
        heading_residual_normalized = pred_parsed['heading_residual_normalized']
        heading_residual = pred_parsed['heading_residual']
        size_scores = pred_parsed['size_scores']
        size_residual_normalized = pred_parsed['size_residual_normalized']
        size_residual = pred_parsed['size_residual']

        center_label = gt['box3d_center']
        heading_class_label = gt['angle_class']
        heading_residual_label = gt['angle_residual']
        size_class_label = gt['size_class']
        size_residual_label = gt['size_residual']

        bs = center.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(center - stage1_center, dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_class_loss = F.nll_loss(F.log_softmax(heading_scores, dim=1), \
                                        heading_class_label.long())
        hcls_onehot = torch.eye(NUM_HEADING_BIN)[heading_class_label.long().cpu()].cuda()
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)
        heading_residual_normalized_dist = torch.sum( \
            heading_residual_normalized * hcls_onehot.float(), dim=1)

        heading_residual_normalized_loss = \
            huber_loss(heading_residual_normalized_dist -
                       heading_residual_normalized_label, delta=1.0)

        # Size loss
        size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1), \
                                     size_class_label.long())

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER)[size_class_label.long().cpu()].cuda()
        scls_onehot_repeat = scls_onehot.view(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)
        predicted_size_residual_normalized_dist = torch.sum( \
            size_residual_normalized * scls_onehot_repeat.cuda(), dim=1)
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)  # 1,8,3
        mean_size_label = torch.sum(scls_onehot_repeat * mean_size_arr_expand, dim=1)
        size_residual_label_normalized = size_residual_label / mean_size_label.cuda()

        size_normalized_dist = torch.norm(size_residual_label_normalized - \
                                          predicted_size_residual_normalized_dist, dim=1)
        size_residual_normalized_loss = huber_loss(size_normalized_dist,
                                                   delta=1.0)

        # Corner Loss
        corners_3d = get_box3d_corners(center, heading_residual, size_residual).cuda()

        gt_mask = hcls_onehot.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER) * \
                  scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER).repeat(1, NUM_HEADING_BIN, 1)
        corners_3d_pred = torch.sum( \
            gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1) \
            .float().cuda() * corners_3d, \
            dim=[1, 2])  # (bs,8,3)
        heading_bin_centers = torch.from_numpy( \
            np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float().cuda()
        heading_label = heading_residual_label.view(bs, 1) + \
                        heading_bin_centers.view(1, NUM_HEADING_BIN)

        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr) \
            .float().view(1, NUM_SIZE_CLUSTER, 3).cuda()
        size_label = mean_sizes + \
                     size_residual_label.view(bs, 1, 3)
        size_label = torch.sum(scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])

        corners_3d_gt = center_to_corner_box3d_torch(center_label, size_label, heading_label)
        corners_3d_gt_flip = center_to_corner_box3d_torch(center_label, size_label, heading_label + np.pi)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        total_loss = box_loss_weight * (center_loss + \
                                        heading_class_loss + size_class_loss + \
                                        heading_residual_normalized_loss * 20 + \
                                        size_residual_normalized_loss * 20 + \
                                        stage1_center_loss + \
                                        corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'size_class_loss': box_loss_weight * size_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss * 20,
            'size_residual_normalized_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'stage1_center_loss': box_loss_weight * size_residual_normalized_loss * 20,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }
        return losses
