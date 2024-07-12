import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT, g_mean_size_arr
from dataset.dataset import g_mean_size_arr

def parse_output_to_tensors(box_pred):
    """
    Parse model outputs into tensors.
    Args:
        box_pred: (bs, 59)

    Returns:
        center_boxnet: (bs, 3)
        heading_scores: (bs, 12)
        heading_residuals_normalized: (bs, 12)
        heading_residuals: (bs, 12)
        size_scores: (bs, 8)
        size_residuals_normalized: (bs, 8)
        size_residuals: (bs, 8)
    """
    bs = box_pred.shape[0]

    center_boxnet = box_pred[:, :3]
    c = 3

    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residuals_normalized = box_pred[:, c:c + NUM_HEADING_BIN]
    heading_residuals = heading_residuals_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    size_scores = box_pred[:, c:c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residuals_normalized = box_pred[:, c:c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residuals_normalized = size_residuals_normalized.view(bs, NUM_SIZE_CLUSTER, 3)
    size_residuals = size_residuals_normalized * torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()

    return (center_boxnet, heading_scores, heading_residuals_normalized,
            heading_residuals, size_scores, size_residuals_normalized, size_residuals)


def get_box3d_corners_helper(centers, headings, sizes):
    """
    Helper function to compute 3D bounding box corners.

    Args:
        centers: (N, 3)
        headings: (N,)
        sizes: (N, 3)

    Returns:
        corners_3d: (N, 8, 3)
    """
    N = centers.shape[0]
    l = sizes[:, 0].view(N, 1)
    w = sizes[:, 1].view(N, 1)
    h = sizes[:, 2].view(N, 1)

    x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.cat([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)
    z_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)

    corners = torch.cat([x_corners.view(N, 1, 8), y_corners.view(N, 1, 8), z_corners.view(N, 1, 8)], dim=1)

    c = torch.cos(headings).cuda()
    s = torch.sin(headings).cuda()
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()

    row1 = torch.stack([c, zeros, s], dim=1)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-s, zeros, c], dim=1)
    R = torch.cat([row1.view(N, 1, 3), row2.view(N, 1, 3), row3.view(N, 1, 3)], axis=1)

    corners_3d = torch.bmm(R, corners)
    corners_3d += centers.view(N, 3, 1).repeat(1, 1, 8)
    corners_3d = torch.transpose(corners_3d, 1, 2)

    return corners_3d


def get_box3d_corners(center, heading_residuals, size_residuals):
    """
    Compute 3D bounding box corners for each combination of heading and size.

    Args:
        center: (bs, 3)
        heading_residuals: (bs, NH)
        size_residuals: (bs, NS, 3)

    Returns:
        box3d_corners: (bs, NH, NS, 8, 3)
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(np.arange(0, 2 * np.pi, 2 * np.pi / NUM_HEADING_BIN)).float()
    headings = heading_residuals + heading_bin_centers.view(1, -1).cuda()

    mean_sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda() + size_residuals.cuda()
    sizes = mean_sizes + size_residuals
    sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3).repeat(1, NUM_HEADING_BIN, 1, 1).float()
    headings = headings.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER)
    centers = center.view(bs, 1, 1, 3).repeat(1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)

    N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N, 3), headings.view(N), sizes.view(N, 3))

    return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)