import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from dataset.dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from utils.model_utils import parse_output_to_tensors
from models.bbox_estimator_loss import BBoxEstimatorLoss
from dataset.dataset import compute_box3d_iou

class PointNetEstimation(nn.Module):
    def __init__(self, n_classes=3):
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts, one_hot_vec):
        bs = pts.size()[0]
        out1 = F.relu(self.bn1(self.conv1(pts)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))
        out4 = F.relu(self.bn4(self.conv4(out3)))
        global_feat = torch.max(out4, 2, keepdim=False)[0]

        expand_one_hot_vec = one_hot_vec.view(bs, -1)
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        box_pred = self.fc3(x)
        return box_pred


class STNxyz(nn.Module):
    def __init__(self, n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        expand_one_hot_vec = one_hot_vec.view(bs, -1)
        x = torch.cat([x, expand_one_hot_vec], 1)
        x = F.relu(self.fcbn1(self.fc1(x)))
        x = F.relu(self.fcbn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class BBoxEstimator(nn.Module):
    def __init__(self, n_classes=4, n_channel=3):
        super(BBoxEstimator, self).__init__()
        self.n_classes = n_classes
        self.STN = STNxyz(n_classes=4)
        self.est = PointNetEstimation(n_classes=4)

    def forward(self, pts, one_hot_vec):
        pts = pts[:, :3, :]
        clusters_mean = torch.mean(pts, 2).cuda()
        reshaped_center_delta = clusters_mean.view(clusters_mean.shape[0], -1, 1)
        repeated_center_delta = reshaped_center_delta.repeat(1, 1, pts.shape[-1])
        object_pts_xyz = pts - repeated_center_delta

        object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(object_pts_xyz, one_hot_vec)
        stage1_center = center_delta + clusters_mean

        object_pts_xyz_new = object_pts_xyz - center_delta.view(center_delta.shape[0], -1, 1).repeat(
            1, 1, object_pts_xyz.shape[-1])

        box_pred = self.est(object_pts_xyz_new, one_hot_vec)

        if not torch.onnx.is_in_onnx_export():
            (center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores,
             size_residuals_normalized, size_residuals) = parse_output_to_tensors(box_pred)
            center = center_boxnet + stage1_center
            return (stage1_center, center_boxnet, heading_scores, heading_residuals_normalized,
                    heading_residuals, size_scores, size_residuals_normalized, size_residuals, center)
        else:
            return box_pred, stage1_center


if __name__ == '__main__':
    points = torch.zeros(size=(32, 4, 1024), dtype=torch.float32)
    label = torch.ones(size=(32, 3))
    model = BBoxEstimator()
    outputs = model(points, label)

    if not torch.onnx.is_in_onnx_export():
        (logits, mask, stage1_center, center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals,
         size_scores, size_residuals_normalized, size_residuals, center) = outputs
        print(f'logits: {logits.shape}, {logits.dtype}')
        print(f'mask: {mask.shape}, {mask.dtype}')
        print(f'stage1_center: {stage1_center.shape}, {stage1_center.dtype}')
        print(f'center_boxnet: {center_boxnet.shape}, {center_boxnet.dtype}')
        print(f'heading_scores: {heading_scores.shape}, {heading_scores.dtype}')
        print(
            f'heading_residuals_normalized: {heading_residuals_normalized.shape}, {heading_residuals_normalized.dtype}')
        print(f'heading_residuals: {heading_residuals.shape}, {heading_residuals.dtype}')
        print(f'size_scores: {size_scores.shape}, {size_scores.dtype}')
        print(f'size_residuals_normalized: {size_residuals_normalized.shape}, {size_residuals_normalized.dtype}')
        print(f'size_residuals: {size_residuals.shape}, {size_residuals.dtype}')
        print(f'center: {center.shape}, {center.dtype}')

        loss = BBoxEstimatorLoss()
        mask_label = torch.zeros
