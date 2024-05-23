import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from dataset.pointcloud_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from models.model_utils import parse_output_to_tensors
from models.model_utils import FrustumPointNetLoss

from utils.box_utils import compute_box3d_iou
from models.pointnet2_utils import *


class PointNetPPBoxEstimation(nn.Module):
    def __init__(self, n_classes=2):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetPPBoxEstimation, self).__init__()

        self.sa1 = PointNetSetAbstraction(128, 0.2, 64, in_channel=3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(32, 0.4, 64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3,
                                          mlp=[256, 256, 512], group_all=True)

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

    def forward(self, pts, one_hot_vec):  # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        l1_xyz, l1_points = self.sa1(pts, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(bs, -1)

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.bn1(self.fc1(expand_global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        box_pred = self.fc3(x)

        return box_pred


class CenterRegressionNet(nn.Module):
    def __init__(self, n_classes=3):
        super(CenterRegressionNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,
        return x


class BoxEstimatorPointNetPlusPlus(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(BoxEstimatorPointNetPlusPlus, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.STN = CenterRegressionNet(n_classes=self.n_classes)
        self.est = PointNetPPBoxEstimation(n_classes=self.n_classes)
        self.Loss = FrustumPointNetLoss()

    def forward(self, data_dicts):
        point_cloud = data_dicts.get('point_cloud')
        point_cloud = point_cloud[:, :, :3]
        point_cloud = point_cloud.permute(0, 2, 1)

        one_hot = data_dicts.get('one_hot')
        bs = point_cloud.shape[0]

        center_delta = self.STN(point_cloud, one_hot)

        clusters_mean = torch.mean(point_cloud, 2).cuda()
        stage1_center = center_delta + clusters_mean

        reshaped_center_delta = center_delta.view(center_delta.shape[0], -1, 1)
        repeated_center_delta = reshaped_center_delta.repeat(1, 1, point_cloud.shape[-1])
        object_pts_xyz_new = point_cloud - repeated_center_delta

        box_pred = self.est(object_pts_xyz_new, one_hot)

        parsed_pred = parse_output_to_tensors(box_pred, stage1_center)

        box3d_center = parsed_pred.get('center_boxnet') + stage1_center

        losses = self.Loss(box3d_center, stage1_center, parsed_pred, data_dicts)

        for key in losses.keys():
            losses[key] = losses[key] / bs

        with torch.no_grad():
            iou2ds, iou3ds = compute_box3d_iou(box3d_center, parsed_pred, data_dicts)

        metrics = {
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs
        }

        return losses, metrics
