
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from dataset.pointcloud_dataset import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from models.model_utils import parse_output_to_tensors
from models.model_utils import FrustumPointNetLoss

from utils.box_utils import compute_box3d_iou
from models.pointnet2_utils import *


from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
# from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNetPPBoxEstimationCuda(nn.Module):
    def __init__(self, n_classes=2):
        super(PointNetPPBoxEstimationCuda, self).__init__()

        self.sa1 = PointnetSAModule(
            npoint=128,
            radius=0.2,
            nsample=64,
            mlp=[0, 64, 64, 128],
            use_xyz=True,
        )

        self.sa2 = PointnetSAModule(
            npoint=32,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
        )

        self.sa3 = PointnetSAModule(
            mlp=[256, 256, 256, 512],
            use_xyz=True,
        )

        self.fc1 = nn.Linear(512 + n_classes, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

    def forward(self, pts, one_hot_vec):
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        # (B, N, 3) tensor of the xyz coordinates of the features
        pts = pts.permute(0, 2, 1)

        pts, features = self._break_up_pc(pts)

        l1_xyz, l1_points = self.sa1(pts, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        global_feat = l3_points.view(bs, -1)

        print(global_feat.shape)

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.bn1(self.fc1(expand_global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        box_pred = self.fc3(x)

        return box_pred

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


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


class BoxEstimatorPointNetPlusPlusCuda(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(BoxEstimatorPointNetPlusPlusCuda, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.STN = CenterRegressionNet(n_classes=self.n_classes)
        self.est = PointNetPPBoxEstimationCuda(n_classes=self.n_classes)
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

        # print("object_pts_xyz_new.shape")
        # print(object_pts_xyz_new.shape)
        box_pred = self.est(object_pts_xyz_new, one_hot)

        # parsed_pred = parse_output_to_tensors(box_pred, stage1_center)
        #
        # box3d_center = parsed_pred.get('center_boxnet') + stage1_center

        if not torch.onnx.is_in_onnx_export():

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

        else:

            bbox = {
                'box_pred': box_pred,
                'stage1_center': stage1_center,
            }
            print("box_pred")
            print(box_pred.shape)
            print("stage1_center")
            print(stage1_center.shape)


            return bbox
