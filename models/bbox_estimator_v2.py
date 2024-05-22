import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.model_utils import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from models.model_utils import point_cloud_masking, parse_output_to_tensors
from models.model_utils import FrustumPointNetLoss

from models.provider import compute_box3d_iou

from models.pointnet2_utils import *

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 4  # one cluster for each type
NUM_OBJECT_POINT = 512

g_type2class = {'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'car': 0, 'truck': 1, 'bus': 2, 'trailer': 3}


g_type_mean_size = {'car': np.array([4.6344314, 1.9600292, 1.7375569]),
                    'truck': np.array([6.936331, 2.5178623, 2.8506238]),
                    'bus': np.array([11.194943, 2.9501154, 3.4918275]),
                    'trailer': np.array([12.275775, 2.9231303, 3.87086])}


class PointNetEstimationV2(nn.Module):
    def __init__(self, n_classes=2):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(PointNetEstimationV2, self).__init__()

        self.sa1 = PointNetSetAbstraction(128, 0.1, 32, in_channel=3, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(32, 0.2, 32, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128 + 3,
                                          mlp=[128, 128, 256], group_all=True)

        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)

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


class STNxyz(nn.Module):
    def __init__(self, n_classes=3):
        super(STNxyz, self).__init__()
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


class FrustumPointNetv2(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        super(FrustumPointNetv2, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.STN = STNxyz(n_classes=self.n_classes)
        self.est = PointNetEstimationV2(n_classes=self.n_classes)
        self.Loss = FrustumPointNetLoss()

    def forward(self, data_dicts):
        # dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'one_hot', 'seg'])

        point_cloud = data_dicts.get('point_cloud')  # torch.Size([32, 4, 1024]) +

        # [32,512,5]
        point_cloud = point_cloud[:, :, :3]  # +
        point_cloud = point_cloud.permute(0, 2, 1)  # +

        # Maybe reshape reshaped_tensor = tensor.reshape(2, 3)

        one_hot = data_dicts.get('one_hot')  # torch.Size([32, 3]) # +
        bs = point_cloud.shape[0]  # +
        # If not None, use to Compute Loss

        box3d_center_label = data_dicts.get('box3d_center')  # torch.Size([32, 3]) #
        size_class_label = data_dicts.get('size_class')  # torch.Size([32, 1])
        size_residual_label = data_dicts.get('size_residual')  # torch.Size([32, 3])
        heading_class_label = data_dicts.get('angle_class')  # torch.Size([32, 1])
        heading_residual_label = data_dicts.get('angle_residual')  # torch.Size([32, 1])

        # # Mask Point Centroid
        # object_pts_xyz, mask_xyz_mean, mask = \
        #     point_cloud_masking(point_cloud, logits)

        # todo:add .cuda() and remove other cuda moving operations
        clusters_mean = torch.mean(point_cloud, 2)  # bs,3

        # T-Net
        object_pts_xyz = point_cloud.cuda()
        center_delta = self.STN(object_pts_xyz, one_hot)  # (32,3)

        clusters_mean = clusters_mean.cuda()
        stage1_center = center_delta + clusters_mean  # (32,3)

        object_pts_xyz_new = object_pts_xyz - \
                             center_delta.view(center_delta.shape[0], -1, 1).repeat(1, 1, object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot)  # (32, 59)

        center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
            parse_output_to_tensors(box_pred, stage1_center)

        box3d_center = center_boxnet + stage1_center  # bs,3

        losses = self.Loss(
            box3d_center, box3d_center_label, stage1_center, \
            heading_scores, heading_residual_normalized, \
            heading_residual, \
            heading_class_label, heading_residual_label, \
            size_scores, size_residual_normalized, \
            size_residual, \
            size_class_label, size_residual_label)

        for key in losses.keys():
            losses[key] = losses[key] / bs

        with torch.no_grad():
            iou2ds, iou3ds = compute_box3d_iou( \
                box3d_center.detach().cpu().numpy(),
                heading_scores.detach().cpu().numpy(),
                heading_residual.detach().cpu().numpy(),
                size_scores.detach().cpu().numpy(),
                size_residual.detach().cpu().numpy(),
                box3d_center_label.detach().cpu().numpy(),
                heading_class_label.detach().cpu().numpy(),
                heading_residual_label.detach().cpu().numpy(),
                size_class_label.detach().cpu().numpy(),
                size_residual_label.detach().cpu().numpy())

        metrics = {
            'iou2d': iou2ds.mean(),
            'iou3d': iou3ds.mean(),
            'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs
        }

        return losses, metrics
