import torch
import torch.nn as nn
import torch.nn.functional as F


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, num_classes):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channel + num_classes, mlp[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(mlp[0], mlp[1], kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(mlp[1], mlp[2], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mlp[0])
        self.bn2 = nn.BatchNorm2d(mlp[1])
        self.bn3 = nn.BatchNorm2d(mlp[2])

    def forward(self, xyz, points, one_hot_vec):
        # xyz: centroid of regions, points: features of regions
        # Concatenate one-hot vector with input features
        one_hot_vec = one_hot_vec.unsqueeze(2).repeat(1, 1, xyz.size(1))
        points = torch.cat([points, one_hot_vec], dim=1)

        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)
        xyz, points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        points = self.conv1(points)
        points = self.bn1(points)
        points = F.relu(points)
        points = self.conv2(points)
        points = self.bn2(points)
        points = F.relu(points)
        points = self.conv3(points)
        points = self.bn3(points)
        points = F.relu(points)
        points = torch.max(points, 3)[0]
        return xyz, points

class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, mlp[0], kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(mlp[0], mlp[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mlp[0])
        self.bn2 = nn.BatchNorm1d(mlp[1])

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        dist, idx = three_nn(xyz1, xyz2)
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        points_concat = torch.cat([interpolated_points, points1], dim=1)
        points_concat = points_concat.permute(0, 2, 1)
        points_concat = self.conv1(points_concat)
        points_concat = self.bn1(points_concat)
        points_concat = F.relu(points_concat)
        points_concat = self.conv2(points_concat)
        points_concat = self.bn2(points_concat)
        points_concat = F.relu(points_concat)
        return points_concat


class PointNet2BoundingBox(nn.Module):
    def __init__(self, num_classes):
        super(PointNet2BoundingBox, self).__init__()
        self.sa1 = SetAbstraction(npoint=512, radius=0.2, nsample=64, in_channel=3, mlp=[64, 64, 128], num_classes=num_classes)
        self.sa2 = SetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], num_classes=num_classes)
        self.sa3 = SetAbstraction(npoint=1, radius=0.8, nsample=64, in_channel=256, mlp=[256, 512, 1024], num_classes=num_classes)
        self.fp3 = FeaturePropagation(in_channel=512 + 1024, mlp=[512, 512])
        self.fp2 = FeaturePropagation(in_channel=256 + 512, mlp=[256, 256])
        self.fp1 = FeaturePropagation(in_channel=0 + 256, mlp=[256, 128])

        self.fc_center = nn.Linear(128, 3)
        self.fc_dimensions = nn.Linear(128, 3)
        self.fc_yaw = nn.Linear(128, 1)  # Output representation

    def forward(self, xyz, one_hot_vec):
        l1_xyz, l1_points = self.sa1(xyz, None, one_hot_vec)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, one_hot_vec)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, one_hot_vec)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        xyz = xyz.permute(0, 2, 1)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        center = self.fc_center(l0_points.permute(0, 2, 1))
        dimensions = self.fc_dimensions(l0_points.permute(0, 2, 1))
        orientation = self.fc_orientation(l0_points.permute(0, 2, 1))

        return center, dimensions, orientation

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample points from the input point cloud and group points within a radius around each sampled point.

    Args:
    - npoint (int): Number of points to sample.
    - radius (float): Radius for grouping points.
    - nsample (int): Number of points to group around each sampled point.
    - xyz (Tensor): Input point cloud of shape (B, N, 3) where B is the batch size and N is the number of points.
    - points (Tensor): Features associated with each point of shape (B, C, N) where C is the number of channels/features.

    Returns:
    - grouped_xyz (Tensor): Grouped point coordinates of shape (B, npoint, nsample, 3).
    - grouped_points (Tensor): Grouped features of shape (B, npoint, nsample, C).
    """
    B, N, _ = xyz.size()
    S = npoint
    xyz = xyz.view(B, 1, N, 3).repeat(1, S, 1, 1)
    idx = torch.randint(0, N, (B, npoint))
    idx = idx.view(B, npoint, 1).repeat(1, 1, nsample)
    grouped_xyz = torch.gather(xyz, 2, idx)
    grouped_xyz -= xyz
    grouped_points = torch.gather(points, 2, idx)
    return grouped_xyz, grouped_points


def three_nn(xyz1, xyz2):
    """
    Find the three nearest neighbors for each point in xyz1 from xyz2.

    Args:
    - xyz1 (Tensor): First point cloud of shape (B, N, 3) where B is the batch size and N is the number of points.
    - xyz2 (Tensor): Second point cloud of shape (B, M, 3) where M is the number of points.

    Returns:
    - dist (Tensor): Distance of the three nearest neighbors for each point in xyz1 from xyz2 of shape (B, N, 3).
    - idx (Tensor): Indices of the three nearest neighbors for each point in xyz1 from xyz2 of shape (B, N, 3).
    """
    B, N, _ = xyz1.size()
    _, M, _ = xyz2.size()
    dist = torch.cdist(xyz1, xyz2, p=2)  # Pairwise Euclidean distance
    dist, idx = torch.topk(dist, 3, largest=False)  # Find indices of three nearest neighbors
    return dist, idx


def three_interpolate(points, idx, weight):
    """
    Interpolate points based on indices and weights.

    Args:
    - points (Tensor): Input points of shape (B, N, C) where B is the batch size, N is the number of points, and C is the number of channels/features.
    - idx (Tensor): Indices of points to interpolate from of shape (B, N, 3).
    - weight (Tensor): Interpolation weights of shape (B, N, 3).

    Returns:
    - interpolated_points (Tensor): Interpolated points of shape (B, N, C).
    """
    B, N, C = points.size()
    interpolated_points = torch.sum(weight.unsqueeze(3) * torch.gather(points, 1, idx), dim=2)
    return interpolated_points
