import torch
import torchvision
import onnx

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging

# from models.bbox_estimator_pointnet import BoxEstimatorPointNet
# from models.bbox_estimator_pointnet2 import BoxEstimatorPointNetPlusPlus
# from models.bbox_estimator_pointnet2_cuda import BoxEstimatorPointNetPlusPlusCuda
from models.bbox_estimator_pointnet2_tensorrt import BoxEstimatorPointNetPlusPlusTensorRT
from dataset.pointcloud_dataset import PointCloudDataset

classes = ['car', 'truck', 'bus', 'trailer']

# model = BoxEstimatorPointNetPlusPlusCuda(len(classes)).cuda()
model = BoxEstimatorPointNetPlusPlusTensorRT(len(classes)).cuda()
# weights = torch.load('/home/kaan/Downloads/weights_acc0.680-epoch198.zip')
weights = torch.load('/home/kaan/projects/bbox_estimator/weights/tensorrt_imp/acc0.674-epoch047.pth')
# weights = torch.load('/home/kaan/projects/bbox_estimator/weights/v2_cuda_10p_acc0.639-epoch063.pth')

model.load_state_dict(weights['model_state_dict'], strict=False)

model.eval()

# torch.Size([32, 512, 3])
# torch.Size([32, 4])
example_point_cloud = torch.randn(32, 512, 3).cuda()
example_one_hot_vector = torch.randn(32, 4).cuda()

# input[32, 3, 512] --> input[32, 512, 3]
example_point_cloud = example_point_cloud[:, :, :3]
example_point_cloud = example_point_cloud.permute(0, 2, 1)



# Create a dictionary to hold the inputs
input_dict = {'point_cloud': example_point_cloud, 'one_hot': example_one_hot_vector}

# Export the model to ONNX
# torch.onnx.export(model, input_dict, 'pointnet2.onnx', export_params=True)

with torch.no_grad():
    torch.onnx.export(model.STN,
                      (example_point_cloud, example_one_hot_vector, {}),
                      "pointnet2_stn.onnx",
                      opset_version=11,
                      input_names=['point_cloud', 'one_hot'],
                      output_names=['center_delta'],
                      # verbose=True
                      # dynamic_axes={'box3d_center': {0: 'batch_size'},
                      #               'parsed_pred': {0: 'batch_size'}}
                      )

    torch.onnx.export(model.est,
                      (example_point_cloud, example_one_hot_vector, {}),
                      "pointnet2_est.onnx",
                      opset_version=11,
                      input_names=['point_cloud', 'one_hot'],
                      output_names=['box_pred'],
                      # verbose=True
                      # dynamic_axes={'box3d_center': {0: 'batch_size'},
                      #               'parsed_pred': {0: 'batch_size'}}
                      )




# Verify that the ONNX model behaves the same
# (You may need to implement custom CUDA operations here if needed)

