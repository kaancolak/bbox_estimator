import os
import sys
import torch
import onnx
import argparse

# Import the model and any other necessary modules
#import models.pointnet_bbox_estimator as pointnet_bbox_estimator
from models.pointnet_bbox_estimator import BBoxEstimator

def export_model_to_onnx(model_path, example_point_cloud, example_one_hot_vector, onnx_model_path):
    """
    Export a PyTorch model to ONNX format.

    Args:
    - model_path (str): Path to the pretrained model checkpoint.
    - example_point_cloud (torch.Tensor): Example point cloud input tensor.
    - example_one_hot_vector (torch.Tensor): Example one-hot vector input tensor.
    - onnx_model_path (str): Output path for the ONNX model file.
    """
    # Initialize the model
    FrustumPointNet = BBoxEstimator(n_classes=4).cuda()  # Assuming 4 classes like in your example
    weight = torch.load(model_path)
    FrustumPointNet.load_state_dict(weight)
    FrustumPointNet.eval()

    # Export the model to ONNX format
    with torch.no_grad():
        torch.onnx.export(FrustumPointNet,
                          (example_point_cloud, example_one_hot_vector),
                          onnx_model_path,
                          opset_version=11,
                          input_names=['point_cloud', 'one_hot'],
                          output_names=['pred', 'stage1_center'])

    print(f"Model exported to {onnx_model_path}")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', default='log/2024-07-12_11-07-38/checkpoint/model_accuracy_0.43.pth', help='Weight path')
    parser.add_argument('--output_path', default='./', help='Output path')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_path = args.weight_path
    example_point_cloud = torch.randn(32, 3, 512).cuda()
    example_one_hot_vector = torch.randn(32, 4).cuda()
    onnx_model_path = args.output_path + "/pointnet.onnx"

    export_model_to_onnx(model_path, example_point_cloud, example_one_hot_vector, onnx_model_path)



if __name__ == '__main__':
    main()