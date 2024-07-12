import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
from tqdm import tqdm

import os
import sys


from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import provider_fpointnet as provider

logger = trt.Logger(trt.Logger.ERROR)
print(trt.__version__)
from model_util_old import parse_output_to_tensors


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))

        print(f"Binding: {binding}, Size: {size}, Dtype: {dtype}")
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream


def inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    for host_input, device_input in inputs:
        cuda.memcpy_htod_async(device_input, host_input, stream)

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU.
    for host_output, device_output in outputs:
        cuda.memcpy_dtoh_async(host_output, device_output, stream)

    # Synchronize the stream
    stream.synchronize()

    return [host_output for host_output, _ in outputs]


stn_engine_path = "/home/kaan/projects/frustum_pointnets_pytorch/pointnet_bbox_stn.engine"
est_engine_path = "/home/kaan/projects/frustum_pointnets_pytorch/pointnet_bbox_est.engine"
# Load engines
engine1 = load_engine(stn_engine_path)
engine2 = load_engine(est_engine_path)

# Create execution contexts with separate CUDA streams
context1 = engine1.create_execution_context()
context2 = engine2.create_execution_context()

print("Pointnet2 STN Engine:")
# Allocate buffers for first engine
inputs1, outputs1, bindings1, stream1 = allocate_buffers(engine1)

print("Pointnet2 EST Engine:")
# Allocate buffers for second engine
inputs2, outputs2, bindings2, stream2 = allocate_buffers(engine2)

TEST_DATASET = provider.PointCloudDataset("/home/kaan/dataset_concat/", ['car', 'truck', 'bus', 'trailer'],
                                          min_points=200, train=False, use_rotate=True,
                                          use_mirror=False, use_shift=False)
test_loader = DataLoader(TEST_DATASET, batch_size=32, shuffle=False)

test_n_samples = 0
test_total_loss = 0.0
test_iou2d = 0.0
test_iou3d = 0.0
test_acc = 0.0
test_iou3d_acc = 0.0
test_iou2d_acc = 0.0
test_iou2d_acc50 = 0.0

for batch_dict in tqdm(iterable=test_loader):
    # Preprocess the input data, make -> torch.Size([32, 3, 512])

    (obj_points, seg, box_center_label, angle_class_label, angle_residual_label, size_class_label, residual_label,
     rot_angle_label, one_hot_vector) = batch_dict

    obj_points = obj_points.transpose(2, 1)
    test_n_samples += obj_points.shape[0]

    clusters_mean = torch.mean(obj_points, 2)
    reshaped_center_delta = clusters_mean.view(clusters_mean.shape[0], -1, 1)
    repeated_center_delta = reshaped_center_delta.repeat(1, 1, obj_points.shape[-1])
    object_pts_stn_input = obj_points - repeated_center_delta


    # Fill pointcloud and one_hot_vector data
    np.copyto(inputs1[0][0], object_pts_stn_input.cpu().numpy().ravel())
    np.copyto(inputs1[1][0], one_hot_vector.cpu().numpy().ravel())

    # Run inference on the first engine
    output_data1 = inference(context1, bindings1, inputs1, outputs1, stream1)
    output_data1 = np.array(output_data1).reshape(32, -1)
    print(output_data1)

    # Mid-process
    # Calculate the center of the clusters
    output_data1 = torch.from_numpy(output_data1).float()
    reshaped_center_delta = output_data1.view(output_data1.shape[0], -1, 1)
    repeated_center_delta = reshaped_center_delta.repeat(1, 1, object_pts_stn_input.shape[-1])
    object_pts_est_input = object_pts_stn_input - repeated_center_delta


    # Copy the output of the first engine to the input of the second engine
    # Fill pointcloud and one_hot_vector data
    np.copyto(inputs2[0][0], object_pts_est_input.cpu().numpy().ravel())
    np.copyto(inputs2[1][0], one_hot_vector.cpu().numpy().ravel())

    # Run inference on the second engine
    output_data2 = inference(context2, bindings2, inputs2, outputs2, stream2)
    reshaped_output_data2 = np.array(output_data2[0]).reshape(32, 59)
    print(reshaped_output_data2)

    # Convert numpy array to torch tensor on GPU
    box_pred = torch.from_numpy(reshaped_output_data2).float()
    stage1_center = output_data1.cuda() + clusters_mean.cuda()

    logits = 0
    mask = 0
    (center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores,
     size_residuals_normalized, size_residuals) = parse_output_to_tensors(box_pred.cuda(), logits, mask, stage1_center)

    center = center_boxnet + stage1_center

    ious2d, ious3d = provider.compute_box3d_iou(center.cpu().detach().numpy(),
                                                heading_scores.cpu().detach().numpy(),
                                                heading_residuals.cpu().detach().numpy(),
                                                size_scores.cpu().detach().numpy(),
                                                size_residuals.cpu().detach().numpy(),
                                                box_center_label.cpu().detach().numpy(),
                                                angle_class_label.cpu().detach().numpy(),
                                                angle_residual_label.cpu().detach().numpy(),
                                                size_class_label.cpu().detach().numpy(),
                                                residual_label.cpu().detach().numpy())

    test_iou2d += np.sum(ious2d)
    test_iou3d += np.sum(ious3d)

    test_acc = 0

    test_iou3d_acc += np.sum(ious3d >= 0.7)
    test_iou2d_acc += np.sum(ious2d >= 0.7)
    test_iou2d_acc50 += np.sum(ious2d >= 0.5)

    print("test_iou2d", test_iou2d / test_n_samples)
    print("test_iou3d", test_iou3d / test_n_samples)
    print("test_iou3d_acc", test_iou3d_acc / test_n_samples)
    print("test_iou2d_acc", test_iou2d_acc / test_n_samples)
    print("test_n_sample", test_n_samples)

    break



