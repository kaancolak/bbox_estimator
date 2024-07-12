import os
import sys
import time
import torch
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

import dataset.dataset as provider
from utils.model_utils import parse_output_to_tensors

# Initialize TensorRT logger
logger = trt.Logger(trt.Logger.ERROR)
print(trt.__version__)

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        # Append the device buffer to device bindings
        bindings.append(int(device_mem))

        print(f"Binding: {binding}, Size: {size}, Dtype: {dtype}")
        # Append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

def inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU
    for host_input, device_input in inputs:
        cuda.memcpy_htod_async(device_input, host_input, stream)

    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # Transfer predictions back from the GPU
    for host_output, device_output in outputs:
        cuda.memcpy_dtoh_async(host_output, device_output, stream)

    # Synchronize the stream
    stream.synchronize()

    return [host_output for host_output, _ in outputs]

# Load engine
stn_engine_path = "pointnet_bbox_single_model.engine"
engine1 = load_engine(stn_engine_path)

# Create execution context with separate CUDA stream
context1 = engine1.create_execution_context()

# Allocate buffers for the engine
inputs1, outputs1, bindings1, stream1 = allocate_buffers(engine1)

# Load test dataset
TEST_DATASET = provider.PointCloudDataset(
    "/home/kaan/dataset_concat/",
    ['car', 'truck', 'bus', 'trailer'],
    min_points=10000,
    train=False,
    use_rotate=True,
    use_mirror=False,
    use_shift=False
)
test_loader = DataLoader(TEST_DATASET, batch_size=32, shuffle=False)

# Initialize metrics
test_n_samples = 0
test_total_loss = 0.0
test_iou2d = 0.0
test_iou3d = 0.0
test_acc = 0.0
test_iou3d_acc = 0.0
test_iou2d_acc = 0.0
test_iou2d_acc50 = 0.0

# Run inference on test dataset
for batch_dict in tqdm(test_loader):
    obj_points, seg, box_center_label, angle_class_label, angle_residual_label, size_class_label, residual_label, rot_angle_label, one_hot_vector = batch_dict

    obj_points = obj_points.transpose(2, 1)
    test_n_samples += obj_points.shape[0]

    time1 = time.time()

    # Fill pointcloud and one_hot_vector data
    np.copyto(inputs1[0][0], obj_points.cpu().numpy().ravel())
    np.copyto(inputs1[1][0], one_hot_vector.cpu().numpy().ravel())

    # Run inference on the engine
    output_data1 = inference(context1, bindings1, inputs1, outputs1, stream1)

    time2 = time.time()
    inference_time = (time2 - time1) / 32
    print("Inference Time: ", inference_time)
    print("Inference Time (ms): ", inference_time * 1000)

    stage1_center = np.array(output_data1[0]).reshape(32, -1)
    box_pred = np.array(output_data1[1]).reshape(32, -1)

    stage1_center = torch.from_numpy(stage1_center).float()
    box_pred = torch.from_numpy(box_pred).float()

    logits, mask = 0, 0
    (center_boxnet, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals) = parse_output_to_tensors(box_pred.cuda())
    center = center_boxnet + stage1_center.cuda()

    ious2d, ious3d = provider.compute_box3d_iou(
        center.cpu().detach().numpy(),
        heading_scores.cpu().detach().numpy(),
        heading_residuals.cpu().detach().numpy(),
        size_scores.cpu().detach().numpy(),
        size_residuals.cpu().detach().numpy(),
        box_center_label.cpu().detach().numpy(),
        angle_class_label.cpu().detach().numpy(),
        angle_residual_label.cpu().detach().numpy(),
        size_class_label.cpu().detach().numpy(),
        residual_label.cpu().detach().numpy()
    )

    test_iou2d += np.sum(ious2d)
    test_iou3d += np.sum(ious3d)
    test_acc = 0
    test_iou3d_acc += np.sum(ious3d >= 0.7)
    test_iou2d_acc += np.sum(ious2d >= 0.7)
    test_iou2d_acc50 += np.sum(ious2d >= 0.5)

    print("TensorRT Model Accuracy:")
    print("test_iou2d", test_iou2d / test_n_samples)
    print("test_iou3d", test_iou3d / test_n_samples)
    print("test_iou3d_acc", test_iou3d_acc / test_n_samples)
    print("test_iou2d_acc", test_iou2d_acc / test_n_samples)
    print("test_n_sample", test_n_samples)

    break