import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from utils import load_plugins

from dataset.pointcloud_dataset import PointCloudDataset
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import torch

from models.model_utils import parse_output_to_tensors, parse_output_to_tensors_cpu

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.pointcloud_dataset import PointCloudDataset, NUM_HEADING_BIN, g_class2type, g_type_mean_size
from utils.box_utils import center_to_corner_box3d_numpy, class2angle, class2size
from visualization.check_dataset import  class2size_test
from utils.box_utils import compute_box3d

from models.bbox_estimator_pointnet2_tensorrt import BoxEstimatorPointNetPlusPlusTensorRT


def visualize_point_cloud_with_bbox(obj_points, bbox, ax):
    """Visualizes the point cloud and bounding box for a single data point."""

    x, y, z = obj_points[:, 0], obj_points[:, 1], obj_points[:, 2]

    ax.scatter(x, y, z)


    draw_box(ax, bbox['center'], bbox['size'], bbox['heading_angle'])

def draw_box(ax, center, size, yaw):
    """Draws a 3D bounding box on the provided axes."""
    bbox_params = np.array([center.tolist() + size.tolist() + [yaw]], dtype=np.float32)
    corners = center_to_corner_box3d_numpy(bbox_params[:, :3], bbox_params[:, 3:6], bbox_params[:, 6]).squeeze(0)

    # Plot the edges of the bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)  # Vertical edges
    ]
    for start, end in edges:
        ax.plot3D(*zip(corners[start], corners[end]), color='b')


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


stn_engine_path = "/home/kaan/projects/bbox_estimator/pointnet2_stn.engine"
est_engine_path = "/home/kaan/projects/bbox_estimator/pointnet2_est.engine"
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

# Iterate over the test dataset
test_dataset = PointCloudDataset('/home/kaan/dataset_concat/', ['car', 'truck', 'bus', 'trailer'], min_points=500,
                                 train=True,
                                 augment_data=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
for batch_dict in tqdm(iterable=test_loader):
    # Preprocess the input data, make -> torch.Size([32, 3, 512])

    point_cloud = batch_dict['point_cloud']
    point_cloud = point_cloud[:, :, :3]
    point_cloud = point_cloud.permute(0, 2, 1)

    # Fill pointcloud and one_hot_vector data
    np.copyto(inputs1[0][0], point_cloud.cpu().numpy().ravel())
    np.copyto(inputs1[1][0], batch_dict['one_hot'].cpu().numpy().ravel())

    # Run inference on the first engine
    output_data1 = inference(context1, bindings1, inputs1, outputs1, stream1)
    output_data1 = np.array(output_data1).reshape(32, -1)

    # Mid-process
    # Calculate the center of the clusters
    output_data1 = torch.from_numpy(output_data1).float()
    clusters_mean = torch.mean(point_cloud, 2)
    stage1_center = output_data1 + clusters_mean
    reshaped_center_delta = stage1_center.view(stage1_center.shape[0], -1, 1)
    repeated_center_delta = reshaped_center_delta.repeat(1, 1, point_cloud.shape[-1])
    object_pts_xyz_new = point_cloud - repeated_center_delta

    # Copy the output of the first engine to the input of the second engine
    # Fill pointcloud and one_hot_vector data
    np.copyto(inputs2[0][0], object_pts_xyz_new.cpu().numpy().ravel())
    np.copyto(inputs2[1][0], batch_dict['one_hot'].cpu().numpy().ravel())

    # Run inference on the second engine
    output_data2 = inference(context2, bindings2, inputs2, outputs2, stream2)
    reshaped_output_data2 = np.array(output_data2[0]).reshape(32, 43)



    # Convert numpy array to torch tensor on GPU
    reshaped_output_data2 = torch.from_numpy(reshaped_output_data2).float()

    # # Parse the output of the second engine
    parsed_pred = parse_output_to_tensors_cpu(reshaped_output_data2)
    box3d_center = parsed_pred.get('center_boxnet') + stage1_center  # bs,3

    bboxes = compute_box3d(
        box3d_center.detach().cpu().numpy(),
        parsed_pred['heading_scores'].detach().cpu().numpy(),
        parsed_pred['heading_residual'].detach().cpu().numpy(),
        parsed_pred['size_scores'].detach().cpu().numpy(),
        parsed_pred['size_residual'].detach().cpu().numpy())

    print("Tensorrt output")
    print(bboxes[0]['center'])
    print(bboxes[0]['size'])
    print(bboxes[0]['heading_angle'])


    classes = ['car', 'truck', 'bus', 'trailer']
    model = BoxEstimatorPointNetPlusPlusTensorRT(len(classes)).cuda()
    weights = torch.load('/home/kaan/projects/bbox_estimator/weights/tensorrt_imp/acc0.674-epoch047.pth')
    model.load_state_dict(weights['model_state_dict'], strict=False)
    model.eval()
    aa = {key: value.cuda() for key, value in batch_dict.items()}
    model_output = model(aa)



    heading_angle = class2angle(batch_dict['angle_class'][0], batch_dict['angle_residual'][0], NUM_HEADING_BIN)
    box_size = class2size_test(batch_dict['size_class'][0], batch_dict['size_residual'])[0]

    print("Ground truth")
    print(batch_dict['box3d_center'][0])
    print(box_size)
    print(heading_angle)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    data_point = batch_dict['point_cloud'][0].cpu().numpy()

    x, y, z = data_point[:, 0], data_point[:, 1], data_point[:, 2]
    ax.scatter(x, y, z)

    draw_box(ax, bboxes[0]['center'], bboxes[0]['size'], bboxes[0]['heading_angle'])

    # visualize_point_cloud_with_bbox(data_point, bboxes_gt[0], ax)
    plt.show()


    break
    input("Press key to continue...")




