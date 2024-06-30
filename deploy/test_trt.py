import torch
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import pycuda.autoinit
from tqdm import tqdm
from utils import load_plugins

from dataset.pointcloud_dataset import PointCloudDataset
from torch.utils.data import DataLoader

from models.model_utils import parse_output_to_tensors

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.pointcloud_dataset import PointCloudDataset, NUM_HEADING_BIN, g_class2type, g_type_mean_size
from utils.box_utils import center_to_corner_box3d_numpy, class2angle, class2size

from utils.box_utils import compute_box3d

logger = trt.Logger(trt.Logger.ERROR)
    # create engine

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

engine_file = "/home/kaan/projects/bbox_estimator/pointnet2_stn.engine"
with open(engine_file, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

pred_labels = []
bs = 32
with engine.create_execution_context() as context:
    stream = cuda.Stream()
    context.set_binding_shape(engine.get_binding_index("point_cloud"), (bs, 512, 3))
    context.set_binding_shape(engine.get_binding_index("one_hot"), (bs, 4))
    assert context.all_binding_shapes_specified

    h_inputs = {'point_cloud': np.zeros((bs, 512, 3), dtype=float),
                'one_hot': np.zeros((bs, 4), dtype=float)}
    d_inputs = {}
    h_outputs = {}
    d_outputs = {}
    t_outputs = {}
    for binding in engine:
        if engine.binding_is_input(binding):
            d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
        else:
            size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
            d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)

    test_dataset = PointCloudDataset('/home/kaan/dataset_concat/', ['car', 'truck', 'bus', 'trailer'], min_points=500, train=False,
                                     augment_data=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    for batch_dict in tqdm(iterable=test_loader):

        # batch_dict = {key: value.cuda() for key, value in batch_dict.items()}

        bs = 32
        print("aaa")
        print(engine.get_binding_index("point_cloud"))
        context.set_binding_shape(engine.get_binding_index("point_cloud"), (bs, 512, 3))
        context.set_binding_shape(engine.get_binding_index("one_hot"), (bs, 4))
        assert context.all_binding_shapes_specified

        h_inputs = {'point_cloud': np.zeros((bs, 512, 3), dtype=float),
                    'one_hot': np.zeros((bs, 4), dtype=float)}
        d_inputs = {}
        h_outputs = {}
        d_outputs = {}
        t_outputs = {}
        print(5)
        for binding in engine:
            if engine.binding_is_input(binding):
                d_inputs[binding] = cuda.mem_alloc(h_inputs[binding].nbytes)
            else:
                size = trt.volume(context.get_binding_shape(engine.get_binding_index(binding)))
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                h_outputs[binding] = cuda.pagelocked_empty(size, dtype)
                d_outputs[binding] = cuda.mem_alloc(h_outputs[binding].nbytes)
        print(6)
        print(batch_dict.keys())
        h_inputs = {'point_cloud': batch_dict['point_cloud'].cpu().numpy(),
                    'one_hot': batch_dict['one_hot'].cpu().numpy()}
        print(7)
        print(h_inputs['point_cloud'].shape)
        print(type(h_inputs['point_cloud']))
        print(type(h_inputs['one_hot']))
        print(d_inputs.keys())
        print(type(d_inputs['point_cloud']))
        print(type(d_inputs['one_hot']))

        # <class 'numpy.ndarray'>
        # <class 'numpy.ndarray'>
        # dict_keys(['point_cloud', 'one_hot'])
        # <class 'pycuda._driver.DeviceAllocation'>
        # <class 'pycuda._driver.DeviceAllocation'>


        print(7)
        for key in h_inputs:
            cuda.memcpy_htod_async(d_inputs[key], h_inputs[key], stream)
        print(77)
        stream.synchronize()

        # Prepare bindings for execution
        bindings = [int(d_inputs[k]) for k in d_inputs] + [int(d_outputs[k]) for k in d_outputs]

        # Execute asynchronously
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Synchronize CUDA stream after execution to ensure completion
        stream.synchronize()
        print(8)

        for key in h_outputs:
            cuda.memcpy_dtoh_async(h_outputs[key], d_outputs[key], stream)
        stream.synchronize()
        print(9)

        output_boxes = torch.from_numpy(h_outputs['box_pred'].reshape((bs, 43))).cuda()
        stage1_center = torch.from_numpy(h_outputs['stage1_center'].reshape((bs, 3))).cuda()

        print("output_boxes")
        print(output_boxes.shape)
        #first element of output_boxes
        print(output_boxes[0])

        print("stage1_center")
        print(stage1_center.shape)
        print(stage1_center[0])


        #get first element from batch
        data_point = batch_dict['point_cloud'][0].cpu().numpy()
        print(data_point.shape)

        parsed_pred = parse_output_to_tensors(output_boxes, stage1_center)

        box3d_center = parsed_pred.get('center_boxnet') + stage1_center  # bs,3



        bboxes = compute_box3d(
            box3d_center.detach().cpu().numpy(),
            parsed_pred['heading_scores'].detach().cpu().numpy(),
            parsed_pred['heading_residual'].detach().cpu().numpy(),
            parsed_pred['size_scores'].detach().cpu().numpy(),
            parsed_pred['size_residual'].detach().cpu().numpy())

        print(bboxes[0])
        print(batch_dict['box3d_center'][0])
        print(batch_dict['box3d_center'][0])


        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # visualize_point_cloud_with_bbox(data_point, bboxes[0], ax)
        # plt.show()
        #
        # input("Press key to continue...")


        # final_output_dicts = [{'box_pred': output_boxes[i, :output_boxes[i]],
        #                         'stage1_center': output_scores[i, :output_scores[i]]}
        #                         for i in range(bs)]
        #
        # pred_labels.extend(dataloader.dataset.generate_prediction_dicts(batch_dict, final_output_dicts,
        #                                                                     dataloader.dataset.class_names))
