import numpy as np
import onnx
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit  # this automatically init the cuda

# from rd3d.api import demo
from utils import load_plugins
from utils.calibrator import Calibrator


def load_and_check_onnx():
    # onnx_model = onnx.load(fold / (file + '.onnx'))
    onnx_model = onnx.load("/home/kaan/projects/bbox_estimator/pointnet2_stn.onnx")

    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("model is invalid: %s" % (e))
    return onnx_model


def parse_onnx(onnx_model, network, logger):

    print(f"onnx model opset version: {onnx_model.opset_import[0].version}")
    print(network)
    # parse onnx
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')


def set_profile(profile, onnx_model, bs=(1, 1, 8)):
    input_names = [n.name for n in onnx_model.graph.input]
    input_shapes = [[it.dim_value for it in n.type.tensor_type.shape.dim[1:]] for n in onnx_model.graph.input]
    for name, shape in zip(input_names, input_shapes):
        s1 = [bs[0]] + shape
        s2 = [bs[1]] + shape
        s3 = [bs[2]] + shape
        print(f"profile {name}: {s1}, {s2}, {s3}")
        profile.set_shape(name, s1, s2, s3)
    return profile


def build_engine(log, type):
    # create builder and network
    logger = trt.Logger(getattr(trt.Logger, log))
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # parse onnx model into tensorrt model
    onnx_model = load_and_check_onnx()
    parse_onnx(onnx_model, network, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int((1 << 32)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # profile = set_profile(builder.create_optimization_profile(), onnx_model, (1, 1, 8))
    # config.add_optimization_profile(profile)

    if type == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif type == "INT8":
        assert builder.platform_has_fast_int8
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(fold / (file + '.calib'), dataloader)
        profile = set_profile(builder.create_optimization_profile(), onnx_model, (1, 1, 8))
        config.set_calibration_profile(profile)

    output_path = "pointnet2_stn.engine"
    with builder.build_serialized_network(network, config) as engine, open(output_path, mode='wb') as f:
        f.write(engine)
    return output_path


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=Path)
    parser.add_argument('--log', default="WARNING")
    parser.add_argument('--type', type=str, default="FP32")
    # _, dataloader, args = demo(parser)
    log = "WARNING"
    type = "FP32"
    # fold = args.onnx.parent
    # file = args.onnx.stem

    save_path = build_engine(log, type)
    print(f"save trt engine: {save_path}")
