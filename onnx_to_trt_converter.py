import numpy as np
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # this automatically init the cuda

def load_and_check_onnx(path):
    onnx_model = onnx.load(path)

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


def build_engine(log, type, path):
    # create builder and network
    logger = trt.Logger(getattr(trt.Logger, log))
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # parse onnx model into tensorrt model
    onnx_model = load_and_check_onnx(path)
    parse_onnx(onnx_model, network, logger)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int((1 << 32)))
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if type == "FP16":
        config.set_flag(trt.BuilderFlag.FP16)

    output_path = "pointnet_bbox_single_model.engine"
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
    log = "WARNING"
    type = "FP32"

    path = "pointnet_opset11.onnx"
    save_path = build_engine(log, type, path)
    print(f"save trt engine: {save_path}")
