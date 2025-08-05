
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_pt', type=str, default='yolov5s6u.pt', help='Input PyTorch model')
parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
parser.add_argument('--batch', type=int, default=4, help='Batch size')
parser.add_argument('--half', action='store_true', help='Use half-precision FP16')
parser.add_argument('--device', type=str, default='cuda:0', help='Target GPU device (e.g., cuda:0 or cuda:1)')
args = parser.parse_args()
if 'cuda:' in args.device:
    os.environ['CUDA_VISIBLE_DEVICES']=args.device.replace('cuda:','')

import torch
import ultralytics
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def build_static_engine(onnx_file_path, engine_file_path:str, device, fp16=True):
    with torch.cuda.device(device):
        with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser, \
            builder.create_builder_config() as config:
            
            if fp16 and builder.platform_has_fast_fp16:
                print("Platform supports FP16, enabling FP16 optimization...")
                config.set_flag(trt.BuilderFlag.FP16)
                if 'fp16' not in engine_file_path.lower():
                    engine_file_path = engine_file_path.replace('.trt', '.FP16.trt')

            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            engine = builder.build_serialized_network(network, config)
            if engine is None:
                print("Failed to build the engine.")
                return None

            with open(engine_file_path, "wb") as f:
                f.write(engine)
            print(f"TensorRT engine saved to {engine_file_path}")

def pt2onnx(pt_model: str, imgsz=640, batch=1, half=False, device='cuda:0'):
    print(f"Using device: {device}")    
    print(f"Loading model from {pt_model}")
    model = ultralytics.YOLO(pt_model)
    print(f"Exporting to onnx with imgsz={imgsz}, batch={batch}, half={half}")
    try:
        mp = model.export(
            format='onnx',
            imgsz=imgsz,
            batch=batch,
            half=half,
            device=device,
            nms=False,
        )
        new_name = mp.replace('.onnx', 
                              f'_imgsz_{imgsz}_batch_{batch}_{"FP16" if half else "FP32"}_{device.replace(":", "@")}.onnx')
        os.rename(mp, new_name)
        print(f"Export successful: {new_name}")
        return new_name
    except Exception as e:
        print(f"Failed to export model: {e}")

def pt2trt(pt_model: str, imgsz=640, batch=1, half=False, device='cuda:0'):
    print(f"Exporting to TensorRT with imgsz={imgsz}, batch={batch}, half={half}")
    try:
        mp = pt2onnx(pt_model, imgsz, batch, half, device)
        new_name = mp.replace('.onnx', '.trt')
        build_static_engine(mp, new_name, device, fp16=half)
        print(f"Export successful: {new_name}")
    except Exception as e:
        print(f"Failed to export model: {e}")

if __name__ == '__main__':
    pt2trt(args.input_pt, args.imgsz, args.batch, args.half, args.device)
