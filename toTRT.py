import argparse
import ultralytics

def pt2trt(pt_model, imgsz=640, batch=1, half=False):
    print(f"Loading model from {pt_model}")
    model = ultralytics.YOLO(pt_model)
    print(f"Exporting to TensorRT with imgsz={imgsz}, batch={batch}, half={half}")
    try:
        model.export(
            format='engine',
            imgsz=imgsz,
            batch=batch,
            half=half,
        )
        print("Export successful.")
    except Exception as e:
        print(f"Failed to export model: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_pt', type=str, help='Input PyTorch model')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--half', action='store_true', help='Use half-precision FP16')
    args = parser.parse_args()

    pt2trt(args.input_pt, args.imgsz, args.batch, args.half)
