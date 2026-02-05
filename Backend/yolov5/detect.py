# YOLOv5 🚀 by Ultralytics
# Full-featured detect.py with webcam, video, image support (FIXED)

import argparse
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(ROOT)

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import (
    LOGGER,
    check_file,
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_boxes,
    increment_path,
    cv2,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",
    source="data/images",
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=False,
    save_img=True,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    project=ROOT / "runs/detect",
    name="exp",
    exist_ok=False,
    line_thickness=3,
):
    source = str(source)
    save_img = not nosave

    # Check source
    webcam = source.isnumeric() or source.endswith(".txt") or source.startswith(
        ("rtsp://", "http://", "https://")
    )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True)
        bs = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
        bs = 1

    # Warmup
    model.warmup(imgsz=(1 if webcam else bs, 3, *imgsz))

    # Inference loop
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        # Inference
        pred = model(im)

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )

        # Process detections
        for i, det in enumerate(pred):
            # 🔥 FIX: handle webcam (list) vs image (array)
            im0 = im0s[i].copy() if isinstance(im0s, list) else im0s.copy()

            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names)
            )

            if len(det):
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape
                ).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f"{names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))

            result = annotator.result()

            # Show
            if view_img:
                cv2.imshow("YOLOv5 LIVE", result)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return

            # Save
            if save_img and not webcam:
                cv2.imwrite(str(save_dir / Path(path).name), result)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt")
    parser.add_argument("--source", type=str, default="data/images")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[640])
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--device", default="")
    parser.add_argument("--view-img", action="store_true")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--save-crop", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--classes", nargs="+", type=int)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--project", default=ROOT / "runs/detect")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")

    opt = parser.parse_args()
    opt.imgsz = (
        opt.imgsz if len(opt.imgsz) > 1 else [opt.imgsz[0], opt.imgsz[0]]
    )
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
