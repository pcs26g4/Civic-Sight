from flask import Flask, Response
import cv2
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size, cv2
from utils.plots import Annotator
from utils.torch_utils import select_device

app = Flask(__name__)

# 1. Load YOLOv5 model
weights = "weights/last.pt"
device = select_device("")
model = DetectMultiBackend(weights, device=device)
stride, names = model.stride, model.names
imgsz = check_img_size((640, 640), s=stride)

# 2. Webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        original = frame.copy()
        resized = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(resized).to(device).float().permute(2,0,1).unsqueeze(0) / 255

        pred = model(img)
        pred = non_max_suppression(pred, 0.25, 0.45)

        annotator = Annotator(original)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original.shape).round()

                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label)

        ret, buffer = cv2.imencode('.jpg', annotator.result())
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def home():
    return "YOLOv5 Backend Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
