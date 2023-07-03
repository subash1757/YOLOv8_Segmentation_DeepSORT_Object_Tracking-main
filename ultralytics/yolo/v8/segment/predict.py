import hydra
import torch
from google.colab.patches import cv2_imshow

from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import colors, save_one_box

from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from numpy import random

import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    if label == 0:  # person
        color = (85, 45, 255)
    elif label == 2:  # Car
        color = (222, 82, 175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img, label, (c1[0], c1[1] - 2),
            0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@hydra.main(config_path="configs/yolov4.yaml", strict=False)
def main(cfg):
    init_tracker()

    source, weights, view_img, save_txt, imgsz = cfg.get("source"), cfg.get("weights"), cfg.get("view_img"), cfg.get("save_txt"), cfg.get("imgsz")
    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http")

    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = DetectionPredictor(
        cfg.get("yaml"),
        weights,
        imgsz,
        device,
        conf_thres=cfg.get("conf_thres"),
        iou_thres=cfg.get("iou_thres"),
        augment=cfg.get("augment"),
        half=half,
    )

    # Set Dataloader
    dataset = model.set_dataloader(source, imgsz)

    # Run inference
    if webcam:
        dataset_size = 0
    else:
        dataset_size = len(dataset)

    for img, _, _, path in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model.predict(img, save_img=False)

        # Apply NMS
        pred = model.non_max_suppression(pred, None, None, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = ops.box_xyxy_to_cxcywh(det[:, :4])
                det[:, :4] = ops.xywh2xyxy(det[:, :4] / gn)

                det[:, :4] = det[:, :4].clamp(min=0, max=1)

                xyxy = det[:, :4].detach().cpu().numpy()

                confs = det[:, 4].detach().cpu().numpy()

                classes = det[:, 5].detach().cpu().numpy()

                for box, conf, clss in zip(xyxy, confs, classes):
                    x, y, w, h = box

                    x *= imgsz
                    y *= imgsz
                    w *= imgsz
                    h *= imgsz

                    # Tracking
                    bbox_xywh = xyxy_to_xywh(x, y, x + w, y + h)
                    bbox_tlwh = xyxy_to_tlwh([bbox_xywh])

                    frame = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0], cv2.COLOR_RGB2BGR)

                    features = deepsort.extract_features(frame, bbox_tlwh)
                    outputs = deepsort.update(bbox_tlwh, features)[0]
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        for i, box in enumerate(bbox_xyxy):
                            x1, y1, x2, y2 = [int(i) for i in box]
                            label = f"{int(identities[i])}"
                            color = compute_color_for_labels(int(identities[i]))
                            UI_box((x1, y1, x2, y2), frame, color=color, label=label)
                    cv2_imshow(frame)

if __name__ == "__main__":
    main()
