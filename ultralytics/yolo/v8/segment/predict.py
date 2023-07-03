import hydra
import torch

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
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

@hydra.main(config_path=f'{ROOT}/configs', config_name="config")
def main(cfg):
    init_tracker()

    model = DetectionPredictor(cfg)
    if torch.cuda.is_available():
        model.cuda()

    cap = cv2.VideoCapture(0)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()
        img, _, _, _ = model.preprocess(img, None, None, None)
        pred = model.predict(img, None, None, None)
        pred = model.postprocess(pred, None, None, None)

        bbox_tlwh = []
        confs = []
        class_indexes = []
        for box in pred.xyxy[0]:
            x_c, y_c, w, h = xyxy_to_xywh(*box)
            obj = [x_c, y_c, w, h]
            bbox_tlwh.append(obj)
            confs.append(box[4].item())
            class_indexes.append(int(box[5].item()))

        if len(bbox_tlwh) > 0:
            xywhs = torch.Tensor(bbox_tlwh)
            confss = torch.Tensor(confs)
            cls_indexes = torch.Tensor(class_indexes)

            outputs = deepsort.update(xywhs, confss, cls_indexes, img)
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)
                for tlwh, track_id in zip(tlwh_bboxs, identities):
                    track_id = int(track_id)
                    color = compute_color_for_labels(track_id)
                    bbox_left, bbox_top, bbox_w, bbox_h = tlwh
                    bbox_right = bbox_left + bbox_w
                    bbox_bottom = bbox_top + bbox_h
                    bbox = (bbox_left, bbox_top, bbox_right, bbox_bottom)

                    if track_id not in data_deque:
                        data_deque[track_id] = deque(maxlen=64)

                    label = 'ID {}'.format(track_id)
                    UI_box(bbox, img, color=color, label=label, line_thickness=2)

                    center = (int((bbox_left + bbox_right) / 2), int(bbox_bottom))
                    data_deque[track_id].append(center)
                    for i in range(len(data_deque[track_id]) - 1, 0, -1):
                        if data_deque[track_id][i] is None or data_deque[track_id][i - 1] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(img, data_deque[track_id][i], data_deque[track_id][i - 1], color, thickness)

                    if len(data_deque[track_id]) > 63:
                        data_deque[track_id].popleft()

        cv2.imshow("Object Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
