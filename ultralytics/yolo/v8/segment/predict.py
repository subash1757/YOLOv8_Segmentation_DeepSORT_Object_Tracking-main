import torch
import hydra
import cv2
from deep_sort import deepsort
from deep_sort.utils.parser import get_config
from deep_sort.utils.draw import compute_color_for_labels, bbox_to_color
from deep_sort.utils.visualization import create_unique_color_float, create_unique_color_uchar
from deep_sort.utils.visualization import draw_track_bboxes
from deep_sort.utils.visualization import draw_detection_bboxes
from deep_sort.utils.visualization import draw_eventbbox
from deep_sort.utils.visualization import draw_idenity_eventbbox
from deep_sort.deep_sort import DeepSort

deepsortcfg = get_config()
deepsortcfg.merge_from_file('deep_sort.yaml')
deepsortcfg.merge_from_file('deep_sort_R101.yaml')
deepsortcfg.MODEL.REID_CKPT = '/content/drive/MyDrive/Colab Notebooks/YOLOv4_DeepSort_Pytorch/YOLOv4_DeepSort_Pytorch-master/model_data/market1501_model.pth'
deepsort = DeepSort(deepsortcfg)

def UI_box(x, img, color=(0, 255, 0), label=None, line_thickness=None):
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

@hydra.main(config_path="configs/yolov4.yaml")
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
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thres=model.conf_thres,
            iou_thres=model.iou_thres,
            classes=model.classes,
            agnostic=model.agnostic_nms,
        )

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0s.shape
                ).round()  # scale coordinates

                for *xyxy, conf, cls in det:
                    x, y, w, h = int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])

                    _xywh = xyxy_to_xywh(x, y, x + w, y + h)
                    bbox_tlwh = xyxy_to_tlwh([_xywh])

                    frame = cv2.cvtColor(
                        im0s.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR
                    )

                    features = deepsort.extract_features(frame, bbox_tlwh)
                    outputs = deepsort.update(bbox_tlwh, features)[0]
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        for i, box in enumerate(bbox_xyxy):
                            x1, y1, x2, y2 = [int(i) for i in box]
                            label = f"{int(identities[i])}"
                            color = compute_color_for_labels(int(identities[i]))
                            UI_box(
                                (x1, y1, x2, y2),
                                frame,
                                color=color,
                                label=label,
                            )
                    cv2.imshow("Deep SORT", frame)
                    cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
