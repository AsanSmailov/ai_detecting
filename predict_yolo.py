# yolo_predict.py

import torch
import cv2
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

class YoloPredictor:
    def __init__(self, weights_path="best.pt", device='cpu', imgsz=(640, 640)):
        self.device = select_device(device)
        self.imgsz = imgsz
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.model.eval()
        self.names = self.model.names

    def predict(self, image_bgr):
        """
        Выполняет предсказание на входном BGR-изображении (numpy).
        Возвращает:
            - изображение с bounding boxes
            - список: [([x1, y1, x2, y2], conf, cls), ...]
        """
        original_shape = image_bgr.shape[:2]
        img = cv2.resize(image_bgr, self.imgsz)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(self.device)

        boxes_info = []

        with torch.no_grad():
            pred = self.model(img_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        for *xyxy, conf, cls in pred:
            xyxy = [int(x.item()) for x in xyxy]
            confidence = float(conf.item())
            class_id = int(cls.item())

            # Сохраняем координаты и метки
            boxes_info.append((xyxy, confidence, class_id))

            # Рисуем
            label = f'{self.names[class_id]} {confidence:.2f}'
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img, boxes_info

