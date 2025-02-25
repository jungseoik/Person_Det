from ultralytics.nn.autobackend import AutoBackend
import torch
import os
import cv2
from utils.augment import preprocess_images
from utils.nms import non_max_suppression
from utils.draw import draw_boxes
from utils.utils import load_model_backend
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 코드 최상단에 추가

OD_CONFIDENCE_THRESHOLD = 0.15
NUM_MAX_DETECTIONS_IN_BATCH = 300

WEIGHTS = "PersonDet_v3.1.3.pt"
# WEIGHTS = "yolo11s.pt"
# WEIGHTS = "yolo11x.pt"
# WEIGHTS = "yolo11n.pt"

DEVICE = "cuda"
image_path = "sample.png"
im = cv2.imread(image_path)

preprocessed_tensor, letterboxed_image = preprocess_images(
    im=im,
    device=load_model_backend(DEVICE)
)
cv2.imshow('Detection', letterboxed_image)

yolov8 = AutoBackend(weights=WEIGHTS , device=load_model_backend(DEVICE))

raw_od_result = yolov8(preprocessed_tensor)
nms_od_result = non_max_suppression(
            raw_od_result,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            agnostic=True,
            max_det=NUM_MAX_DETECTIONS_IN_BATCH,
            classes=[0]
        )  # 각 batch별 최대 인원을 7명으로 제한

print(nms_od_result)

detections = nms_od_result[0]
count = nms_od_result[0].shape[0]
print(count)
result_image = draw_boxes(letterboxed_image, detections)  

cv2.imshow('Detection Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()