from ultralytics import YOLO
import torch
import os
from assets.config import PERSON_DET , YOLO_11_N , YOLO_11_S , YOLO_11_X, IMAGE
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


model = YOLO(PERSON_DET)  

results = model.predict(IMAGE, conf=0.15, classes=[0])
# results = model.predict("sample.mp4", conf=0.15, classes=[0])

for result in results:
    count = result.boxes.shape[0]  # Boxes object for bounding box outputs
    print(count)
    # result.show()  # display to screen
