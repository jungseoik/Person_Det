from eval.set_pred_factory import evaluate_images
from assets.config import YOLO_11_N
from eval.evaluator import calculate_metrics

# calculate_metrics()
evaluate_images(model_type='clip_ebc', conf=0.15, classes=[0])
