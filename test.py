from eval.set_pred_factory import evaluate_images
from eval.evaluator import calculate_metrics

# calculate_metrics()
evaluate_images(model_type='yolov11s',  conf=0.05, classes=[0])
