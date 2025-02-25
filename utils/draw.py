import cv2
import torch

def draw_boxes(image, detections, color=(0, 255, 0), thickness=2):
    """
    이미지에 바운딩 박스를 그리는 함수
    
    Args:
        image: OpenCV 이미지 (원본 이미지)
        detections: NMS 결과 텐서 (x1, y1, x2, y2, conf, cls)
        color: 박스 색상 (B,G,R)
        thickness: 선 두께
    """
    if len(detections) == 0:
        return image
    
    # 이미지 복사
    output_image = image.copy()
    
    # GPU 텐서를 numpy로 변환
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    
    # 각 검출 결과에 대해 박스 그리기
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])  # 정수로 변환
        
        # 바운딩 박스 그리기
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
        
        # 신뢰도 텍스트 추가
        text = f"{conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # 텍스트 배경 박스 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(output_image, 
                     (x1, y1 - text_height - 5),
                     (x1 + text_width, y1),
                     color, -1)  # -1은 채우기
                     
        # 텍스트 그리기
        cv2.putText(output_image, text, (x1, y1 - 5),
                   font, font_scale, (0, 0, 0), thickness)
        
    return output_image