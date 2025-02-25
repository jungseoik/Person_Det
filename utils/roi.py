from typing import List

import cv2
import time
import numpy as np
import time

LETTER_BOX_COLOR = (114, 114, 114)

def crop_region(frame: np.array, region: list):
    if len(region) > 0:
        region = np.array(region)
        xmin, xmax, ymin, ymax = (
            region[:, 0].min(),
            region[:, 0].max(),
            region[:, 1].min(),
            region[:, 1].max(),
        )
        h, w = frame.shape[:2]
        mask = np.zeros([h, w], dtype=np.uint8)
        cv2.drawContours(
            mask,
            [np.array(region, dtype=np.int32)],
            -1,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        dst = cv2.bitwise_and(frame, frame, mask=mask)
        dst = dst[ymin:ymax, xmin:xmax, :]
    else:
        dst = frame
    return dst


def calc_expand_coord(roi, frame_hw, expand_ratio: float = 0.05):
    if len(roi) == 0:
        return []
    xmax = np.array(roi)[:, 0].max(axis=0)
    ymax = np.array(roi)[:, 1].max(axis=0)
    xmin = np.array(roi)[:, 0].min(axis=0)
    ymin = np.array(roi)[:, 1].min(axis=0)

    width = xmax - xmin
    height = ymax - ymin
    re_xmin = max(0, int(xmin - width * expand_ratio))
    re_ymin = max(0, int(ymin - height * expand_ratio))
    re_xmax = min(frame_hw[1], int(xmax + width * expand_ratio))
    re_ymax = min(frame_hw[0], int(ymax + height * expand_ratio))

    return np.array([[re_xmin, re_ymin], [re_xmin, re_ymax], [re_xmax, re_ymax], [re_xmax, re_ymin]])


def pair_list(input_list):
    """
    리스트를 두 개씩 묶어 중첩 리스트로 변환하는 함수
    """
    if len(input_list) % 2 != 0:
        raise ValueError("Input list length must be even.")

    return np.array([input_list[i : i + 2] for i in range(0, len(input_list), 2)], dtype=np.int32)


def process_batches_with_roi(
    batches: List[np.array], user_params
) -> List[np.array]:
    """batches와 user_params를 받아 roi에 해당하는 이미지만 반환합니다.

    Args:
        batches (List[np.array]): 이미지 리스트
        user_params (List[AddStreamModel]): ROI 정보가 포함된 사용자 매개변수 리스트

    Returns:
        List[np.array]: roi에 해당하는 이미지 리스트
    """
    results = []

    for batch, user_param in zip(batches, user_params):
        # user_param에서 ROI 좌표 추출 및 변환
        coordinate = pair_list(user_param.roi.coordinates)
        # roi_info = calc_expand_coord(roi=coordinate, frame_hw=batch.shape[:2])

        # ROI에 해당하는 이미지 크롭
        cropped_image = crop_region(batch, coordinate)
        results.append(cropped_image)

    return results
