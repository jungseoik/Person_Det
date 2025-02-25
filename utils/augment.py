import cv2
import numpy as np
import torch

LETTER_BOX_COLOR = (114, 114, 114)
OD_INPUT_SIZE = [640, 640]
CLS_INPUT_SIZE = [224, 224]

class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=LETTER_BOX_COLOR
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels

OD_LETTERBOX = LetterBox(new_shape=OD_INPUT_SIZE, scaleup=True, auto=False, stride=32)
CLS_LETTERBOX = LetterBox(new_shape=CLS_INPUT_SIZE, scaleup=True, auto=False, stride=32)

LETTERBOX_DICT = {
    "OBJECT_DETECTION": OD_LETTERBOX,
    "CLASSIFICATION": CLS_LETTERBOX
}

def letterboxing_frame(im, letterbox_instance):
    """
    Pre-transform single input image before inference.

    Args:
        im (np.ndarray): (h, w, 3) input image
    Returns:
        (np.ndarray): Transformed image
    """
    return letterbox_instance(image=im)

def preprocess_images(im, device, half=False, type="OBJECT_DETECTION"):
    """
    Preprocesses a single image for inference.

    Args:
        im (torch.Tensor or numpy.ndarray): The input image (h, w, 3)
        device (torch.device): The device to use for computation
        half (bool, optional): Whether to convert the image to half-precision floating point. Defaults to False
        type (str, optional): Type of preprocessing ("OBJECT_DETECTION" or "CLASSIFICATION"). Defaults to "OBJECT_DETECTION"
    Returns:
        tuple: (preprocessed_tensor, letterboxed_image)
            - preprocessed_tensor (torch.Tensor): The preprocessed image tensor (1, 3, h, w)
            - letterboxed_image (np.ndarray): The letterboxed image (h, w, 3)
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        letter_im = letterboxing_frame(im, letterbox_instance=LETTERBOX_DICT[type])
        im_tensor = letter_im[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, HWC to CHW
        im_tensor = np.ascontiguousarray(im_tensor)  # contiguous
        im_tensor = torch.from_numpy(im_tensor)
        im_tensor = im_tensor.unsqueeze(0)  # add batch dimension

    im_tensor = im_tensor.to(device)
    im_tensor = im_tensor.half() if half else im_tensor.float()  # uint8 to fp16/32
    if not_tensor:
        im_tensor /= 255  # 0 - 255 to 0.0 - 1.0
    return im_tensor, letter_im