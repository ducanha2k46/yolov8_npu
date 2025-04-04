import cv2
import numpy as np

from typing import Tuple, List, Dict, Any


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int],
    color=(114, 114, 114),
    auto=True,
    scaleup=True,
    stride=32,
):
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    if not scaleup:
        ratio = min(ratio, 1.0)
    new_unpad = int(round(ratio * w)), int(round(ratio * h))
    dw, dh = (new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1])
    dw /= 2
    dh /= 2

    if ratio != 1.0:
        interpolation = cv2.INTER_LINEAR if ratio > 1 else cv2.INTER_AREA
        img = cv2.resize(img, new_unpad, interpolation=interpolation)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, (dw, dh)


class PreProcessor:
    def __init__(
        self, new_shape: Tuple[int, int] = (640, 640), tensor_type: str = "uint8"
    ):
        self.new_shape = new_shape
        self.tensor_type = tensor_type

    def __call__(self, images: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        img, ratio, (padw, padh) = letterbox(images, self.new_shape)
        img = img.transpose([2, 0, 1])[::-1]  # HWC -> CHW
        preproc_params = {"ratio": ratio, "pad": (padw, padh)}
        if self.tensor_type == "uint8":
            input_ = np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.uint8)
        else:
            input_ = (
                np.ascontiguousarray(np.expand_dims(img, 0), dtype=np.float32) / 255.0
            )
        return input_, preproc_params
