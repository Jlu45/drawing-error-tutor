import os
import base64
import hashlib
import numpy as np
import cv2
from typing import Optional, Tuple


def load_image(image_path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    if not image_path or not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path, flags)
    return img


def encode_image_base64(image_path: str, max_size: Optional[Tuple[int, int]] = None) -> Optional[str]:
    if not image_path or not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    if max_size is not None:
        img = resize_image(img, max_size)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def decode_base64_image(b64_string: str) -> Optional[np.ndarray]:
    try:
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def resize_image(img: np.ndarray, max_size: Tuple[int, int], keep_aspect: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    max_w, max_h = max_size

    if keep_aspect:
        scale = min(max_w / w, max_h / h)
        if scale >= 1.0:
            return img
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = max_w, max_h

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def convert_to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def compute_image_hash(image_path: str) -> str:
    if not image_path or not os.path.exists(image_path):
        return ""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
