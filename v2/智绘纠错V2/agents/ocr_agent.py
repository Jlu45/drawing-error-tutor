"""
OCR Agent
=========
基于RapidOCR的文字识别Agent。
复用原版OCRAgent核心逻辑，增加预检感知能力。
"""

import os
import sys
import cv2
import numpy as np
import hashlib
import threading
import logging
from typing import Dict, List, Optional

_v2_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _v2_root not in sys.path:
    sys.path.insert(0, _v2_root)

from utils.image_utils import imread_chinese

from agents.base import BaseAgent, AgentResult, DrawingRegion

logger = logging.getLogger("OCRAgent")


class ImageCache:
    """线程安全的图像缓存（单例模式）"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._cache = {}
                cls._instance._max_size = 20
            return cls._instance

    def _hash(self, path: str) -> str:
        return hashlib.md5(f"{path}:{os.path.getmtime(path)}".encode()).hexdigest()

    def get(self, image_path: str) -> Optional[np.ndarray]:
        h = self._hash(image_path)
        if h in self._cache:
            return self._cache[h]['img'].copy()
        return None

    def put(self, image_path: str, img: np.ndarray):
        h = self._hash(image_path)
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k]['ts'])
            del self._cache[oldest]
        self._cache[h] = {'img': img.copy(), 'ts': __import__('time').time()}


class PreprocessPipeline:
    """图像预处理管线"""
    @staticmethod
    def run(img: np.ndarray, mode: str = "ocr") -> np.ndarray:
        if img is None:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        if mode == "ocr":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            denoised = cv2.fastNlMeansDenoising(binary, h=10)
            return denoised
        elif mode == "geometry":
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            return blurred
        elif mode == "structure":
            return gray
        return gray


class OCRAgent(BaseAgent):
    """OCR文字识别Agent"""

    def __init__(self):
        super().__init__("OCR", max_retries=1)
        self.ocr = None
        self.initialize()

    def _do_initialize(self) -> bool:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            return True
        except Exception as e:
            logger.error(f"[OCR] RapidOCR init failed: {e}")
            return False

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        validation = self.validate_input(image_path)
        if validation:
            return AgentResult("OCR", False, {}, [validation], confidence=0.0)

        cache = ImageCache()
        img = cache.get(image_path)
        if img is None:
            img = imread_chinese(image_path)
            if img is None:
                return AgentResult("OCR", False, {}, [f"Cannot read: {image_path}"], confidence=0.0)
            cache.put(image_path, img)

        region = kwargs.get('region')
        work_img = img.copy()
        if region:
            h, w = work_img.shape[:2]
            x1, y1 = max(0, region.x), max(0, region.y)
            x2, y2 = min(w, region.x + region.w), min(h, region.y + region.h)
            work_img = work_img[y1:y2, x1:x2]

        # 支持通过kwargs调整预处理模式
        preprocess_mode = kwargs.get('preprocess_mode', 'ocr')
        processed = PreprocessPipeline.run(work_img, preprocess_mode)
        result, _ = self.ocr(processed)

        ocr_items = []
        if result:
            for item in result:
                bbox, text, confidence = item
                ocr_items.append({
                    'text': text,
                    'confidence': float(confidence),
                    'bbox': bbox if region is None else self._offset_bbox(bbox, region)
                })

        high_conf = sum(1 for t in ocr_items if t['confidence'] > 0.7)
        confidence = high_conf / max(len(ocr_items), 1)

        return AgentResult("OCR", True, {
            'texts': ocr_items,
            'total_count': len(ocr_items),
            'high_confidence_count': high_conf
        }, confidence=confidence)

    def _offset_bbox(self, bbox, region):
        if not bbox:
            return bbox
        return [[point[0] + region.x, point[1] + region.y] for point in bbox]
