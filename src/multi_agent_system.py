import os
import cv2
import numpy as np
import json
import re
import time
import hashlib
import threading
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger("DrawingAgent")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")


@dataclass
class AgentResult:
    agent_name: str
    success: bool
    data: Dict
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class DrawingRegion:
    name: str
    x: int
    y: int
    w: int
    h: int
    ocr_text: List[Dict] = field(default_factory=list)
    geometry: Dict = field(default_factory=dict)
    description: str = ""


@dataclass
class AnalysisMetrics:
    total_time_ms: float = 0.0
    agent_timings: Dict[str, float] = field(default_factory=dict)
    agent_status: Dict[str, str] = field(default_factory=dict)
    parallel_speedup: float = 1.0
    cache_hits: int = 0
    retry_count: int = 0
    quality_score: float = 0.0


class BaseAgent(ABC):
    def __init__(self, name: str, max_retries: int = 2, timeout: float = 60.0):
        self.name = name
        self.max_retries = max_retries
        self.timeout = timeout
        self._initialized = False

    @abstractmethod
    def _do_initialize(self) -> bool:
        pass

    @abstractmethod
    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        pass

    def initialize(self) -> bool:
        try:
            self._initialized = self._do_initialize()
            status = "OK" if self._initialized else "FAIL"
            logger.info(f"[{self.name}] Initialize: {status}")
            return self._initialized
        except Exception as e:
            logger.error(f"[{self.name}] Initialize error: {e}")
            self._initialized = False
            return False

    def analyze(self, image_path: str, **kwargs) -> AgentResult:
        if not self._initialized:
            return AgentResult(self.name, False, {}, ["Agent not initialized"], confidence=0.0)

        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                result = self._do_analyze(image_path, **kwargs)
                result.execution_time_ms = (time.time() - start) * 1000
                if result.success:
                    return result
                if attempt < self.max_retries:
                    logger.warning(f"[{self.name}] Attempt {attempt+1} failed, retrying...")
                    time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                if attempt < self.max_retries:
                    logger.warning(f"[{self.name}] Exception on attempt {attempt+1}: {e}")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"[{self.name}] All attempts exhausted: {e}")
                    return AgentResult(self.name, False, {}, [str(e)],
                                       execution_time_ms=elapsed, confidence=0.0)

        return AgentResult(self.name, False, {}, ["All retries exhausted"], confidence=0.0)

    def validate_input(self, image_path: str) -> Optional[str]:
        if not image_path:
            return "Image path is empty"
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'):
            return f"Unsupported format: {ext}"
        return None


class ImageCache:
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
        self._cache[h] = {'img': img.copy(), 'ts': time.time()}

    def invalidate(self, image_path: str):
        h = self._hash(image_path)
        self._cache.pop(h, None)


class PreprocessPipeline:
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
            img = cv2.imread(image_path)
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

        processed = PreprocessPipeline.run(work_img, "ocr")
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


class GeometryAgent(BaseAgent):
    def __init__(self):
        super().__init__("Geometry", max_retries=1)
        self.initialize()

    def _do_initialize(self) -> bool:
        return True

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        validation = self.validate_input(image_path)
        if validation:
            return AgentResult("Geometry", False, {}, [validation], confidence=0.0)

        cache = ImageCache()
        img = cache.get(image_path)
        if img is None:
            img = cv2.imread(image_path)
            if img is None:
                return AgentResult("Geometry", False, {}, [f"Cannot read: {image_path}"], confidence=0.0)
            cache.put(image_path, img)

        region = kwargs.get('region')
        work_img = img.copy()
        if region:
            h, w = work_img.shape[:2]
            x1, y1 = max(0, region.x), max(0, region.y)
            x2, y2 = min(w, region.x + region.w), min(h, region.y + region.h)
            work_img = work_img[y1:y2, x1:x2]

        gray = PreprocessPipeline.run(work_img, "geometry")
        result = {
            'lines': self._detect_lines(gray),
            'circles': self._detect_circles(gray),
            'arrows': self._detect_arrows(gray, work_img),
            'contours': self._detect_contours(gray),
            'line_types': self._classify_line_types(gray),
            'dimension_structures': self._detect_dimension_structures(gray)
        }

        total_elements = len(result['lines']) + len(result['circles']) + len(result['arrows'])
        confidence = min(1.0, total_elements / 20.0)

        return AgentResult("Geometry", True, result, confidence=confidence)

    def _detect_lines(self, gray):
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=30, maxLineGap=10)
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                detected.append({
                    'start': (int(x1), int(y1)), 'end': (int(x2), int(y2)),
                    'length': float(length), 'angle': float(angle),
                    'is_horizontal': abs(angle) < 10 or abs(angle) > 170,
                    'is_vertical': 80 < abs(angle) < 100
                })
        return detected

    def _detect_circles(self, gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1,
                                    minDist=30, param1=50, param2=30,
                                    minRadius=5, maxRadius=300)
        detected = []
        if circles is not None:
            for c in circles[0]:
                detected.append({
                    'center': (int(c[0]), int(c[1])),
                    'radius': int(c[2]),
                    'is_large': c[2] > 50
                })
        return detected

    def _detect_arrows(self, gray, img):
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 2000:
                continue
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if 3 <= len(approx) <= 5:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / h if h > 0 else 0
                if 0.3 < aspect < 3.0:
                    arrows.append({'bbox': [int(x), int(y), int(w), int(h)],
                                   'area': float(area), 'vertices': len(approx)})
        return arrows

    def _detect_contours(self, gray):
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            x, y, w, h = cv2.boundingRect(cnt)
            shapes.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'area': float(area), 'circularity': float(circularity),
                'is_circle_like': circularity > 0.7,
                'is_rect_like': 0.3 < circularity < 0.7
            })
        return shapes[:50]

    def _classify_line_types(self, gray):
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=30, maxLineGap=10)
        solid_count = dashed_count = center_line_count = total = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 20:
                    continue
                total += 1
                profile = self._get_line_profile(gray, x1, y1, x2, y2)
                if profile:
                    gap_ratio = profile.get('gap_ratio', 0)
                    if gap_ratio < 0.1:
                        solid_count += 1
                    elif gap_ratio < 0.35:
                        dashed_count += 1
                    else:
                        center_line_count += 1
        return {'total_lines': total, 'solid_count': solid_count,
                'dashed_count': dashed_count, 'center_line_count': center_line_count}

    def _get_line_profile(self, gray, x1, y1, x2, y2):
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 10:
            return None
        num_samples = max(int(length / 2), 5)
        white_count = black_count = 0
        for i in range(num_samples):
            t = i / num_samples
            x, y = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
            if 0 <= y < gray.shape[0] and 0 <= x < gray.shape[1]:
                val = gray[y, x]
                if val > 200:
                    white_count += 1
                elif val < 55:
                    black_count += 1
        total = white_count + black_count
        if total == 0:
            return None
        return {'gap_ratio': white_count / total}

    def _detect_dimension_structures(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=15, maxLineGap=5)
        dim_structures = []
        if lines is None:
            return dim_structures
        short_lines = [line[0] for line in lines
                       if 15 < np.sqrt((line[0][2]-line[0][0])**2 + (line[0][3]-line[0][1])**2) < 150]
        for i, l1 in enumerate(short_lines):
            for j, l2 in enumerate(short_lines):
                if i >= j:
                    continue
                if self._are_parallel(l1, l2):
                    dist = self._line_distance(l1, l2)
                    if 8 < dist < 80:
                        dim_structures.append({
                            'line1': l1.tolist(), 'line2': l2.tolist(),
                            'distance': float(dist), 'type': 'dimension_pair'
                        })
                        if len(dim_structures) >= 20:
                            return dim_structures
        return dim_structures

    def _are_parallel(self, l1, l2, angle_thresh=10):
        a1 = np.degrees(np.arctan2(l1[3] - l1[1], l1[2] - l1[0]))
        a2 = np.degrees(np.arctan2(l2[3] - l2[1], l2[2] - l2[0]))
        diff = abs(a1 - a2)
        return diff < angle_thresh or abs(diff - 180) < angle_thresh

    def _line_distance(self, l1, l2):
        cx1, cy1 = (l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2
        cx2, cy2 = (l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2
        return np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


class StructureAgent(BaseAgent):
    def __init__(self):
        super().__init__("Structure", max_retries=1)
        self.initialize()

    def _do_initialize(self) -> bool:
        return True

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        validation = self.validate_input(image_path)
        if validation:
            return AgentResult("Structure", False, {}, [validation], confidence=0.0)

        cache = ImageCache()
        img = cache.get(image_path)
        if img is None:
            img = cv2.imread(image_path)
            if img is None:
                return AgentResult("Structure", False, {}, [f"Cannot read: {image_path}"], confidence=0.0)
            cache.put(image_path, img)

        h, w = img.shape[:2]
        regions = self._detect_regions(w, h)
        title_block = self._detect_title_block(img, w, h)
        view_areas = self._detect_view_areas(img, w, h)
        has_border = self._detect_border(img, w, h)

        confidence = 0.5
        if title_block.get('detected'):
            confidence += 0.3
        if has_border:
            confidence += 0.2

        return AgentResult("Structure", True, {
            'image_size': {'width': w, 'height': h},
            'regions': regions,
            'title_block': title_block,
            'view_areas': view_areas,
            'has_border': has_border
        }, confidence=confidence)

    def _detect_regions(self, w, h):
        regions = []
        if w > 800 and h > 600:
            regions.append(DrawingRegion("左上区域", 0, 0, w // 2, h // 2))
            regions.append(DrawingRegion("右上区域", w // 2, 0, w // 2, h // 2))
            regions.append(DrawingRegion("左下区域", 0, h // 2, w // 2, h // 2))
            regions.append(DrawingRegion("右下区域", w // 2, h // 2, w // 2, h // 2))
            regions.append(DrawingRegion("标题栏区域", int(w * 0.65), int(h * 0.85), int(w * 0.35), int(h * 0.15)))
            regions.append(DrawingRegion("中心区域", int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)))
        else:
            regions.append(DrawingRegion("全图", 0, 0, w, h))
        return regions

    def _detect_title_block(self, img, w, h):
        y1, y2 = int(h * 0.85), h
        x1, x2 = int(w * 0.65), w
        if y1 >= y2 or x1 >= x2:
            return {'detected': False}
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return {'detected': False}
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect_contours = [c for c in contours
                         if len(cv2.approxPolyDP(c, 5, True)) == 4 and cv2.contourArea(c) > 50]
        return {'detected': len(rect_contours) > 3, 'grid_cells': len(rect_contours),
                'region': [x1, y1, x2 - x1, y2 - y1]}

    def _detect_view_areas(self, img, w, h):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        views = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.02):
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            views.append({'bbox': [int(x), int(y), int(cw), int(ch)],
                          'area_ratio': float(area / (w * h))})
        views.sort(key=lambda v: v['area_ratio'], reverse=True)
        return views[:6]

    def _detect_border(self, img, w, h):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        top = gray[0:5, :].mean()
        bottom = gray[h - 5:h, :].mean()
        left = gray[:, 0:5].mean()
        right = gray[:, w - 5:w].mean()
        return all(v < 200 for v in [top, bottom, left, right])


class RuleCheckAgent(BaseAgent):
    def __init__(self):
        super().__init__("RuleCheck", max_retries=0)
        self.initialize()

    def _do_initialize(self) -> bool:
        return True

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        ocr_result = kwargs.get('ocr_result')
        geometry_result = kwargs.get('geometry_result')
        structure_result = kwargs.get('structure_result')

        errors = []
        if ocr_result and ocr_result.success:
            geo_data = geometry_result.data if geometry_result and geometry_result.success else {}
            errors.extend(self._check_dimension_rules(ocr_result.data, geo_data))
            errors.extend(self._check_tolerance_rules(ocr_result.data))
            errors.extend(self._check_title_rules(ocr_result.data))
            errors.extend(self._check_symbol_rules(ocr_result.data))
        if geometry_result and geometry_result.success:
            errors.extend(self._check_line_type_rules(geometry_result.data))
            errors.extend(self._check_geometry_completeness(geometry_result.data))
        if structure_result and structure_result.success:
            errors.extend(self._check_structure_rules(structure_result.data))

        high = sum(1 for e in errors if e.get('severity') == '高')
        medium = sum(1 for e in errors if e.get('severity') == '中')
        low = sum(1 for e in errors if e.get('severity') == '低')
        confidence = max(0, 1.0 - (high * 0.15 + medium * 0.05 + low * 0.02))

        return AgentResult("RuleCheck", True, {
            'errors': errors, 'total_errors': len(errors),
            'high_severity': high, 'medium_severity': medium, 'low_severity': low
        }, confidence=confidence)

    def _check_dimension_rules(self, ocr_data, geo_data):
        errors = []
        texts = ocr_data.get('texts', [])
        has_phi = any('Φ' in t['text'] or 'φ' in t['text'] or 'Ø' in t['text'] for t in texts)
        has_number = any(re.search(r'\d+', t['text']) for t in texts)
        if not has_number:
            errors.append({'type': '尺寸标注', 'description': '未检测到任何数字，可能缺少尺寸标注',
                           'suggestion': '按GB/T 4458.4标准添加完整尺寸标注', 'severity': '高'})
        elif not has_phi:
            circles = geo_data.get('circles', [])
            if circles:
                errors.append({'type': '尺寸标注',
                               'description': f'检测到{len(circles)}个圆但未发现直径符号Φ',
                               'suggestion': '圆形特征尺寸前应加注直径符号Φ或半径符号R', 'severity': '中'})
        dim_structs = geo_data.get('dimension_structures', [])
        for ds in dim_structs[:3]:
            dist = ds.get('distance', 0)
            if dist < 10:
                errors.append({'type': '尺寸标注',
                               'description': f'尺寸线间距过小({dist:.1f}px)，可能难以辨认',
                               'suggestion': '尺寸线间距应足够大', 'severity': '低'})
        return errors

    def _check_tolerance_rules(self, ocr_data):
        texts = ocr_data.get('texts', [])
        if not any('±' in t['text'] or '公差' in t['text'] or 'IT' in t['text'].upper() for t in texts):
            return [{'type': '公差', 'description': '未检测到公差标注',
                     'suggestion': '关键配合尺寸应标注公差（如Φ50H7/g6或线性尺寸偏差）', 'severity': '中'}]
        return []

    def _check_title_rules(self, ocr_data):
        texts = ocr_data.get('texts', [])
        title_keywords = ['名称', '图名', '标题', '材料', '比例', '日期', '设计', '审核', '班级', '学号']
        found = list(set(kw for t in texts for kw in title_keywords if kw in t['text']))
        if len(found) < 2:
            return [{'type': '标题栏',
                     'description': f"标题栏信息不完整（仅检测到: {', '.join(found) if found else '无'}）",
                     'suggestion': "标题栏应包含：图名、比例、材料、设计者、日期等", 'severity': '中'}]
        return []

    def _check_symbol_rules(self, ocr_data):
        texts = ocr_data.get('texts', [])
        if not any('Ra' in t['text'] or '粗糙度' in t['text'] for t in texts):
            return [{'type': '符号', 'description': '未检测到表面粗糙度标注',
                     'suggestion': '零件图应标注表面粗糙度要求（如Ra3.2）', 'severity': '中'}]
        return []

    def _check_line_type_rules(self, geo_data):
        errors = []
        lt = geo_data.get('line_types', {})
        total = lt.get('total_lines', 0)
        solid = lt.get('solid_count', 0)
        center = lt.get('center_line_count', 0)
        circles = geo_data.get('circles', [])
        large_circles = [c for c in circles if c.get('is_large', False)]
        if large_circles and center == 0:
            errors.append({'type': '线型',
                           'description': f'检测到{len(large_circles)}个大圆但未检测到中心线(点画线)',
                           'suggestion': '圆心位置应使用细点画线绘制中心线', 'severity': '中'})
        if total > 0 and solid < total * 0.2:
            errors.append({'type': '线型', 'description': f'实线比例偏低({solid}/{total})',
                           'suggestion': '可见轮廓线应使用粗实线', 'severity': '低'})
        return errors

    def _check_geometry_completeness(self, geo_data):
        errors = []
        lines = geo_data.get('lines', [])
        circles = geo_data.get('circles', [])
        arrows = geo_data.get('arrows', [])
        if len(lines) < 5 and len(circles) < 2:
            errors.append({'type': '几何完整性',
                           'description': '检测到的几何元素过少，图纸可能不清晰或分辨率不足',
                           'suggestion': '建议上传更高分辨率的图纸', 'severity': '高'})
        if len(arrows) < 2:
            errors.append({'type': '几何完整性', 'description': '箭头/尺寸终端检测不足',
                           'suggestion': '检查尺寸标注的终端形式是否完整', 'severity': '低'})
        return errors

    def _check_structure_rules(self, structure_data):
        errors = []
        if not structure_data.get('title_block', {}).get('detected', False):
            errors.append({'type': '结构', 'description': '未检测到标准标题栏结构',
                           'suggestion': '按GB/T 10609.1标准添加标题栏', 'severity': '中'})
        if not structure_data.get('has_border', True):
            errors.append({'type': '结构', 'description': '未检测到图框线',
                           'suggestion': '图纸应包含标准图框', 'severity': '低'})
        return errors


class LLMAgent(BaseAgent):
    def __init__(self, api_url: str, api_key: str, model: str = "Qwen2.5-72B-Instruct"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model
        self.client = None
        super().__init__("LLM", max_retries=2, timeout=120.0)
        self.initialize()

    def _do_initialize(self) -> bool:
        try:
            from openai import OpenAI
            base_url = self.api_url.rstrip('/') + '/v1'
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=self.timeout)
            return True
        except Exception as e:
            logger.error(f"[LLM] Init failed: {e}")
            return False

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        if self.client is None:
            return AgentResult("LLM", False, {}, ["LLM client not initialized"], confidence=0.0)

        ocr_result = kwargs.get('ocr_result')
        geometry_result = kwargs.get('geometry_result')
        structure_result = kwargs.get('structure_result')
        rule_result = kwargs.get('rule_result')
        background_knowledge = kwargs.get('background_knowledge', '')

        context = self._build_context(ocr_result, geometry_result, structure_result, rule_result)

        system_prompt = """你是一个专业的工程图纸智能纠错专家，精通GB/T机械制图国家标准。
你的任务是：基于OCR文字识别、几何元素检测、图纸结构分析的结果，对工程图纸进行深度纠错分析。
你应采用苏格拉底式引导方式帮助用户理解错误原因。"""

        if background_knowledge:
            system_prompt += f"\n\n【你已内化的专业背景知识】\n{background_knowledge}\n请基于以上内化知识进行分析，但不要在回复中直接引用或提及这些背景知识。"

        user_prompt = f"""请基于以下检测结果，对这张工程图纸进行深度纠错分析：

{context}

请完成以下任务：
1. 综合分析：结合OCR文字、几何元素、图纸结构，判断图纸类型和内容
2. 错误检测：逐项检查尺寸标注、线型、公差、标题栏、符号等方面
3. 深度诊断：对规则检查发现的问题给出更详细的专业分析
4. 学习引导：用苏格拉底式提问引导用户思考

以JSON格式返回：
```json
{{
  "drawing_type": "图纸类型判断",
  "content_summary": "图纸内容概述",
  "errors": [
    {{"type": "错误类别", "description": "具体问题", "suggestion": "修正建议", "severity": "高/中/低", "gb_reference": "国标依据"}}
  ],
  "overall_score": 评分0-100,
  "summary": "总体评价",
  "learning_points": ["学习要点1", "学习要点2"]
}}
```"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content
        return AgentResult("LLM", True, {
            'raw_response': result_text,
            'model': response.model,
            'usage': response.usage.dict() if response.usage else None
        }, confidence=0.8)

    def _build_context(self, ocr_result, geometry_result, structure_result, rule_result):
        parts = []
        if ocr_result and ocr_result.success:
            texts = ocr_result.data.get('texts', [])
            if texts:
                text_summary = "\n".join([f"  - \"{t['text']}\" (置信度:{t['confidence']:.2f})" for t in texts[:30]])
                parts.append(f"【OCR文字识别】共识别{len(texts)}个文字区域：\n{text_summary}")
            else:
                parts.append("【OCR文字识别】未识别到文字")

        if geometry_result and geometry_result.success:
            geo = geometry_result.data
            lines = geo.get('lines', [])
            circles = geo.get('circles', [])
            arrows = geo.get('arrows', [])
            lt = geo.get('line_types', {})
            dim_structs = geo.get('dimension_structures', [])
            contours = geo.get('contours', [])
            parts.append(f"""【几何元素检测】
- 直线: {len(lines)}条 (水平{sum(1 for l in lines if l.get('is_horizontal'))}条, 垂直{sum(1 for l in lines if l.get('is_vertical'))}条)
- 圆: {len(circles)}个 (大圆{sum(1 for c in circles if c.get('is_large'))}个)
- 箭头: {len(arrows)}个
- 尺寸线对: {len(dim_structs)}对
- 轮廓形状: {len(contours)}个
- 线型分布: 实线{lt.get('solid_count',0)}条, 虚线{lt.get('dashed_count',0)}条, 点画线{lt.get('center_line_count',0)}条""")

        if structure_result and structure_result.success:
            s = structure_result.data
            parts.append(f"""【图纸结构分析】
- 图纸尺寸: {s.get('image_size',{}).get('width',0)}x{s.get('image_size',{}).get('height',0)}px
- 标题栏: {'已检测到' if s.get('title_block',{}).get('detected') else '未检测到'} (网格单元:{s.get('title_block',{}).get('grid_cells',0)})
- 图框: {'已检测到' if s.get('has_border') else '未检测到'}
- 视图区域: {len(s.get('view_areas',[]))}个""")

        if rule_result and rule_result.success:
            r = rule_result.data
            parts.append(f"""【规则检查初步结果】
- 总错误数: {r.get('total_errors',0)}
- 高严重度: {r.get('high_severity',0)}, 中严重度: {r.get('medium_severity',0)}, 低严重度: {r.get('low_severity',0)}""")
            for err in r.get('errors', []):
                parts.append(f"  ⚠ [{err.get('severity','?')}] {err.get('type','')}: {err.get('description','')}")

        return "\n\n".join(parts)


class DrawingOrchestrator:
    def __init__(self, api_url: str, api_key: str, llm_model: str = "Qwen2.5-72B-Instruct"):
        self.agents: Dict[str, BaseAgent] = {}
        self.metrics = AnalysisMetrics()
        self._register_agents(api_url, api_key, llm_model)
        from rl_memory_unit import RLMemoryUnit
        self.rl_memory = RLMemoryUnit(state_dim=10)
        self._current_session_id = ""
        logger.info("[Orchestrator] Multi-agent system initialized (with RL Memory Unit)")

    def _register_agents(self, api_url, api_key, llm_model):
        self.agents = {
            'ocr': OCRAgent(),
            'geometry': GeometryAgent(),
            'structure': StructureAgent(),
            'rule': RuleCheckAgent(),
            'llm': LLMAgent(api_url, api_key, llm_model),
        }

    def analyze(self, image_path: str, background_knowledge: str = "") -> Dict:
        total_start = time.time()
        self.metrics = AnalysisMetrics()

        logger.info(f"[Orchestrator] Starting analysis: {image_path}")

        phase1_results = self._run_parallel_phase(image_path)

        ocr_result = phase1_results.get('ocr', AgentResult("OCR", False, {}))
        geometry_result = phase1_results.get('geometry', AgentResult("Geometry", False, {}))
        structure_result = phase1_results.get('structure', AgentResult("Structure", False, {}))

        rl_params = self.rl_memory.get_policy_params()
        ocr_result = self._enhance_ocr_if_needed(image_path, ocr_result, structure_result,
                                                   threshold=rl_params.ocr_enhance_threshold)

        rule_result = self._run_rule_check(ocr_result, geometry_result, structure_result)

        llm_result = self._run_llm_analysis(
            ocr_result, geometry_result, structure_result, rule_result, background_knowledge
        )

        result = self._merge_results(ocr_result, geometry_result, structure_result,
                                     rule_result, llm_result, rl_params)

        self.metrics.total_time_ms = (time.time() - total_start) * 1000
        self.metrics.quality_score = self._compute_quality_score(result)

        state = self.rl_memory.extract_state(result)
        action = self.rl_memory.select_action(state)
        self.rl_memory.apply_action(action)
        self._current_session_id = f"{os.path.basename(image_path)}_{int(time.time())}"
        self.rl_memory.register_session(self._current_session_id, state, action, result)

        result['metrics'] = {
            'total_time_ms': round(self.metrics.total_time_ms, 1),
            'agent_timings': {k: round(v, 1) for k, v in self.metrics.agent_timings.items()},
            'agent_status': self.metrics.agent_status,
            'quality_score': round(self.metrics.quality_score, 2),
            'parallel_speedup': round(self.metrics.parallel_speedup, 2),
            'rl_session_id': self._current_session_id,
            'rl_stats': self.rl_memory.get_stats()
        }

        logger.info(f"[Orchestrator] Complete: {self.metrics.total_time_ms:.0f}ms, "
                     f"quality={self.metrics.quality_score:.2f}, "
                     f"rl_action={self.rl_memory.dqn.predict_greedy(state)}")

        return result

    def _run_parallel_phase(self, image_path: str) -> Dict[str, AgentResult]:
        parallel_start = time.time()
        results = {}
        sequential_time = 0.0

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.agents['ocr'].analyze, image_path): 'ocr',
                executor.submit(self.agents['geometry'].analyze, image_path): 'geometry',
                executor.submit(self.agents['structure'].analyze, image_path): 'structure',
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result(timeout=60)
                    results[name] = result
                    self.metrics.agent_timings[name] = result.execution_time_ms
                    self.metrics.agent_status[name] = 'OK' if result.success else 'FAIL'
                    sequential_time += result.execution_time_ms
                    logger.info(f"[Orchestrator] {name}: {'OK' if result.success else 'FAIL'} "
                                f"({result.execution_time_ms:.0f}ms)")
                except Exception as e:
                    results[name] = AgentResult(name, False, {}, [str(e)])
                    self.metrics.agent_timings[name] = 0
                    self.metrics.agent_status[name] = 'ERROR'
                    logger.error(f"[Orchestrator] {name}: ERROR - {e}")

        parallel_time = (time.time() - parallel_start) * 1000
        if parallel_time > 0:
            self.metrics.parallel_speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0

        return results

    def _enhance_ocr_if_needed(self, image_path: str, ocr_result: AgentResult,
                                structure_result: AgentResult, threshold: int = 5) -> AgentResult:
        if not ocr_result.success or ocr_result.data.get('total_count', 0) >= threshold:
            return ocr_result
        if not structure_result.success:
            return ocr_result

        regions = structure_result.data.get('regions', [])
        for region in regions:
            if region.name == "标题栏区域":
                region_ocr = self.agents['ocr'].analyze(image_path, region=region)
                if region_ocr.success and region_ocr.data.get('total_count', 0) > 0:
                    ocr_result.data['texts'].extend(region_ocr.data.get('texts', []))
                    ocr_result.data['total_count'] = len(ocr_result.data['texts'])
                    ocr_result.data['high_confidence_count'] = sum(
                        1 for t in ocr_result.data['texts'] if t['confidence'] > 0.7)
                    logger.info(f"[Orchestrator] Title block OCR: +{region_ocr.data.get('total_count', 0)} texts")
                break

        return ocr_result

    def _run_rule_check(self, ocr_result, geometry_result, structure_result) -> AgentResult:
        start = time.time()
        result = self.agents['rule'].analyze(
            "", ocr_result=ocr_result, geometry_result=geometry_result,
            structure_result=structure_result
        )
        result.execution_time_ms = (time.time() - start) * 1000
        self.metrics.agent_timings['rule'] = result.execution_time_ms
        self.metrics.agent_status['rule'] = 'OK' if result.success else 'FAIL'
        logger.info(f"[Orchestrator] RuleCheck: {result.data.get('total_errors', 0)} errors "
                     f"({result.execution_time_ms:.0f}ms)")
        return result

    def _run_llm_analysis(self, ocr_result, geometry_result, structure_result,
                          rule_result, background_knowledge) -> AgentResult:
        start = time.time()
        result = self.agents['llm'].analyze(
            "", ocr_result=ocr_result, geometry_result=geometry_result,
            structure_result=structure_result, rule_result=rule_result,
            background_knowledge=background_knowledge
        )
        result.execution_time_ms = (time.time() - start) * 1000
        self.metrics.agent_timings['llm'] = result.execution_time_ms
        self.metrics.agent_status['llm'] = 'OK' if result.success else 'FAIL'
        logger.info(f"[Orchestrator] LLM: {'OK' if result.success else 'FAIL'} "
                     f"({result.execution_time_ms:.0f}ms)")

        if not result.success:
            result = self._generate_local_analysis(ocr_result, geometry_result,
                                                    structure_result, rule_result)
            result.execution_time_ms = (time.time() - start) * 1000
            self.metrics.agent_status['llm'] = 'DEGRADED'
            logger.info(f"[Orchestrator] LLM degraded to local analysis")

        return result

    def _generate_local_analysis(self, ocr_result, geometry_result,
                                  structure_result, rule_result) -> AgentResult:
        ocr_texts = ocr_result.data.get('texts', []) if ocr_result.success else []
        geo = geometry_result.data if geometry_result.success else {}
        structure = structure_result.data if structure_result.success else {}
        rule_errors = rule_result.data.get('errors', []) if rule_result.success else []

        drawing_type = "工程图纸"
        content_parts = []
        for t in ocr_texts[:10]:
            content_parts.append(t.get('text', ''))
        content_summary = "、".join(content_parts) if content_parts else "未识别到文字内容"

        if any('减速' in t.get('text', '') or '轴' in t.get('text', '') for t in ocr_texts):
            drawing_type = "减速器相关图纸"
        elif any('装配' in t.get('text', '') for t in ocr_texts):
            drawing_type = "装配图"
        elif any('零件' in t.get('text', '') for t in ocr_texts):
            drawing_type = "零件图"

        local_errors = []
        for e in rule_errors:
            local_errors.append({
                'type': e.get('type', ''),
                'description': e.get('description', ''),
                'suggestion': e.get('suggestion', ''),
                'severity': e.get('severity', '中'),
                'gb_reference': 'GB/T 4458.4' if '尺寸' in e.get('type', '') else
                               'GB/T 4457.4' if '线型' in e.get('type', '') else
                               'GB/T 1800.1' if '公差' in e.get('type', '') else
                               'GB/T 10609.1' if '标题栏' in e.get('type', '') else
                               'GB/T 131' if '符号' in e.get('type', '') else ''
            })

        total = rule_result.data.get('total_errors', 0) if rule_result.success else 0
        high = rule_result.data.get('high_severity', 0) if rule_result.success else 0
        if total == 0:
            summary = "图纸整体符合机械制图基本规范，未发现明显错误。建议进一步检查细节部分。"
        elif high > 0:
            summary = f"图纸存在{high}个高严重度问题需优先修正，共{total}个问题需要关注。"
        else:
            summary = f"图纸存在{total}个中低严重度问题，建议按规范修正。"

        learning_points = []
        error_types = set(e.get('type', '') for e in rule_errors)
        if '尺寸标注' in error_types:
            learning_points.append("尺寸标注应完整、清晰、不交叉，圆形特征需标注Φ符号")
        if '线型' in error_types:
            learning_points.append("粗实线用于可见轮廓，虚线用于不可见轮廓，点画线用于中心线和对称线")
        if '公差' in error_types:
            learning_points.append("关键配合尺寸应标注公差，公差等级选择需考虑加工精度和配合要求")
        if '标题栏' in error_types:
            learning_points.append("标题栏应包含图名、比例、材料、设计者、日期等基本信息")
        if '符号' in error_types:
            learning_points.append("表面粗糙度、形位公差等符号标注需符合GB/T标准")
        if not learning_points:
            learning_points.append("请仔细检查图纸细节，确保所有标注符合GB/T制图标准")

        local_json = json.dumps({
            'drawing_type': drawing_type,
            'content_summary': content_summary,
            'errors': local_errors,
            'overall_score': max(0, 100 - total * 8),
            'summary': summary,
            'learning_points': learning_points
        }, ensure_ascii=False)

        return AgentResult("LLM", True, {
            'raw_response': local_json,
            'model': 'local_rule_engine',
            'usage': None
        }, confidence=0.5)

    def _compute_quality_score(self, result: Dict) -> float:
        score = 0.0
        ocr_count = len(result.get('ocr_results', []))
        if ocr_count > 0:
            score += min(0.3, ocr_count / 50.0)
        error_count = result.get('report', {}).get('total_errors', 0)
        if error_count > 0:
            score += 0.3
        api_result = result.get('api_result')
        if api_result:
            score += 0.2
        geo = result.get('geo_result')
        if geo:
            total_elements = len(geo.get('lines', [])) + len(geo.get('circles', []))
            if total_elements > 5:
                score += 0.2
        return min(1.0, score)

    def _merge_results(self, ocr_result, geometry_result, structure_result,
                       rule_result, llm_result, rl_params=None) -> Dict:
        ocr_texts = ocr_result.data.get('texts', []) if ocr_result.success else []
        detection_items = []
        if geometry_result.success:
            geo = geometry_result.data
            for l in geo.get('lines', [])[:10]:
                detection_items.append({'class': '直线', 'confidence': 1.0,
                                        'bbox': [l['start'][0], l['start'][1], l['end'][0], l['end'][1]]})
            for c in geo.get('circles', []):
                detection_items.append({'class': '圆', 'confidence': 1.0,
                                        'bbox': [c['center'][0]-c['radius'], c['center'][1]-c['radius'],
                                                 c['center'][0]+c['radius'], c['center'][1]+c['radius']]})
            for a in geo.get('arrows', []):
                detection_items.append({'class': '箭头', 'confidence': 0.8, 'bbox': a.get('bbox', [])})

        rule_errors = rule_result.data.get('errors', []) if rule_result.success else []
        llm_errors = []
        llm_summary = ""
        llm_score = None
        llm_learning_points = []
        api_result = None

        if llm_result.success:
            api_result = {
                'raw_response': llm_result.data.get('raw_response', ''),
                'model': llm_result.data.get('model', ''),
                'usage': llm_result.data.get('usage', None)
            }
            try:
                raw = llm_result.data.get('raw_response', '')
                start_idx = raw.find('{')
                end_idx = raw.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    llm_data = json.loads(raw[start_idx:end_idx])
                    llm_errors = llm_data.get('errors', [])
                    llm_summary = llm_data.get('summary', '')
                    llm_score = llm_data.get('overall_score', None)
                    llm_learning_points = llm_data.get('learning_points', [])
            except Exception:
                pass

        all_errors = list(rule_errors)
        existing_descs = {e.get('description', '') for e in all_errors}
        for le in llm_errors:
            desc = le.get('description', '')
            if desc and desc not in existing_descs:
                all_errors.append({
                    'type': le.get('type', 'LLM检测'),
                    'description': desc,
                    'suggestion': le.get('suggestion', ''),
                    'severity': le.get('severity', '中'),
                    'source': 'llm_analysis',
                    'gb_reference': le.get('gb_reference', '')
                })
                existing_descs.add(desc)

        if rl_params is None:
            from rl_memory_unit import PolicyParameters
            rl_params = PolicyParameters()
        severity_weights = {
            '高': rl_params.severity_weight_high,
            '中': rl_params.severity_weight_medium,
            '低': rl_params.severity_weight_low
        }
        weighted = sum(severity_weights.get(e.get('severity', '中'), rl_params.severity_weight_medium) for e in all_errors)
        base_score = max(0, 100 - weighted * rl_params.score_penalty_per_weight)
        fusion_ratio = rl_params.llm_score_fusion_ratio
        overall_score = int(base_score * (1 - fusion_ratio) + llm_score * fusion_ratio) if llm_score is not None else base_score

        error_categories = {}
        for e in all_errors:
            cat = e.get('type', '其他')
            error_categories[cat] = error_categories.get(cat, 0) + 1

        feedback = []
        if llm_learning_points:
            feedback.extend(llm_learning_points)
        else:
            for e in all_errors[:5]:
                desc = e.get('description', '')
                etype = e.get('type', '')
                if '尺寸' in etype:
                    feedback.append(f'关于"{desc}"——哪些关键尺寸是必须标注的？')
                elif '线型' in etype:
                    feedback.append(f'关于"{desc}"——你能区分不同线型的含义吗？')
                elif '公差' in etype:
                    feedback.append(f'关于"{desc}"——如何选择合适的公差等级？')
                elif '标题栏' in etype:
                    feedback.append(f'关于"{desc}"——标题栏应包含哪些必要信息？')
                else:
                    feedback.append(f'关于"{desc}"——请思考如何修正这个问题。')
        if not feedback:
            feedback.append('请仔细检查图纸细节，确保符合GB/T制图标准。')

        return {
            'ocr_results': ocr_texts,
            'detection_results': detection_items,
            'errors': all_errors,
            'feedback': feedback,
            'api_result': api_result,
            'geo_result': geometry_result.data if geometry_result.success else None,
            'structure_result': structure_result.data if structure_result.success else None,
            'report': {
                'total_errors': len(all_errors),
                'error_categories': error_categories,
                'overall_score': overall_score,
                'summary': llm_summary or f"共检测到{len(all_errors)}个问题需要关注"
            }
        }
