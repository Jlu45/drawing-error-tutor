"""
几何检测Agent
=============
基于OpenCV的几何元素检测Agent。
检测5类几何元素：直线、圆、箭头、轮廓、线型。
"""

import os
import sys
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional

_v2_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _v2_root not in sys.path:
    sys.path.insert(0, _v2_root)

from utils.image_utils import imread_chinese

from agents.base import BaseAgent, AgentResult
from agents.ocr_agent import ImageCache, PreprocessPipeline

logger = logging.getLogger("GeometryAgent")


class GeometryAgent(BaseAgent):
    """几何元素检测Agent"""

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
            img = imread_chinese(image_path)
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

        # 支持通过kwargs调整检测参数
        canny_low = kwargs.get('canny_low', 50)
        canny_high = kwargs.get('canny_high', 150)
        hough_threshold = kwargs.get('hough_threshold', 80)

        gray = PreprocessPipeline.run(work_img, "geometry")

        # 预计算边缘和线段（避免重复Canny+HoughLinesP）
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        all_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
                                    minLineLength=30, maxLineGap=10)

        result = {
            'lines': self._format_lines(all_lines),
            'circles': self._detect_circles(gray),
            'arrows': self._detect_arrows_from_edges(edges),
            'contours': self._detect_contours_from_edges(edges),
            'line_types': self._classify_line_types_fast(gray, all_lines),
            'dimension_structures': self._detect_dimension_structures_fast(all_lines)
        }

        total_elements = len(result['lines']) + len(result['circles']) + len(result['arrows'])
        confidence = min(1.0, total_elements / 20.0)

        return AgentResult("Geometry", True, result, confidence=confidence)

    def _format_lines(self, lines):
        """格式化线段检测结果"""
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

    def _detect_arrows_from_edges(self, edges):
        """从预计算的边缘检测箭头"""
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 500:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if 3 <= len(approx) <= 5:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = w / max(h, 1)
                    if 0.3 < aspect < 3.0:
                        arrows.append({
                            'bbox': [int(x), int(y), int(x + w), int(y + h)],
                            'area': float(area),
                            'vertices': len(approx)
                        })
        return arrows[:50]

    def _detect_contours_from_edges(self, edges):
        """从预计算的边缘检测轮廓"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detected.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'area': float(area),
                    'perimeter': float(cv2.arcLength(cnt, True))
                })
        return detected[:30]

    def _classify_line_types_fast(self, gray, lines):
        """线型分类（复用已有线段结果）"""
        solid_count = 0
        dashed_count = 0
        center_line_count = 0

        if lines is not None:
            max_lines = min(len(lines), 200)
            for line_idx in range(max_lines):
                line = lines[line_idx]
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 10:
                    continue

                num_samples = min(int(length), 30)
                samples = []
                for i in range(num_samples):
                    t = i / max(num_samples - 1, 1)
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    if 0 <= py < gray.shape[0] and 0 <= px < gray.shape[1]:
                        samples.append(gray[py, px])

                if len(samples) < 5:
                    continue

                threshold = 128
                binary_samples = [1 if s < threshold else 0 for s in samples]
                transitions = sum(1 for i in range(1, len(binary_samples))
                                  if binary_samples[i] != binary_samples[i-1])
                transition_rate = transitions / len(samples)

                if transition_rate < 0.05:
                    solid_count += 1
                elif transition_rate < 0.2:
                    dashed_count += 1
                else:
                    center_line_count += 1

        return {
            'solid_count': solid_count,
            'dashed_count': dashed_count,
            'center_line_count': center_line_count,
            'total_lines': solid_count + dashed_count + center_line_count
        }

    def _detect_dimension_structures_fast(self, lines):
        """检测尺寸线结构（复用已有线段结果）"""
        dimension_pairs = []
        if lines is not None and len(lines) >= 2:
            max_check = min(len(lines), 30)
            for i in range(max_check):
                for j in range(i + 1, max_check):
                    l1, l2 = lines[i][0], lines[j][0]
                    angle1 = np.degrees(np.arctan2(l1[3] - l1[1], l1[2] - l1[0]))
                    angle2 = np.degrees(np.arctan2(l2[3] - l2[1], l2[2] - l2[0]))
                    if abs(angle1 - angle2) < 15:
                        mid1 = ((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2)
                        mid2 = ((l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2)
                        dist = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
                        if 10 < dist < 200:
                            dimension_pairs.append({
                                'line1': {'start': (int(l1[0]), int(l1[1])),
                                          'end': (int(l1[2]), int(l1[3]))},
                                'line2': {'start': (int(l2[0]), int(l2[1])),
                                          'end': (int(l2[2]), int(l2[3]))},
                                'distance': float(dist),
                                'angle_diff': abs(angle1 - angle2)
                            })
        return dimension_pairs[:20]

    def _detect_lines(self, gray, canny_low=50, canny_high=150, hough_threshold=80):
        edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_threshold,
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
                                    minRadius=5, maxRadius=500)
        detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0]:
                detected.append({
                    'center': (int(c[0]), int(c[1])),
                    'radius': int(c[2]),
                    'is_large': int(c[2]) > 50
                })
        return detected

    def _detect_arrows(self, gray, work_img):
        """检测箭头（尺寸标注终端）"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 500:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if 3 <= len(approx) <= 5:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = w / max(h, 1)
                    if 0.3 < aspect < 3.0:
                        arrows.append({
                            'bbox': [int(x), int(y), int(x + w), int(y + h)],
                            'area': float(area),
                            'vertices': len(approx)
                        })
        return arrows[:50]  # 限制数量

    def _detect_contours(self, gray):
        """检测轮廓"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                detected.append({
                    'bbox': [int(x), int(y), int(x + w), int(y + h)],
                    'area': float(area),
                    'perimeter': float(cv2.arcLength(cnt, True))
                })
        return detected[:30]

    def _classify_line_types(self, gray):
        """线型分类（实线/虚线/点画线）"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=30, maxLineGap=10)

        solid_count = 0
        dashed_count = 0
        center_line_count = 0

        if lines is not None:
            max_lines = min(len(lines), 200)
            for line_idx in range(max_lines):
                line = lines[line_idx]
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < 10:
                    continue

                num_samples = min(int(length), 30)
                samples = []
                for i in range(num_samples):
                    t = i / max(num_samples - 1, 1)
                    px = int(x1 + t * (x2 - x1))
                    py = int(y1 + t * (y2 - y1))
                    if 0 <= py < gray.shape[0] and 0 <= px < gray.shape[1]:
                        samples.append(gray[py, px])

                if len(samples) < 5:
                    continue

                threshold = 128
                binary_samples = [1 if s < threshold else 0 for s in samples]
                transitions = sum(1 for i in range(1, len(binary_samples))
                                  if binary_samples[i] != binary_samples[i-1])
                transition_rate = transitions / len(samples)

                if transition_rate < 0.05:
                    solid_count += 1
                elif transition_rate < 0.2:
                    dashed_count += 1
                else:
                    center_line_count += 1

        return {
            'solid_count': solid_count,
            'dashed_count': dashed_count,
            'center_line_count': center_line_count,
            'total_lines': solid_count + dashed_count + center_line_count
        }

    def _detect_dimension_structures(self, gray):
        """检测尺寸线结构（平行线对 + 箭头）"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                                minLineLength=30, maxLineGap=10)

        dimension_pairs = []
        if lines is not None and len(lines) >= 2:
            max_check = min(len(lines), 30)
            for i in range(max_check):
                for j in range(i + 1, max_check):
                    l1, l2 = lines[i][0], lines[j][0]
                    # 检查是否为近似平行的短线对
                    angle1 = np.degrees(np.arctan2(l1[3] - l1[1], l1[2] - l1[0]))
                    angle2 = np.degrees(np.arctan2(l2[3] - l2[1], l2[2] - l2[0]))
                    if abs(angle1 - angle2) < 15:
                        # 计算距离
                        mid1 = ((l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2)
                        mid2 = ((l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2)
                        dist = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
                        if 10 < dist < 200:
                            dimension_pairs.append({
                                'line1': {'start': (int(l1[0]), int(l1[1])),
                                          'end': (int(l1[2]), int(l1[3]))},
                                'line2': {'start': (int(l2[0]), int(l2[1])),
                                          'end': (int(l2[2]), int(l2[3]))},
                                'distance': float(dist),
                                'angle_diff': abs(angle1 - angle2)
                            })

        return dimension_pairs[:20]
