"""
结构分析Agent
=============
分析图纸整体结构：图框、标题栏、视图区域。
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

logger = logging.getLogger("StructureAgent")


class StructureAgent(BaseAgent):
    """图纸结构分析Agent"""

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
            img = imread_chinese(image_path)
            if img is None:
                return AgentResult("Structure", False, {}, [f"Cannot read: {image_path}"], confidence=0.0)
            cache.put(image_path, img)

        h, w = img.shape[:2]
        gray = PreprocessPipeline.run(img, "structure")

        # 检测图框
        has_border = self._detect_border(gray, w, h)

        # 检测标题栏
        title_block = self._detect_title_block(gray, w, h)

        # 分割视图区域
        view_areas = self._detect_view_areas(gray, w, h)

        # 6区域分割
        regions = self._segment_regions(w, h, title_block)

        confidence = 0.7
        if has_border:
            confidence += 0.1
        if title_block.get('detected'):
            confidence += 0.2

        return AgentResult("Structure", True, {
            'image_size': {'width': w, 'height': h},
            'has_border': has_border,
            'title_block': title_block,
            'view_areas': view_areas,
            'regions': regions
        }, confidence=min(1.0, confidence))

    def _detect_border(self, gray, w, h):
        """检测图框线"""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=max(w, h) * 0.3, maxLineGap=10)

        if lines is None:
            return False

        # 检查四条边是否都有线段
        has_top = has_bottom = has_left = has_right = False
        margin = 20

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if y1 < margin and y2 < margin:
                has_top = True
            elif y1 > h - margin and y2 > h - margin:
                has_bottom = True
            elif x1 < margin and x2 < margin:
                has_left = True
            elif x1 > w - margin and x2 > w - margin:
                has_right = True

        return sum([has_top, has_bottom, has_left, has_right]) >= 3

    def _detect_title_block(self, gray, w, h):
        """检测标题栏区域"""
        # 标题栏通常在右下角
        title_region = gray[int(h * 0.85):h, int(w * 0.5):w]

        if title_region.size == 0:
            return {'detected': False, 'grid_cells': 0, 'bbox': []}

        # 使用网格检测
        edges = cv2.Canny(title_region, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                                minLineLength=20, maxLineGap=5)

        # 统计水平和垂直线段
        h_lines = 0
        v_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 10 or abs(angle) > 170:
                    h_lines += 1
                elif 80 < abs(angle) < 100:
                    v_lines += 1

        # 网格单元数 = (水平线+1) * (垂直线+1)
        grid_cells = max(0, (h_lines + 1) * (v_lines + 1))
        detected = grid_cells >= 4  # 至少2x2网格

        bbox = [int(w * 0.5), int(h * 0.85), w, h] if detected else []

        return {
            'detected': detected,
            'grid_cells': grid_cells,
            'bbox': bbox,
            'h_lines': h_lines,
            'v_lines': v_lines
        }

    def _detect_view_areas(self, gray, w, h):
        """检测视图区域"""
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        view_areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (w * h) * 0.05:  # 至少占图纸面积的5%
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(ch, 1)
                if 0.3 < aspect < 3.0:  # 宽高比合理
                    view_areas.append({
                        'bbox': [int(x), int(y), int(x + cw), int(y + ch)],
                        'area': float(area),
                        'aspect_ratio': float(aspect)
                    })

        # 按面积降序排列
        view_areas.sort(key=lambda v: v['area'], reverse=True)
        return view_areas[:6]

    def _segment_regions(self, w, h, title_block):
        """6区域分割"""
        regions = []

        # 标题栏区域
        if title_block.get('detected'):
            bbox = title_block.get('bbox', [])
            if bbox:
                regions.append({
                    'name': '标题栏区域',
                    'x': bbox[0], 'y': bbox[1],
                    'w': bbox[2] - bbox[0], 'h': bbox[3] - bbox[1]
                })

        # 主视图区域（标题栏上方）
        title_top = title_block.get('bbox', [0, int(h * 0.85)])[1] if title_block.get('detected') else int(h * 0.85)
        regions.append({
            'name': '主视图区域',
            'x': 0, 'y': 0,
            'w': w, 'h': title_top
        })

        return regions
