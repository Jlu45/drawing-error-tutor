"""
Atlas 特征适配器 (Phase 1)
============================
将 V2 Agent 输出（OCR/Geometry/Structure）转换为 atlas 规则引擎
所需的统一特征字典。
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("AtlasFeatureAdapter")


class AtlasFeatureAdapter:
    """Phase 1 特征适配：V2 Agent 输出 → atlas 特征字典"""

    def build(
        self,
        ocr_result: Dict,
        geometry_result: Dict,
        structure_result: Dict,
        image_shape: Tuple[int, int],
    ) -> Dict:
        try:
            ocr_patterns = self._extract_ocr_patterns(ocr_result)
            line_candidates = self._extract_line_candidates(geometry_result)
            geometry_groups = self._group_geometry(geometry_result)
            regions = self._extract_regions(structure_result, image_shape)
            all_text = " ".join(
                t.get("text", "")
                for t in (ocr_result.get("texts") or [])
            )
            ocr_patterns["all_text"] = all_text
            features = {
                "ocr_patterns": ocr_patterns,
                "line_candidates": line_candidates,
                "geometry_groups": geometry_groups,
                "regions": regions,
                "raw_refs": {
                    "ocr": ocr_result,
                    "geometry": geometry_result,
                    "structure": structure_result,
                },
            }
            logger.info(
                f"[AtlasFeatureAdapter] 特征构建完成: "
                f"diameter={len(ocr_patterns.get('diameter_texts', []))}, "
                f"center_lines={len(line_candidates.get('center_lines', []))}, "
                f"hole_groups={len(geometry_groups.get('hole_groups', []))}"
            )
            return features
        except Exception as e:
            logger.error(f"[AtlasFeatureAdapter] 特征构建异常: {e}")
            return {
                "ocr_patterns": {"all_text": ""},
                "line_candidates": {},
                "geometry_groups": {},
                "regions": {},
                "raw_refs": {
                    "ocr": ocr_result or {},
                    "geometry": geometry_result or {},
                    "structure": structure_result or {},
                },
            }

    def _extract_ocr_patterns(self, ocr_result: Dict) -> Dict:
        diameter_texts = []
        radius_texts = []
        chamfer_texts = []
        scale_texts = []
        roughness_texts = []
        eqs_texts = []
        count_texts = []
        roman_labels = []
        view_labels = []
        texts = ocr_result.get("texts") or []
        diameter_re = re.compile(r"[Φφ⌀]\s*\d+\.?\d*")
        radius_re = re.compile(r"[Rr]\s*\d+\.?\d*")
        chamfer_re = re.compile(r"C\d+\.?\d*|\d+\s*[×x]\s*45\s*[°˚]")
        scale_re = re.compile(r"\d+\s*[:：]\s*\d+")
        roughness_re = re.compile(r"Ra|其余|全部")
        eqs_re = re.compile(r"EQS", re.IGNORECASE)
        count_re = re.compile(r"\d+\s*[×xX]\s*[Φφ⌀Rr]\s*\d+")
        roman_re = re.compile(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]")
        view_label_re = re.compile(r"\b[A-Z]\b")
        for item in texts:
            text = item.get("text", "")
            if not text:
                continue
            entry = {
                "text": text,
                "confidence": item.get("confidence", 0.0),
                "bbox": item.get("bbox", []),
            }
            if diameter_re.search(text):
                diameter_texts.append(entry)
            if radius_re.search(text):
                radius_texts.append(entry)
            if chamfer_re.search(text):
                chamfer_texts.append(entry)
            if scale_re.search(text):
                scale_texts.append(entry)
            if roughness_re.search(text):
                roughness_texts.append(entry)
            if eqs_re.search(text):
                eqs_texts.append(entry)
            if count_re.search(text):
                count_texts.append(entry)
            if roman_re.search(text):
                roman_labels.append(entry)
            if view_label_re.search(text) and len(text.strip()) == 1:
                view_labels.append(entry)
        return {
            "diameter_texts": diameter_texts,
            "radius_texts": radius_texts,
            "chamfer_texts": chamfer_texts,
            "scale_texts": scale_texts,
            "roughness_texts": roughness_texts,
            "eqs_texts": eqs_texts,
            "count_texts": count_texts,
            "roman_labels": roman_labels,
            "view_labels": view_labels,
        }

    def _extract_line_candidates(self, geometry_result: Dict) -> Dict:
        center_lines = []
        coarse_dash_lines = []
        coarse_chain_lines = []
        wavy_lines = []
        line_types = geometry_result.get("line_types", {})
        if isinstance(line_types, dict):
            center_line_count = line_types.get("center_line_count", 0)
            if center_line_count > 0:
                lines = geometry_result.get("lines", [])
                for line in lines:
                    start = line.get("start", (0, 0))
                    end = line.get("end", (0, 0))
                    length = line.get("length", 0)
                    angle = line.get("angle", 0)
                    is_h = line.get("is_horizontal", False)
                    is_v = line.get("is_vertical", False)
                    center_lines.append({
                        "start": start,
                        "end": end,
                        "length": length,
                        "angle": angle,
                        "is_horizontal": is_h,
                        "is_vertical": is_v,
                    })
        contours = geometry_result.get("contours", [])
        for cnt in contours:
            bbox = cnt.get("bbox", [0, 0, 0, 0])
            perimeter = cnt.get("perimeter", 0)
            area = cnt.get("area", 0)
            if perimeter > 0 and area > 0:
                circularity = 4 * 3.14159 * area / (perimeter * perimeter)
                if circularity < 0.3:
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if w > 0 and 0.1 < h / w < 10:
                        wavy_lines.append({
                            "bbox": bbox,
                            "perimeter": perimeter,
                            "area": area,
                            "circularity": circularity,
                        })
        return {
            "center_lines": center_lines,
            "coarse_dash_lines": coarse_dash_lines,
            "coarse_chain_lines": coarse_chain_lines,
            "wavy_lines": wavy_lines,
        }

    def _group_geometry(self, geometry_result: Dict) -> Dict:
        circles = geometry_result.get("circles", [])
        if not circles:
            return {"hole_groups": []}
        valid_circles = []
        for c in circles:
            center = c.get("center", (0, 0))
            radius = c.get("radius", 0)
            if radius > 0:
                valid_circles.append({
                    "center": center,
                    "radius": radius,
                    "is_large": c.get("is_large", False),
                })
        if not valid_circles:
            return {"hole_groups": []}
        small_circles = [c for c in valid_circles if not c["is_large"]]
        if not small_circles:
            return {"hole_groups": []}
        radius_groups = self._cluster_by_radius(small_circles, tolerance_ratio=0.15)
        hole_groups = []
        for group in radius_groups:
            if len(group) < 2:
                continue
            avg_radius = np.mean([c["radius"] for c in group])
            centers = [c["center"] for c in group]
            spatial_clusters = self._cluster_by_position(centers, max_dist_factor=3.0)
            for cluster in spatial_clusters:
                if len(cluster) >= 2:
                    cluster_circles = [group[i] for i in cluster]
                    hole_groups.append({
                        "circles": cluster_circles,
                        "count": len(cluster_circles),
                        "avg_radius": float(avg_radius),
                        "centers": [c["center"] for c in cluster_circles],
                    })
        return {"hole_groups": hole_groups}

    def _cluster_by_radius(
        self, circles: List[Dict], tolerance_ratio: float = 0.15
    ) -> List[List[Dict]]:
        if not circles:
            return []
        sorted_circles = sorted(circles, key=lambda c: c["radius"])
        groups = []
        current_group = [sorted_circles[0]]
        for c in sorted_circles[1:]:
            ref_radius = current_group[0]["radius"]
            if ref_radius > 0 and abs(c["radius"] - ref_radius) / ref_radius <= tolerance_ratio:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]
        groups.append(current_group)
        return groups

    def _cluster_by_position(
        self, centers: List[Tuple], max_dist_factor: float = 3.0
    ) -> List[List[int]]:
        n = len(centers)
        if n == 0:
            return []
        if n == 1:
            return [[0]]
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                distances.append(dist)
        if not distances:
            return [list(range(n))]
        avg_dist = np.mean(distances)
        threshold = avg_dist * max_dist_factor
        visited = [False] * n
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if visited[j]:
                        continue
                    dx = centers[current][0] - centers[j][0]
                    dy = centers[current][1] - centers[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist <= threshold:
                        cluster.append(j)
                        visited[j] = True
                        queue.append(j)
            clusters.append(cluster)
        return clusters

    def _extract_regions(
        self, structure_result: Dict, image_shape: Tuple[int, int]
    ) -> Dict:
        h, w = image_shape[:2]
        top_right = [int(w * 0.7), 0, w, int(h * 0.3)]
        title_block_region = [0, 0, 0, 0]
        try:
            title_block = structure_result.get("title_block", {})
            if isinstance(title_block, dict) and title_block.get("detected"):
                bbox = title_block.get("bbox")
                if bbox and len(bbox) == 4:
                    title_block_region = bbox
                else:
                    title_block_region = [int(w * 0.55), int(h * 0.85), w, h]
            else:
                title_block_region = [int(w * 0.55), int(h * 0.85), w, h]
        except Exception:
            title_block_region = [int(w * 0.55), int(h * 0.85), w, h]
        return {
            "top_right": top_right,
            "title_block": title_block_region,
        }
