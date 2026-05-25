"""
Atlas 规则插件 (Phase 2)
=========================
基于图册规则的检测引擎。根据 atlas_features 中的特征，
逐条执行匹配的 atlas 规则，输出 issue 列表。

7 类规则处理器：
- geometry_threshold: 几何阈值（中心线延伸、弯折线等）
- ocr_regex: OCR 正则匹配（倒角、EQS、比例等）
- hole_group: 孔组标注
- roughness: 粗糙度标注
- title_block: 标题栏检查
- view_heuristic: 视图启发式
- context_line_usage: 上下文线型使用
"""

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from atlas.atlas_registry import AtlasRegistry

logger = logging.getLogger("AtlasRulePlugin")

VALID_CHAMFER_PATTERNS = [
    re.compile(r"C\d+\.?\d*"),
    re.compile(r"\d+\.?\d*\s*[×x]\s*45\s*[°˚]"),
]
SUSPECT_CHAMFER_PATTERNS = [
    re.compile(r"\d+\.?\d*\s*[×x]\s*\d+\.?\d*\s*(?!45)[°˚]"),
    re.compile(r"倒角\s*\d+"),
]

SEVERITY_MAP = {
    "P0": "high",
    "P1": "medium",
    "P2": "low",
}


class AtlasRulePlugin:
    """Phase 2 规则插件：执行 atlas 规则检测"""

    def __init__(self, atlas_registry: AtlasRegistry):
        self.registry = atlas_registry
        self.rule_handlers = {
            "geometry_threshold": self._check_geometry_threshold,
            "ocr_regex": self._check_ocr_regex,
            "hole_group": self._check_hole_group,
            "roughness": self._check_roughness,
            "title_block": self._check_title_block,
            "view_heuristic": self._check_view_heuristic,
            "context_line_usage": self._check_context_line_usage,
        }

    def check(
        self,
        atlas_features: Dict,
        contracts: Optional[List] = None,
    ) -> List[Dict]:
        active_rules = self._collect_active_rules(contracts)
        if not active_rules:
            logger.info("[AtlasRulePlugin] 无活跃规则，跳过检测")
            return []
        all_issues = []
        max_issues_per_rule = 3
        max_total_issues = 30
        for rule in active_rules:
            try:
                check_type = rule.get("check_type", "")
                handler = self.rule_handlers.get(check_type)
                if handler is None:
                    logger.debug(
                        f"[AtlasRulePlugin] 无处理器: {check_type} "
                        f"(rule={rule.get('rule_id', '?')})"
                    )
                    continue
                issues = handler(atlas_features, rule)
                if len(issues) > max_issues_per_rule:
                    logger.warning(
                        f"[AtlasRulePlugin] 规则 {rule.get('rule_id', '?')} "
                        f"产生 {len(issues)} 个问题，截断为 {max_issues_per_rule}"
                    )
                    issues = issues[:max_issues_per_rule]
                all_issues.extend(issues)
                if len(all_issues) >= max_total_issues:
                    logger.warning(
                        f"[AtlasRulePlugin] 总问题数达到 {max_total_issues} 上限，停止检测"
                    )
                    all_issues = all_issues[:max_total_issues]
                    break
            except Exception as e:
                logger.error(
                    f"[AtlasRulePlugin] 规则 {rule.get('rule_id', '?')} "
                    f"执行异常: {e}"
                )
        logger.info(
            f"[AtlasRulePlugin] 检测完成: {len(active_rules)} 条规则, "
            f"{len(all_issues)} 个问题"
        )
        return all_issues

    def _collect_active_rules(self, contracts: Optional[List] = None) -> List[Dict]:
        if contracts:
            rule_ids = set()
            for contract in contracts:
                try:
                    metadata = getattr(contract, "metadata", None)
                    if not isinstance(metadata, dict):
                        continue
                    subchecks = metadata.get("atlas_subchecks", [])
                    for sc in subchecks:
                        rid = sc.get("rule_id", "")
                        if rid:
                            rule_ids.add(rid)
                except Exception:
                    continue
            if rule_ids:
                rules = []
                for rid in rule_ids:
                    rule = self.registry.get_rule(rid)
                    if rule and rule.get("enabled", False):
                        rules.append(rule)
                return rules
        return self.registry.get_active_rules()

    def _make_issue(
        self,
        rule: Dict,
        level: str,
        confidence: float,
        title: str,
        description: str,
        evidence: Dict,
        source: Optional[List[str]] = None,
    ) -> Dict:
        priority = rule.get("priority", "P2")
        severity = SEVERITY_MAP.get(priority, "low")
        return {
            "error_category": rule.get("v2_error_category", "GENERAL_ERROR"),
            "atlas_rule_id": rule.get("rule_id", ""),
            "source_case_id": rule.get("source_case_id", ""),
            "level": level,
            "severity": severity,
            "title": title,
            "description": description,
            "evidence": evidence,
            "suggestion": rule.get("suggestion", ""),
            "teaching_hint": rule.get("teaching_hint", ""),
            "confidence": round(confidence, 2),
            "source": source or ["atlas_rule"],
        }

    def _resolve_level(self, rule: Dict, confidence: float) -> str:
        output_policy = rule.get("output_policy", {})
        confirmed_thresh = rule.get("params", {}).get("confidence_confirmed", 0.80)
        suspected_thresh = rule.get("params", {}).get("confidence_suspected", 0.50)
        if confidence >= confirmed_thresh:
            return output_policy.get("high_confidence", "confirmed_error")
        if confidence >= suspected_thresh:
            return output_policy.get("medium_confidence", "suspected_issue")
        return output_policy.get("low_confidence", "suggestion")

    # ================================================================
    # geometry_threshold: 中心线延伸、短中心线、弯折线
    # ================================================================
    def _check_geometry_threshold(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "CENTER_EXTEND" in rule_id:
                issues.extend(self._check_center_extend(features, rule))
            elif "CENTER_SHORT" in rule_id:
                issues.extend(self._check_center_short(features, rule))
            elif "BEND" in rule_id:
                issues.extend(self._check_bend_line(features, rule))
            else:
                issues.extend(self._check_generic_geometry_threshold(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] geometry_threshold 异常 ({rule_id}): {e}")
        return issues

    def _check_center_extend(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        center_lines = features.get("line_candidates", {}).get("center_lines", [])
        contours = features.get("raw_refs", {}).get("geometry", {}).get("contours", [])
        if not center_lines or not contours:
            return issues
        contour_bboxes = [c.get("bbox", [0, 0, 0, 0]) for c in contours]
        params = rule.get("params", {})
        min_ratio = params.get("min_extension_ratio", 0.005)
        for cl in center_lines:
            try:
                start = cl.get("start", (0, 0))
                end = cl.get("end", (0, 0))
                line_len = cl.get("length", 0)
                if line_len < 10:
                    continue
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                ux, uy = dx / line_len, dy / line_len
                best_ext_start = 0.0
                best_ext_end = 0.0
                for bbox in contour_bboxes:
                    x1, y1, x2, y2 = bbox
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist_to_center = np.sqrt(
                        (start[0] + dx / 2 - cx) ** 2
                        + (start[1] + dy / 2 - cy) ** 2
                    )
                    diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist_to_center > diag * 1.5:
                        continue
                    for t in np.linspace(0, 1, 20):
                        px = start[0] + t * dx
                        py = start[1] + t * dy
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            ext_s = t * line_len
                            ext_e = (1 - t) * line_len
                            best_ext_start = max(best_ext_start, ext_s)
                            best_ext_end = max(best_ext_end, ext_e)
                            break
                if best_ext_start < line_len * min_ratio and best_ext_end < line_len * min_ratio:
                    confidence = 0.6
                    if best_ext_start == 0 and best_ext_end == 0:
                        confidence = 0.45
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "一般中心线疑似未向可见轮廓外充分延伸",
                            f"检测到中心线(长度={line_len:.0f}px)两端延伸不足，"
                            f"起始端延伸≈{best_ext_start:.1f}px，末端延伸≈{best_ext_end:.1f}px",
                            {
                                "line": cl,
                                "extension_start": round(best_ext_start, 1),
                                "extension_end": round(best_ext_end, 1),
                            },
                            source=["atlas_rule", "geometry"],
                        )
                    )
            except Exception:
                continue
        return issues

    def _check_center_short(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        center_lines = features.get("line_candidates", {}).get("center_lines", [])
        circles = features.get("raw_refs", {}).get("geometry", {}).get("circles", [])
        if not center_lines or not circles:
            return issues
        small_circles = [c for c in circles if not c.get("is_large", False)]
        for sc in small_circles:
            try:
                cx, cy = sc.get("center", (0, 0))
                radius = sc.get("radius", 0)
                nearby_center_lines = []
                for cl in center_lines:
                    mid_x = (cl["start"][0] + cl["end"][0]) / 2
                    mid_y = (cl["start"][1] + cl["end"][1]) / 2
                    dist = np.sqrt((mid_x - cx) ** 2 + (mid_y - cy) ** 2)
                    if dist < radius * 2.5:
                        nearby_center_lines.append(cl)
                for cl in nearby_center_lines:
                    start = cl.get("start", (0, 0))
                    end = cl.get("end", (0, 0))
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    line_len = cl.get("length", 0)
                    if line_len < 1:
                        continue
                    ux, uy = dx / line_len, dy / line_len
                    ext_start = 0.0
                    ext_end = 0.0
                    for t in np.linspace(0, 1, 20):
                        px = start[0] + t * dx
                        py = start[1] + t * dy
                        dist_to_center = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                        if dist_to_center > radius:
                            ext_s = t * line_len
                            ext_e = (1 - t) * line_len
                            ext_start = max(ext_start, ext_s)
                            ext_end = max(ext_end, ext_e)
                            break
                    if ext_start > radius * 0.3 or ext_end > radius * 0.3:
                        confidence = 0.65
                        level = self._resolve_level(rule, confidence)
                        issues.append(
                            self._make_issue(
                                rule,
                                level,
                                confidence,
                                "短中心线疑似超出轮廓线",
                                f"小圆(r={radius}px)附近的短中心线超出轮廓线，"
                                f"起始端超出≈{ext_start:.1f}px，末端超出≈{ext_end:.1f}px",
                                {
                                    "circle": sc,
                                    "line": cl,
                                    "extension_start": round(ext_start, 1),
                                    "extension_end": round(ext_end, 1),
                                },
                                source=["atlas_rule", "geometry"],
                            )
                        )
            except Exception:
                continue
        return issues

    def _check_bend_line(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        has_bend_keyword = bool(re.search(r"展开|弯折|折弯", ocr_all))
        if not has_bend_keyword:
            return issues
        lines = features.get("raw_refs", {}).get("geometry", {}).get("lines", [])
        contours = features.get("raw_refs", {}).get("geometry", {}).get("contours", [])
        thin_solid_candidates = []
        for line in lines:
            length = line.get("length", 0)
            if length < 20:
                continue
            angle = abs(line.get("angle", 0))
            is_h = line.get("is_horizontal", False)
            is_v = line.get("is_vertical", False)
            if is_h or is_v:
                thin_solid_candidates.append(line)
        if not thin_solid_candidates:
            confidence = 0.55
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "弯折线画法疑似错误",
                    "OCR检测到弯折相关文字，但未找到对应的弯折线（细实线）",
                    {"bend_keyword_found": True, "thin_solid_count": 0},
                    source=["atlas_rule", "ocr", "geometry"],
                )
            )
        else:
            contour_bboxes = [c.get("bbox", [0, 0, 0, 0]) for c in contours]
            for candidate in thin_solid_candidates:
                start = candidate.get("start", (0, 0))
                end = candidate.get("end", (0, 0))
                intersects = False
                for bbox in contour_bboxes:
                    x1, y1, x2, y2 = bbox
                    for t in np.linspace(0, 1, 20):
                        px = start[0] + t * (end[0] - start[0])
                        py = start[1] + t * (end[1] - start[1])
                        if x1 <= px <= x2 and y1 <= py <= y2:
                            intersects = True
                            break
                    if intersects:
                        break
                if intersects:
                    confidence = 0.6
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "弯折线画法疑似错误",
                            "检测到疑似弯折线与轮廓相交，弯折线应超出轮廓线2mm～5mm",
                            {"line": candidate, "intersects_contour": True},
                            source=["atlas_rule", "geometry"],
                        )
                    )
        return issues

    def _check_generic_geometry_threshold(self, features: Dict, rule: Dict) -> List[Dict]:
        return []

    # ================================================================
    # ocr_regex: 倒角、EQS、局部放大图比例、腰形孔、参考尺寸、
    #            尺寸数字位置、角度标注、公差框格、尺寸公差
    # ================================================================
    def _check_ocr_regex(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "CHAMFER" in rule_id:
                issues.extend(self._check_chamfer(features, rule))
            elif "EQS" in rule_id:
                issues.extend(self._check_eqs(features, rule))
            elif "DETAIL_RATIO" in rule_id or "VIEW_DETAIL" in rule_id:
                issues.extend(self._check_detail_ratio(features, rule))
            elif "SLOT" in rule_id:
                issues.extend(self._check_slot(features, rule))
            elif "REFERENCE" in rule_id:
                issues.extend(self._check_reference(features, rule))
            elif "NUMBER_POS" in rule_id:
                issues.extend(self._check_number_pos(features, rule))
            elif "ANGLE" in rule_id:
                issues.extend(self._check_angle(features, rule))
            elif "TOLERANCE_FRAME" in rule_id:
                issues.extend(self._check_tolerance_frame(features, rule))
            elif "DIM_TOLERANCE" in rule_id:
                issues.extend(self._check_dim_tolerance(features, rule))
            else:
                issues.extend(self._check_generic_ocr_regex(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] ocr_regex 异常 ({rule_id}): {e}")
        return issues

    def _check_chamfer(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        chamfer_texts = features.get("ocr_patterns", {}).get("chamfer_texts", [])
        has_valid = False
        has_suspect = False
        for ct in chamfer_texts:
            text = ct.get("text", "")
            for pat in VALID_CHAMFER_PATTERNS:
                if pat.search(text):
                    has_valid = True
                    break
            for pat in SUSPECT_CHAMFER_PATTERNS:
                if pat.search(text):
                    has_suspect = True
                    break
        if has_suspect and not has_valid:
            confidence = 0.75
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "倒角标注格式疑似错误",
                    "检测到非标准倒角标注格式，45°倒角应使用C加宽度（如C2）标注",
                    {"chamfer_texts": chamfer_texts, "has_valid": has_valid, "has_suspect": has_suspect},
                    source=["atlas_rule", "ocr"],
                )
            )
        diameter_texts = features.get("ocr_patterns", {}).get("diameter_texts", [])
        radius_texts = features.get("ocr_patterns", {}).get("radius_texts", [])
        all_dim_texts = diameter_texts + radius_texts
        for dt in all_dim_texts:
            text = dt.get("text", "")
            if re.search(r"\d+\.?\d*\s*[°˚]", text) and not re.search(r"C\d+", text):
                if not re.search(r"\d+[°˚]\s*[×x]\s*\d+", text):
                    confidence = 0.55
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "倒角标注格式疑似错误",
                            f"检测到角度值'{text}'但缺少标准倒角标注格式",
                            {"text": text},
                            source=["atlas_rule", "ocr"],
                        )
                    )
        return issues

    def _check_eqs(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        eqs_texts = features.get("ocr_patterns", {}).get("eqs_texts", [])
        has_eqs = len(eqs_texts) > 0
        hole_groups = features.get("geometry_groups", {}).get("hole_groups", [])
        for group in hole_groups:
            count = group.get("count", 0)
            if count < 3:
                continue
            centers = group.get("centers", [])
            if len(centers) < 3:
                continue
            is_uniform = self._check_uniform_distribution(centers)
            if is_uniform and not has_eqs:
                confidence = 0.65
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "均布孔缺少EQS标注",
                        f"检测到{count}个均匀分布的孔，但未发现EQS标注",
                        {"hole_count": count, "avg_radius": group.get("avg_radius", 0)},
                        source=["atlas_rule", "ocr", "geometry"],
                    )
                )
        return issues

    def _check_uniform_distribution(self, centers: List) -> bool:
        if len(centers) < 3:
            return False
        try:
            cx = np.mean([c[0] for c in centers])
            cy = np.mean([c[1] for c in centers])
            angles = []
            for c in centers:
                angle = np.degrees(np.arctan2(c[1] - cy, c[0] - cx))
                angles.append(angle % 360)
            angles.sort()
            n = len(angles)
            expected_gap = 360.0 / n
            gaps = []
            for i in range(n):
                next_i = (i + 1) % n
                gap = (angles[next_i] - angles[i]) % 360
                gaps.append(gap)
            if not gaps:
                return False
            avg_gap = np.mean(gaps)
            max_deviation = max(abs(g - avg_gap) for g in gaps)
            return max_deviation < expected_gap * 0.3
        except Exception:
            return False

    def _check_detail_ratio(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        roman_labels = features.get("ocr_patterns", {}).get("roman_labels", [])
        scale_texts = features.get("ocr_patterns", {}).get("scale_texts", [])
        if not roman_labels:
            return issues
        for rl in roman_labels:
            label_text = rl.get("text", "")
            label_bbox = rl.get("bbox", [])
            nearby_scale = None
            for st in scale_texts:
                scale_bbox = st.get("bbox", [])
                if not label_bbox or not scale_bbox:
                    continue
                lx = (label_bbox[0] + label_bbox[2]) / 2 if len(label_bbox) >= 4 else 0
                ly = (label_bbox[1] + label_bbox[3]) / 2 if len(label_bbox) >= 4 else 0
                sx = (scale_bbox[0] + scale_bbox[2]) / 2 if len(scale_bbox) >= 4 else 0
                sy = (scale_bbox[1] + scale_bbox[3]) / 2 if len(scale_bbox) >= 4 else 0
                dist = np.sqrt((lx - sx) ** 2 + (ly - sy) ** 2)
                if dist < 200:
                    nearby_scale = st
                    break
            if nearby_scale is None:
                confidence = 0.6
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "局部放大图比例标注疑似缺失",
                        f"检测到罗马数字标记'{label_text}'，但附近未发现比例标注",
                        {"roman_label": label_text, "nearby_scale": None},
                        source=["atlas_rule", "ocr"],
                    )
                )
        return issues

    def _check_slot(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        radius_texts = features.get("ocr_patterns", {}).get("radius_texts", [])
        circles = features.get("raw_refs", {}).get("geometry", {}).get("circles", [])
        if not radius_texts or not circles:
            return issues
        slot_candidates = self._detect_slot_shapes(circles)
        for slot in slot_candidates:
            center1 = slot["circle1"].get("center", [0, 0])
            center2 = slot["circle2"].get("center", [0, 0])
            c1x = float(center1[0]) if not isinstance(center1[0], (list, tuple)) else float(center1[0][0])
            c1y = float(center1[1]) if not isinstance(center1[1], (list, tuple)) else float(center1[1][0])
            c2x = float(center2[0]) if not isinstance(center2[0], (list, tuple)) else float(center2[0][0])
            c2y = float(center2[1]) if not isinstance(center2[1], (list, tuple)) else float(center2[1][0])
            mid_x = (c1x + c2x) / 2
            mid_y = (c1y + c2y) / 2
            has_nearby_r = False
            for rt in radius_texts:
                bbox = rt.get("bbox", [])
                if len(bbox) >= 4:
                    tx = (bbox[0] + bbox[2]) / 2
                    ty = (bbox[1] + bbox[3]) / 2
                    dist = np.sqrt((tx - mid_x) ** 2 + (ty - mid_y) ** 2)
                    if dist < slot["circle1"].get("radius", 0) * 5:
                        has_nearby_r = True
                        break
            if has_nearby_r:
                confidence = 0.65
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "腰形孔标注方式疑似错误",
                        "检测到疑似腰形孔，附近发现R标注，腰形孔应标注两圆弧中心距和宽度",
                        {"slot": slot},
                        source=["atlas_rule", "ocr", "geometry"],
                    )
                )
        return issues

    def _detect_slot_shapes(self, circles: List[Dict]) -> List[Dict]:
        slots = []
        small_circles = [c for c in circles if not c.get("is_large", False)]
        for i, c1 in enumerate(small_circles):
            for j, c2 in enumerate(small_circles):
                if j <= i:
                    continue
                r1, r2 = c1.get("radius", 0), c2.get("radius", 0)
                if r1 <= 0 or r2 <= 0:
                    continue
                if abs(r1 - r2) / max(r1, r2) > 0.2:
                    continue
                center1 = c1.get("center", (0, 0))
                center2 = c2.get("center", (0, 0))
                c1x = float(center1[0]) if not isinstance(center1[0], (list, tuple)) else float(center1[0][0])
                c1y = float(center1[1]) if not isinstance(center1[1], (list, tuple)) else float(center1[1][0])
                c2x = float(center2[0]) if not isinstance(center2[0], (list, tuple)) else float(center2[0][0])
                c2y = float(center2[1]) if not isinstance(center2[1], (list, tuple)) else float(center2[1][0])
                dist = np.sqrt(
                    (c1x - c2x) ** 2 + (c1y - c2y) ** 2
                )
                if r1 * 1.5 < dist < r1 * 8:
                    slots.append({"circle1": c1, "circle2": c2, "distance": dist})
        return slots

    def _check_reference(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        ref_matches = list(re.finditer(r"参考", ocr_all))
        if not ref_matches:
            return issues
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        for rm in ref_matches:
            ref_pos = rm.start()
            context_start = max(0, ref_pos - 30)
            context_end = min(len(ocr_all), ref_pos + 30)
            context = ocr_all[context_start:context_end]
            has_parentheses = bool(re.search(r"\(\d+\.?\d*\)", context))
            if not has_parentheses:
                confidence = 0.6
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "参考尺寸疑似未加括号",
                        f"'参考'附近未发现括号标注的尺寸数字，上下文: '{context}'",
                        {"context": context, "has_parentheses": False},
                        source=["atlas_rule", "ocr"],
                    )
                )
        return issues

    def _check_number_pos(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        dim_structures = (
            features.get("raw_refs", {})
            .get("geometry", {})
            .get("dimension_structures", [])
        )
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        if not dim_structures or not texts:
            return issues
        for ds in dim_structures:
            try:
                l1 = ds.get("line1", {})
                l2 = ds.get("line2", {})
                s1 = l1.get("start", [0, 0])
                e1 = l1.get("end", [0, 0])
                s2 = l2.get("start", [0, 0])
                e2 = l2.get("end", [0, 0])
                mid1_x = (float(s1[0]) + float(e1[0])) / 2
                mid1_y = (float(s1[1]) + float(e1[1])) / 2
                mid2_x = (float(s2[0]) + float(e2[0])) / 2
                mid2_y = (float(s2[1]) + float(e2[1])) / 2
                dim_mid_x = (mid1_x + mid2_x) / 2
                dim_mid_y = (mid1_y + mid2_y) / 2
                angle = ds.get("angle_diff", 0)
                is_vertical = 70 < angle < 110
                is_horizontal = angle < 20 or angle > 160
                nearby_texts = []
                for t in texts:
                    bbox = t.get("bbox", [])
                    if len(bbox) >= 4:
                        tx = (float(bbox[0]) + float(bbox[2])) / 2
                        ty = (float(bbox[1]) + float(bbox[3])) / 2
                        dist = np.sqrt((tx - dim_mid_x) ** 2 + (ty - dim_mid_y) ** 2)
                        if dist < 50:
                            nearby_texts.append(t)
                for nt in nearby_texts:
                    bbox = nt.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    text_y = (float(bbox[1]) + float(bbox[3])) / 2
                    text_x = (float(bbox[0]) + float(bbox[2])) / 2
                    if is_vertical and text_x > dim_mid_x + 10:
                        confidence = 0.55
                        level = self._resolve_level(rule, confidence)
                        issues.append(
                            self._make_issue(
                                rule,
                                level,
                                confidence,
                                "尺寸数字位置疑似错误",
                                "垂直尺寸的数字应在尺寸线左侧，字头朝左",
                                {"text": nt.get("text", "")},
                                source=["atlas_rule", "ocr", "geometry"],
                            )
                        )
                    elif is_horizontal and text_y > dim_mid_y + 10:
                        confidence = 0.55
                        level = self._resolve_level(rule, confidence)
                        issues.append(
                            self._make_issue(
                                rule,
                                level,
                                confidence,
                                "尺寸数字位置疑似错误",
                                "水平尺寸的数字应在尺寸线上方，字头朝上",
                                {"text": nt.get("text", "")},
                                source=["atlas_rule", "ocr", "geometry"],
                            )
                        )
            except Exception:
                pass
        return issues

    def _check_angle(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        angle_matches = re.findall(r"\d+\.?\d*\s*[°˚]", ocr_all)
        if not angle_matches:
            return issues
        for am in angle_matches:
            try:
                value = float(re.sub(r"[°˚]", "", am).strip())
                if 0 < value < 360 and value != 45:
                    confidence = 0.5
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "角度标注疑似需要检查",
                            f"检测到角度值{am}，请确认尺寸线为圆弧且数字水平书写",
                            {"angle_text": am},
                            source=["atlas_rule", "ocr"],
                        )
                    )
            except (ValueError, TypeError):
                continue
        return issues

    def _check_tolerance_frame(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        tolerance_symbols = ["⊥", "∥", "⌒", "○", "⏥", "∠", "↗", "⊕", "⌭"]
        found_symbols = [s for s in tolerance_symbols if s in ocr_all]
        if not found_symbols:
            latin_tolerance = re.findall(
                r"\b(?:flatness|straightness|roundness|cylindricity|parallelism|"
                r"perpendicularity|angularity|position|concentricity|symmetry)\b",
                ocr_all,
                re.IGNORECASE,
            )
            if not latin_tolerance:
                return issues
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        for t in texts:
            text = t.get("text", "")
            for sym in found_symbols:
                if sym in text:
                    parts = re.split(r"[|｜]", text)
                    if len(parts) >= 2:
                        first_part = parts[0].strip()
                        if not re.search(r"[⊥∥⌒○⏥∠↗⊕⌭]", first_part):
                            confidence = 0.65
                            level = self._resolve_level(rule, confidence)
                            issues.append(
                                self._make_issue(
                                    rule,
                                    level,
                                    confidence,
                                    "形位公差框格填写顺序疑似错误",
                                    f"框格'{text}'中公差项目符号未出现在第一格",
                                    {"text": text, "symbol": sym},
                                    source=["atlas_rule", "ocr"],
                                )
                            )
                    break
        return issues

    def _check_dim_tolerance(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        tolerance_patterns = re.findall(
            r"[\-+]?\d+\.?\d*\s*[⁻⁺\-+]\s*\d+\.?\d*", ocr_all
        )
        code_patterns = re.findall(r"[HhGgFfEeDdCcBbAa]\d+", ocr_all)
        if not tolerance_patterns and not code_patterns:
            return issues
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        for t in texts:
            text = t.get("text", "")
            if re.search(r"[\-+]\d+\.?\d*", text):
                upper_lower = re.findall(r"[+\-]\s*\d+\.?\d*", text)
                if len(upper_lower) >= 2:
                    try:
                        vals = []
                        for v in upper_lower:
                            vals.append(float(v.replace(" ", "")))
                        if vals[0] > 0 and vals[1] < 0:
                            pass
                        elif vals[0] < 0 and vals[1] > 0:
                            confidence = 0.6
                            level = self._resolve_level(rule, confidence)
                            issues.append(
                                self._make_issue(
                                    rule,
                                    level,
                                    confidence,
                                    "尺寸公差标注格式疑似错误",
                                    f"上下偏差顺序可能有误: '{text}'，上偏差应在上、下偏差应在下",
                                    {"text": text},
                                    source=["atlas_rule", "ocr"],
                                )
                            )
                    except (ValueError, IndexError):
                        pass
        return issues

    def _check_generic_ocr_regex(self, features: Dict, rule: Dict) -> List[Dict]:
        return []

    # ================================================================
    # hole_group: 成规律孔组标注
    # ================================================================
    def _check_hole_group(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        hole_groups = features.get("geometry_groups", {}).get("hole_groups", [])
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        for group in hole_groups:
            count = group.get("count", 0)
            if count < 3:
                continue
            centers = group.get("centers", [])
            is_grid = self._check_grid_pattern(centers)
            is_circular = self._check_uniform_distribution(centers)
            if is_grid or is_circular:
                count_texts = features.get("ocr_patterns", {}).get("count_texts", [])
                has_count_annotation = len(count_texts) > 0
                if not has_count_annotation:
                    confidence = 0.6
                    level = self._resolve_level(rule, confidence)
                    pattern_type = "网格" if is_grid else "圆周"
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "成规律分布孔组标注方式疑似错误",
                            f"检测到{count}个{pattern_type}分布的孔，"
                            f"应采用简化标注方式而非逐一标注",
                            {
                                "hole_count": count,
                                "pattern_type": pattern_type,
                                "avg_radius": group.get("avg_radius", 0),
                            },
                            source=["atlas_rule", "geometry", "ocr"],
                        )
                    )
        return issues

    def _check_grid_pattern(self, centers: List) -> bool:
        if len(centers) < 4:
            return False
        try:
            xs = sorted(set(round(c[0] / 5) * 5 for c in centers))
            ys = sorted(set(round(c[1] / 5) * 5 for c in centers))
            if len(xs) >= 2 and len(ys) >= 2:
                x_gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
                y_gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
                if x_gaps and y_gaps:
                    x_uniform = max(x_gaps) - min(x_gaps) < max(x_gaps) * 0.3
                    y_uniform = max(y_gaps) - min(y_gaps) < max(y_gaps) * 0.3
                    return x_uniform and y_uniform
        except Exception:
            pass
        return False

    # ================================================================
    # roughness: 全部/其余/符号方向
    # ================================================================
    def _check_roughness(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "ROUGH_ALL" in rule_id:
                issues.extend(self._check_rough_all(features, rule))
            elif "ROUGH_QIYU" in rule_id:
                issues.extend(self._check_rough_qiyu(features, rule))
            elif "ROUGH_SYMBOL_DIR" in rule_id:
                issues.extend(self._check_rough_symbol_dir(features, rule))
            else:
                issues.extend(self._check_roughness_generic(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] roughness 异常 ({rule_id}): {e}")
        return issues

    def _check_rough_all(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        roughness_texts = features.get("ocr_patterns", {}).get("roughness_texts", [])
        all_text = features.get("ocr_patterns", {}).get("all_text", "")
        has_all_keyword = any("全部" in t.get("text", "") for t in roughness_texts)
        if not has_all_keyword:
            return issues
        top_right = features.get("regions", {}).get("top_right", [0, 0, 0, 0])
        all_in_top_right = False
        for rt in roughness_texts:
            if "全部" in rt.get("text", ""):
                bbox = rt.get("bbox", [])
                if len(bbox) >= 4:
                    tx = (bbox[0] + bbox[2]) / 2
                    ty = (bbox[1] + bbox[3]) / 2
                    if (top_right[0] <= tx <= top_right[2]
                            and top_right[1] <= ty <= top_right[3]):
                        all_in_top_right = True
                        break
        if not all_in_top_right:
            confidence = 0.75
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    '粗糙度"全部"标注位置疑似错误',
                    '"全部"粗糙度标注应在图样右上角，但检测到的位置不在右上角区域',
                    {"roughness_texts": roughness_texts, "top_right": top_right},
                    source=["atlas_rule", "ocr"],
                )
            )
        ra_texts = [t for t in roughness_texts if "Ra" in t.get("text", "")]
        if len(ra_texts) > 1 and has_all_keyword:
            confidence = 0.7
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    '粗糙度"全部"标注方式疑似错误',
                    '标注"全部"时只需在右上角标注一个粗糙度符号，'
                    '但检测到多个Ra标注',
                    {"ra_count": len(ra_texts), "has_all": True},
                    source=["atlas_rule", "ocr"],
                )
            )
        return issues

    def _check_rough_qiyu(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        roughness_texts = features.get("ocr_patterns", {}).get("roughness_texts", [])
        has_qiyu = any("其余" in t.get("text", "") for t in roughness_texts)
        if not has_qiyu:
            return issues
        ra_values = []
        for rt in roughness_texts:
            text = rt.get("text", "")
            ra_match = re.search(r"Ra\s*(\d+\.?\d*)", text)
            if ra_match:
                try:
                    ra_values.append(float(ra_match.group(1)))
                except ValueError:
                    pass
        if len(ra_values) < 2:
            return issues
        from collections import Counter
        ra_counter = Counter(ra_values)
        most_common_ra, most_common_count = ra_counter.most_common(1)[0]
        top_right = features.get("regions", {}).get("top_right", [0, 0, 0, 0])
        qiyu_in_top_right = False
        for rt in roughness_texts:
            if "其余" in rt.get("text", ""):
                bbox = rt.get("bbox", [])
                if len(bbox) >= 4:
                    tx = (bbox[0] + bbox[2]) / 2
                    ty = (bbox[1] + bbox[3]) / 2
                    if (top_right[0] <= tx <= top_right[2]
                            and top_right[1] <= ty <= top_right[3]):
                        qiyu_in_top_right = True
                        break
        if not qiyu_in_top_right:
            confidence = 0.65
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    '粗糙度"其余"标注位置疑似错误',
                    '"其余"粗糙度标注应在图样右上角',
                    {"qiyu_in_top_right": False},
                    source=["atlas_rule", "ocr"],
                )
            )
        qiyu_ra = None
        for rt in roughness_texts:
            text = rt.get("text", "")
            if "其余" in text:
                ra_match = re.search(r"Ra\s*(\d+\.?\d*)", text)
                if ra_match:
                    try:
                        qiyu_ra = float(ra_match.group(1))
                    except ValueError:
                        pass
                break
        if qiyu_ra is not None and qiyu_ra != most_common_ra:
            confidence = 0.6
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    '粗糙度"其余"标注值疑似错误',
                    f'"其余"标注的Ra值为{qiyu_ra}，但图纸中最常见的Ra值为{most_common_ra}，'
                    f'"其余"应标注多数表面的粗糙度要求',
                    {"qiyu_ra": qiyu_ra, "most_common_ra": most_common_ra},
                    source=["atlas_rule", "ocr"],
                )
            )
        return issues

    def _check_rough_symbol_dir(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        roughness_texts = features.get("ocr_patterns", {}).get("roughness_texts", [])
        ra_texts = [t for t in roughness_texts if "Ra" in t.get("text", "")]
        if len(ra_texts) <= 1:
            return issues
        contours = features.get("raw_refs", {}).get("geometry", {}).get("contours", [])
        if not contours:
            return issues
        for rt in ra_texts:
            bbox = rt.get("bbox", [])
            if len(bbox) < 4:
                continue
            tx = (bbox[0] + bbox[2]) / 2
            ty = (bbox[1] + bbox[3]) / 2
            nearest_contour = None
            min_dist = float("inf")
            for cnt in contours:
                cbbox = cnt.get("bbox", [0, 0, 0, 0])
                cx = (cbbox[0] + cbbox[2]) / 2
                cy = (cbbox[1] + cbbox[3]) / 2
                dist = np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_contour = cbbox
            if nearest_contour and min_dist < 100:
                cbbox = nearest_contour
                if ty < cbbox[1] and tx > cbbox[2]:
                    confidence = 0.45
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "表面粗糙度符号方向疑似错误",
                            "粗糙度符号位于轮廓右上外侧，尖端可能未指向被注表面",
                            {"text_bbox": bbox, "contour_bbox": cbbox},
                            source=["atlas_rule", "ocr", "geometry"],
                        )
                    )
        return issues

    def _check_roughness_generic(self, features: Dict, rule: Dict) -> List[Dict]:
        return []

    # ================================================================
    # title_block: 比例标准、阶段标记
    # ================================================================
    def _check_title_block(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "SCALE_STANDARD" in rule_id:
                issues.extend(self._check_scale_standard(features, rule))
            elif "STAGE_MARK" in rule_id:
                issues.extend(self._check_stage_mark(features, rule))
            else:
                issues.extend(self._check_title_block_generic(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] title_block 异常 ({rule_id}): {e}")
        return issues

    def _check_scale_standard(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        scale_texts = features.get("ocr_patterns", {}).get("scale_texts", [])
        title_block_region = features.get("regions", {}).get("title_block", [0, 0, 0, 0])
        params = rule.get("params", {})
        standard_scales = params.get(
            "standard_scales",
            ["1:1", "1:2", "1:5", "1:10", "2:1", "5:1", "10:1"],
        )
        forbid_slash = params.get("forbid_slash_format", True)
        for st in scale_texts:
            text = st.get("text", "")
            bbox = st.get("bbox", [])
            in_title_block = False
            if len(bbox) >= 4 and len(title_block_region) >= 4:
                tx = (bbox[0] + bbox[2]) / 2
                ty = (bbox[1] + bbox[3]) / 2
                if (title_block_region[0] <= tx <= title_block_region[2]
                        and title_block_region[1] <= ty <= title_block_region[3]):
                    in_title_block = True
            scale_clean = re.sub(r"\s+", "", text)
            is_standard = scale_clean in standard_scales
            has_slash = "/" in scale_clean
            if has_slash and forbid_slash:
                confidence = 0.8
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "标题栏比例标注格式错误",
                        f"比例'{text}'使用了斜线格式，应使用冒号格式（如1:2而非1/2）",
                        {"text": text, "in_title_block": in_title_block},
                        source=["atlas_rule", "ocr"],
                    )
                )
            elif not is_standard and in_title_block:
                confidence = 0.7
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "标题栏比例标注疑似非标准",
                        f"比例'{text}'不在标准比例列表中，"
                        f"标准比例包括: {', '.join(standard_scales)}",
                        {"text": text, "standard_scales": standard_scales},
                        source=["atlas_rule", "ocr"],
                    )
                )
        return issues

    def _check_stage_mark(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        params = rule.get("params", {})
        valid_marks = params.get("valid_marks", ["S", "A", "B"])
        require_mark = params.get("require_mark", True)
        title_block_region = features.get("regions", {}).get("title_block", [0, 0, 0, 0])
        found_marks = []
        for t in texts:
            text = t.get("text", "").strip()
            if text in valid_marks:
                bbox = t.get("bbox", [])
                in_title = False
                if len(bbox) >= 4 and len(title_block_region) >= 4:
                    tx = (bbox[0] + bbox[2]) / 2
                    ty = (bbox[1] + bbox[3]) / 2
                    if (title_block_region[0] <= tx <= title_block_region[2]
                            and title_block_region[1] <= ty <= title_block_region[3]):
                        in_title = True
                found_marks.append({"mark": text, "in_title_block": in_title})
        if require_mark and not found_marks:
            confidence = 0.55
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "阶段标记疑似缺失",
                    "标题栏中未检测到阶段标记(S/A/B)",
                    {"found_marks": [], "valid_marks": valid_marks},
                    source=["atlas_rule", "ocr"],
                )
            )
        for fm in found_marks:
            if not fm["in_title_block"]:
                confidence = 0.6
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "阶段标记位置疑似错误",
                        f"阶段标记'{fm['mark']}'不在标题栏区域内",
                        {"mark": fm["mark"]},
                        source=["atlas_rule", "ocr"],
                    )
                )
        return issues

    def _check_title_block_generic(self, features: Dict, rule: Dict) -> List[Dict]:
        return []

    # ================================================================
    # view_heuristic: 局部视图方向、第三角、焊缝箭头、图幅选择
    # ================================================================
    def _check_view_heuristic(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "LOCAL_DIRECTION" in rule_id:
                issues.extend(self._check_local_direction(features, rule))
            elif "LOCAL_THIRD_ANGLE" in rule_id:
                issues.extend(self._check_local_third_angle(features, rule))
            elif "WELD_ARROW" in rule_id:
                issues.extend(self._check_weld_arrow(features, rule))
            elif "SHEET_SELECT" in rule_id:
                issues.extend(self._check_sheet_select(features, rule))
            else:
                issues.extend(self._check_view_heuristic_generic(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] view_heuristic 异常 ({rule_id}): {e}")
        return issues

    def _check_local_direction(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        view_labels = features.get("ocr_patterns", {}).get("view_labels", [])
        arrows = features.get("raw_refs", {}).get("geometry", {}).get("arrows", [])
        view_areas = (
            features.get("raw_refs", {})
            .get("structure", {})
            .get("view_areas", [])
        )
        params = rule.get("params", {})
        require_arrow = params.get("require_arrow", True)
        require_letter = params.get("require_letter", True)
        require_view_name = params.get("require_view_name", True)
        if not view_labels and not view_areas:
            return issues
        for vl in view_labels:
            label_text = vl.get("text", "")
            label_bbox = vl.get("bbox", [])
            if len(label_bbox) < 4:
                continue
            lx = (label_bbox[0] + label_bbox[2]) / 2
            ly = (label_bbox[1] + label_bbox[3]) / 2
            nearby_arrow = False
            for arrow in arrows:
                abbox = arrow.get("bbox", [0, 0, 0, 0])
                if len(abbox) >= 4:
                    ax = (abbox[0] + abbox[2]) / 2
                    ay = (abbox[1] + abbox[3]) / 2
                    dist = np.sqrt((lx - ax) ** 2 + (ly - ay) ** 2)
                    if dist < 300:
                        nearby_arrow = True
                        break
            if require_arrow and not nearby_arrow:
                confidence = 0.6
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "局部视图投影方向标注疑似缺失",
                        f"检测到视图标记'{label_text}'，但附近未发现投影方向箭头",
                        {"label": label_text, "nearby_arrow": False},
                        source=["atlas_rule", "ocr", "geometry"],
                    )
                )
        return issues

    def _check_local_third_angle(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        has_third_angle_symbol = bool(
            re.search(r"第三角|3rd\s*angle|第三角画法", ocr_all, re.IGNORECASE)
        )
        view_areas = (
            features.get("raw_refs", {})
            .get("structure", {})
            .get("view_areas", [])
        )
        if len(view_areas) < 2:
            return issues
        small_views = []
        for va in view_areas:
            bbox = va.get("bbox", [0, 0, 0, 0])
            if len(bbox) >= 4:
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                area = w * h
                if area > 0:
                    small_views.append(va)
        if len(small_views) < 2:
            return issues
        main_view = max(small_views, key=lambda v: v.get("bbox", [0, 0, 0, 0])[2] * v.get("bbox", [0, 0, 0, 0])[3] if len(v.get("bbox", [])) >= 4 else 0)
        main_bbox = main_view.get("bbox", [0, 0, 0, 0])
        if len(main_bbox) < 4:
            return issues
        for sv in small_views:
            if sv is main_view:
                continue
            sbbox = sv.get("bbox", [0, 0, 0, 0])
            if len(sbbox) < 4:
                continue
            sx = (sbbox[0] + sbbox[2]) / 2
            sy = (sbbox[1] + sbbox[3]) / 2
            mx = (main_bbox[0] + main_bbox[2]) / 2
            my = (main_bbox[1] + main_bbox[3]) / 2
            dist = np.sqrt((sx - mx) ** 2 + (sy - my) ** 2)
            main_diag = np.sqrt(
                (main_bbox[2] - main_bbox[0]) ** 2 + (main_bbox[3] - main_bbox[1]) ** 2
            )
            if dist < main_diag * 0.8:
                if not has_third_angle_symbol:
                    confidence = 0.5
                    level = self._resolve_level(rule, confidence)
                    issues.append(
                        self._make_issue(
                            rule,
                            level,
                            confidence,
                            "第三角画法局部视图配置疑似错误",
                            "检测到小视图靠近主视图，但未发现第三角画法标识",
                            {"view_distance": round(dist, 1), "has_third_angle_symbol": False},
                            source=["atlas_rule", "geometry", "ocr"],
                        )
                    )
                break
        return issues

    def _check_weld_arrow(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        weld_keywords = re.findall(r"焊[缝接]|weld", ocr_all, re.IGNORECASE)
        if not weld_keywords:
            return issues
        arrows = features.get("raw_refs", {}).get("geometry", {}).get("arrows", [])
        texts = features.get("raw_refs", {}).get("ocr", {}).get("texts", [])
        weld_texts = [
            t for t in texts
            if re.search(r"焊[缝接]|weld", t.get("text", ""), re.IGNORECASE)
        ]
        for wt in weld_texts:
            wbbox = wt.get("bbox", [])
            if len(wbbox) < 4:
                continue
            wx = (wbbox[0] + wbbox[2]) / 2
            wy = (wbbox[1] + wbbox[3]) / 2
            nearby_arrows = []
            for arrow in arrows:
                abbox = arrow.get("bbox", [0, 0, 0, 0])
                if len(abbox) >= 4:
                    ax = (abbox[0] + abbox[2]) / 2
                    ay = (abbox[1] + abbox[3]) / 2
                    dist = np.sqrt((wx - ax) ** 2 + (wy - ay) ** 2)
                    if dist < 150:
                        nearby_arrows.append(arrow)
            if not nearby_arrows:
                confidence = 0.55
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "焊缝箭头线疑似缺失",
                        f"检测到焊缝标注'{wt.get('text', '')}'，但附近未发现箭头线",
                        {"weld_text": wt.get("text", ""), "nearby_arrows": 0},
                        source=["atlas_rule", "ocr", "geometry"],
                    )
                )
        return issues

    def _check_sheet_select(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        structure = features.get("raw_refs", {}).get("structure", {})
        geometry = features.get("raw_refs", {}).get("geometry", {})
        image_size = structure.get("image_size", {})
        img_w = image_size.get("width", 0)
        img_h = image_size.get("height", 0)
        if img_w <= 0 or img_h <= 0:
            return issues
        img_area = img_w * img_h
        contours = geometry.get("contours", [])
        if not contours:
            return issues
        drawing_area = 0
        for cnt in contours:
            drawing_area += cnt.get("area", 0)
        if img_area <= 0:
            return issues
        utilization = drawing_area / img_area
        params = rule.get("params", {})
        min_ratio = params.get("min_utilization_ratio", 0.70)
        max_ratio = params.get("max_utilization_ratio", 0.90)
        if utilization < min_ratio:
            confidence = 0.55
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "图幅利用率疑似过低",
                    f"图面利用率约{utilization:.0%}，低于建议的{min_ratio:.0%}，"
                    f"可能图幅选择偏大",
                    {"utilization": round(utilization, 3), "min_ratio": min_ratio},
                    source=["atlas_rule", "geometry", "structure"],
                )
            )
        elif utilization > max_ratio:
            confidence = 0.55
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "图幅利用率疑似过高",
                    f"图面利用率约{utilization:.0%}，超过建议的{max_ratio:.0%}，"
                    f"可能图幅选择偏小",
                    {"utilization": round(utilization, 3), "max_ratio": max_ratio},
                    source=["atlas_rule", "geometry", "structure"],
                )
            )
        return issues

    def _check_view_heuristic_generic(self, features: Dict, rule: Dict) -> List[Dict]:
        return []

    # ================================================================
    # context_line_usage: 粗点画线/粗虚线使用、断裂画法
    # ================================================================
    def _check_context_line_usage(self, features: Dict, rule: Dict) -> List[Dict]:
        rule_id = rule.get("rule_id", "")
        issues = []
        try:
            if "COARSE_USAGE" in rule_id:
                issues.extend(self._check_coarse_usage(features, rule))
            elif "LINE_BREAK" in rule_id:
                issues.extend(self._check_line_break(features, rule))
            else:
                issues.extend(self._check_context_line_generic(features, rule))
        except Exception as e:
            logger.error(f"[AtlasRulePlugin] context_line_usage 异常 ({rule_id}): {e}")
        return issues

    def _check_coarse_usage(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        coarse_dash = features.get("line_candidates", {}).get("coarse_dash_lines", [])
        coarse_chain = features.get("line_candidates", {}).get("coarse_chain_lines", [])
        has_coarse = len(coarse_dash) > 0 or len(coarse_chain) > 0
        if not has_coarse:
            return issues
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        has_limit_region = bool(re.search(r"限定|表示区域|范围", ocr_all))
        has_surface_treatment = bool(re.search(r"表面处理|镀|涂|渗|淬火", ocr_all))
        if not has_limit_region and not has_surface_treatment:
            confidence = 0.55
            level = self._resolve_level(rule, confidence)
            issues.append(
                self._make_issue(
                    rule,
                    level,
                    confidence,
                    "粗点画线/粗虚线使用场景疑似错误",
                    "检测到粗点画线或粗虚线，但未发现限定表示区域或表面处理指示的上下文，"
                    "粗点画线仅用于限定表示区域，粗虚线仅用于允许表面处理指示",
                    {
                        "coarse_dash_count": len(coarse_dash),
                        "coarse_chain_count": len(coarse_chain),
                        "has_limit_region": has_limit_region,
                        "has_surface_treatment": has_surface_treatment,
                    },
                    source=["atlas_rule", "geometry", "ocr"],
                )
            )
        return issues

    def _check_line_break(self, features: Dict, rule: Dict) -> List[Dict]:
        issues = []
        wavy_lines = features.get("line_candidates", {}).get("wavy_lines", [])
        ocr_all = features.get("ocr_patterns", {}).get("all_text", "")
        has_break_context = bool(re.search(r"断裂|断开|截断|折断", ocr_all))
        contours = features.get("raw_refs", {}).get("geometry", {}).get("contours", [])
        long_contours = [
            c for c in contours
            if c.get("perimeter", 0) > 500 and c.get("area", 0) > 5000
        ]
        if not long_contours:
            return issues
        for lc in long_contours:
            bbox = lc.get("bbox", [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect < 3:
                continue
            nearby_wavy = False
            for wl in wavy_lines:
                wbbox = wl.get("bbox", [0, 0, 0, 0])
                if len(wbbox) >= 4 and len(bbox) >= 4:
                    cx1 = (bbox[0] + bbox[2]) / 2
                    cy1 = (bbox[1] + bbox[3]) / 2
                    cx2 = (wbbox[0] + wbbox[2]) / 2
                    cy2 = (wbbox[1] + wbbox[3]) / 2
                    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                    if dist < max(w, h) * 0.5:
                        nearby_wavy = True
                        break
            if not nearby_wavy and (has_break_context or aspect > 5):
                confidence = 0.5
                level = self._resolve_level(rule, confidence)
                issues.append(
                    self._make_issue(
                        rule,
                        level,
                        confidence,
                        "断裂画法线型疑似缺失",
                        "检测到细长轮廓（疑似断裂结构），但附近未发现波浪线/双点画线等断裂边界线型",
                        {
                            "contour_bbox": bbox,
                            "aspect_ratio": round(aspect, 2),
                            "nearby_wavy": False,
                            "has_break_context": has_break_context,
                        },
                        source=["atlas_rule", "geometry"],
                    )
                )
        return issues

    def _check_context_line_generic(self, features: Dict, rule: Dict) -> List[Dict]:
        return []
