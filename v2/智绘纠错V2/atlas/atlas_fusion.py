"""
Atlas 结果融合器
================
图册规则结果与基础规则/LLM结果的融合器。

在 Phase 4 (Result Fusion) 中，将三层检测结果（base_rule / atlas_rule / llm）
进行去重合并与置信度提升，输出统一的错误列表。

融合规则：
1. base errors 保持 confirmed_error
2. atlas 高置信硬规则 → confirmed_error
3. atlas 启发式规则 → suspected_issue
4. LLM 独立发现但无OCR/几何证据 → suspected_issue 或 suggestion
5. 重复项合并（相同description的base+atlas → 提升置信度）
6. 按severity排序：高 > 中 > 低
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("AtlasEvidenceFusion")

SEVERITY_ORDER = {
    "高": 0,
    "confirmed_error": 0,
    "high": 0,
    "中": 1,
    "suspected_issue": 1,
    "medium": 1,
    "低": 2,
    "suggestion": 2,
    "low": 2,
}

JACCARD_THRESHOLD = 0.6


class AtlasEvidenceFusion:
    """图册规则结果与基础规则/LLM结果的融合器"""

    def merge(
        self,
        base_errors: List[Dict],
        atlas_errors: List[Dict],
        llm_errors: List[Dict],
    ) -> List[Dict]:
        """
        三层融合：base + atlas + llm

        Args:
            base_errors: 基础规则检测结果
            atlas_errors: 图册规则检测结果
            llm_errors: LLM分析结果

        Returns:
            融合后的错误列表，按severity排序
        """
        try:
            merged = []

            self._merge_base(base_errors, merged)
            self._merge_atlas(atlas_errors, merged)
            self._merge_llm(llm_errors, merged)

            result = self._sort(merged)
            logger.info(
                f"[AtlasEvidenceFusion] 融合完成: "
                f"base={len(base_errors)}, atlas={len(atlas_errors)}, "
                f"llm={len(llm_errors)}, merged={len(result)}"
            )
            return result
        except Exception as e:
            logger.error(f"[AtlasEvidenceFusion] 融合异常: {e}")
            all_errors = list(base_errors or []) + list(atlas_errors or []) + list(llm_errors or [])
            return all_errors

    def _merge_base(self, base_errors: List[Dict], merged: List[Dict]):
        """加入base errors，保持confirmed_error级别"""
        for e in (base_errors or []):
            try:
                e.setdefault("level", "confirmed_error")
                e.setdefault("source", [])
                if isinstance(e.get("source"), list):
                    if "base_rule" not in e["source"]:
                        e["source"].append("base_rule")
                else:
                    e["source"] = ["base_rule"]
                merged.append(e)
            except Exception:
                merged.append(e)

    def _merge_atlas(self, atlas_errors: List[Dict], merged: List[Dict]):
        """加入atlas errors，去重合并"""
        for e in (atlas_errors or []):
            try:
                matched = self._find_duplicate(e, merged)
                if matched:
                    if isinstance(matched.get("source"), list):
                        if "atlas_rule" not in matched["source"]:
                            matched["source"].append("atlas_rule")
                    old_conf = matched.get("confidence", 0.7)
                    new_conf = e.get("confidence", 0.7)
                    matched["confidence"] = max(old_conf, new_conf)
                    source_case_id = e.get("source_case_id", "")
                    if source_case_id:
                        matched["atlas_case_id"] = source_case_id
                    matched["suggestion"] = self._merge_suggestion(
                        matched.get("suggestion", ""),
                        e.get("suggestion", ""),
                    )
                    if matched.get("level") == "suspected_issue" and new_conf >= 0.8:
                        matched["level"] = "confirmed_error"
                        if "severity" not in matched:
                            matched["severity"] = "高"
                else:
                    e.setdefault("source", [])
                    if isinstance(e.get("source"), list):
                        if "atlas_rule" not in e["source"]:
                            e["source"].append("atlas_rule")
                    merged.append(e)
            except Exception:
                merged.append(e)

    def _merge_llm(self, llm_errors: List[Dict], merged: List[Dict]):
        """加入LLM errors，无匹配则降级为suspected_issue"""
        for e in (llm_errors or []):
            try:
                matched = self._find_duplicate(e, merged)
                if matched:
                    if isinstance(matched.get("source"), list):
                        if "llm" not in matched["source"]:
                            matched["source"].append("llm")
                else:
                    e.setdefault("level", "suspected_issue")
                    e.setdefault("source", [])
                    if isinstance(e.get("source"), list):
                        if "llm" not in e["source"]:
                            e["source"].append("llm")
                    merged.append(e)
            except Exception:
                merged.append(e)

    def _find_duplicate(
        self, error: Dict, existing: List[Dict]
    ) -> Optional[Dict]:
        """查找重复或高度相似的错误"""
        desc = error.get("description", "") or error.get("title", "")
        cat = (
            error.get("error_category", "")
            or error.get("type", "")
        )

        for e in existing:
            try:
                e_desc = e.get("description", "") or e.get("title", "")
                e_cat = (
                    e.get("error_category", "")
                    or e.get("type", "")
                )

                if desc and desc == e_desc:
                    return e

                if cat and cat == e_cat and desc and e_desc:
                    jaccard = self._jaccard_similarity(desc, e_desc)
                    if jaccard > JACCARD_THRESHOLD:
                        return e
            except Exception:
                continue

        return None

    @staticmethod
    def _jaccard_similarity(s1: str, s2: str) -> float:
        """计算两个字符串的字符级Jaccard相似度"""
        if not s1 or not s2:
            return 0.0
        try:
            set_a = set(s1)
            set_b = set(s2)
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            if union == 0:
                return 0.0
            return intersection / union
        except Exception:
            return 0.0

    @staticmethod
    def _merge_suggestion(s1: str, s2: str) -> str:
        """合并两条建议"""
        if not s1:
            return s2
        if not s2:
            return s1
        if s1 == s2:
            return s1
        return f"{s1}（图册参考：{s2}）"

    @staticmethod
    def _sort(errors: List[Dict]) -> List[Dict]:
        """按severity排序"""
        def sort_key(e):
            sev = e.get("severity", "")
            lvl = e.get("level", "")
            return SEVERITY_ORDER.get(sev, SEVERITY_ORDER.get(lvl, 1))
        return sorted(errors, key=sort_key)
