"""
Atlas VLM Judge 图册 few-shot 提供器
======================================
为 VLM Judge 提供图册正误对照评分标尺。

在 Phase 5 (VLM Judge) 中，将图册案例以 few-shot 形式注入评审 prompt，
帮助 VLM 建立正确/错误画法的视觉参照，提升评审准确性。
"""

import logging
from typing import Dict, List, Optional

from atlas.atlas_registry import AtlasRegistry

logger = logging.getLogger("AtlasVLMFewshotProvider")


class AtlasVLMFewshotProvider:
    """为VLM Judge提供图册正误对照评分标尺"""

    def __init__(self, atlas_registry: AtlasRegistry):
        self.registry = atlas_registry
        self._fewshot_cache = None

    def get_fewshot_examples(
        self,
        error_categories: List[str] = None,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        获取VLM评审用的few-shot样例

        Args:
            error_categories: 要获取的错误类别列表，None表示全部
            top_k: 最多返回top_k条

        Returns:
            few-shot样例列表，每个包含wrong_image路径、correct_image路径、
            expected_issue、judge_rule
        """
        try:
            if self._fewshot_cache is None:
                self._fewshot_cache = self._build_cache()

            candidates = self._fewshot_cache

            if error_categories:
                cat_set = set(error_categories)
                candidates = [
                    ex for ex in candidates
                    if ex.get("v2_error_category", "") in cat_set
                ]

            candidates.sort(
                key=lambda ex: ex.get("confidence", 0.0), reverse=True
            )
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"[AtlasVLMFewshotProvider] 获取few-shot样例异常: {e}")
            return []

    def build_judge_prompt_extension(
        self, error_categories: List[str] = None
    ) -> str:
        """
        生成VLM Judge prompt的图册扩展部分

        Returns:
            注入到VLM Judge prompt中的文本
        """
        try:
            examples = self.get_fewshot_examples(error_categories, top_k=2)
            if not examples:
                return ""

            parts = ["请参考以下图册正误案例评审系统报告。"]
            for ex in examples:
                parts.append(f"案例：{ex.get('case_name', '')}")
                parts.append(f"期望识别：{ex.get('expected_issue', '')}")
                rules = ex.get("judge_rule", {})
                if isinstance(rules, dict):
                    for dim, desc in rules.items():
                        parts.append(f"  {dim}: {desc}")

            parts.append("\n注意：")
            parts.append("- 错误图展示典型错误；正确图展示推荐画法；")
            parts.append("- 图册案例仅用于辅助评分；")
            parts.append("- 只有当前图纸证据支持时，才能认为报告准确；")
            parts.append("- suspected_issue不能写成确定性错误；")
            parts.append("- suggestion不应计入硬错误。")

            return "\n".join(parts)
        except Exception as e:
            logger.error(
                f"[AtlasVLMFewshotProvider] 生成prompt扩展异常: {e}"
            )
            return ""

    def invalidate_cache(self):
        """清除缓存，下次调用时重新构建"""
        self._fewshot_cache = None

    def _build_cache(self) -> List[Dict]:
        """从registry的cases中构建few-shot缓存"""
        cache = []
        if not self.registry or not self.registry.cases:
            return cache

        for case in self.registry.cases:
            try:
                if not case.get("vlm_fewshot_enabled", False):
                    continue
                if case.get("qa_status") not in ("reviewed", "published"):
                    continue

                wrong_image = case.get("wrong_image", "")
                correct_image = case.get("correct_image", "")
                if not wrong_image or not correct_image:
                    continue

                expected_issue = (
                    case.get("source_text", "")
                    or case.get("case_name", "")
                )

                judge_rule = self._derive_judge_rule(case)

                confidence = self._estimate_confidence(case)

                cache.append({
                    "case_id": case.get("case_id", ""),
                    "case_name": case.get("case_name", ""),
                    "v2_error_category": case.get("v2_error_category", ""),
                    "wrong_image": wrong_image,
                    "correct_image": correct_image,
                    "expected_issue": expected_issue,
                    "judge_rule": judge_rule,
                    "confidence": confidence,
                })
            except Exception:
                continue

        logger.info(
            f"[AtlasVLMFewshotProvider] 缓存构建完成: {len(cache)} 条few-shot样例"
        )
        return cache

    @staticmethod
    def _derive_judge_rule(case: Dict) -> Dict:
        """从案例数据推导评审规则"""
        rules = {}
        try:
            cat = case.get("v2_error_category", "")
            if cat:
                rules["error_category"] = f"应归类为{cat}"

            suggestion = case.get("suggestion", "")
            if suggestion:
                rules["suggestion_quality"] = (
                    f"建议应包含：{suggestion[:80]}"
                )

            teaching_hint = case.get("teaching_hint", "")
            if teaching_hint:
                rules["guidance"] = (
                    f"引导方向：{teaching_hint[:80]}"
                )

            if not rules:
                rules["general"] = "应正确识别该类错误"
        except Exception:
            rules["general"] = "应正确识别该类错误"
        return rules

    @staticmethod
    def _estimate_confidence(case: Dict) -> float:
        """估算案例的评审参考置信度"""
        score = 0.5
        try:
            if case.get("qa_status") == "published":
                score += 0.2
            if case.get("standard_status") == "verified":
                score += 0.15
            if case.get("source_type") == "teaching_atlas":
                score += 0.1
            keywords = case.get("keywords", [])
            if len(keywords) >= 3:
                score += 0.05
        except Exception:
            pass
        return min(score, 1.0)
