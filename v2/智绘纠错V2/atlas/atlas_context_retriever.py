"""
Atlas LLM 图册上下文检索器
==========================
基于关键词匹配的图册案例检索，为 LLM 提供教学参考上下文。

在 Phase 3 (LLM Analysis) 中，将当前检测到的错误与图册案例匹配，
为 LLM 注入相关正误对照案例，提升分析质量。
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("AtlasContextRetriever")


class AtlasContextRetriever:
    """基于关键词匹配的图册案例检索器，为LLM提供教学参考上下文"""

    def __init__(self, atlas_cases: List[Dict]):
        self.cases = atlas_cases or []
        logger.info(
            f"[AtlasContextRetriever] 初始化完成: {len(self.cases)} 条案例"
        )

    def retrieve(self, errors: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        根据当前检测到的错误，检索最相关的图册案例

        Args:
            errors: 当前检测到的错误列表，每个error包含type/description/suggestion等
            top_k: 返回最多top_k条案例

        Returns:
            相关图册案例列表，按相关度降序
        """
        if not errors:
            return []
        try:
            query = self._build_query(errors)
            if not query.strip():
                return []

            scored = self._score_cases(query)
            scored.sort(key=lambda x: x[0], reverse=True)

            result = [c for s, c in scored[:top_k]]
            if result:
                logger.info(
                    f"[AtlasContextRetriever] 检索到 {len(result)} 条相关案例 "
                    f"(scores: {[s for s, _ in scored[:top_k]]})"
                )
            return result
        except Exception as e:
            logger.error(f"[AtlasContextRetriever] 检索异常: {e}")
            return []

    def _build_query(self, errors: List[Dict]) -> str:
        """从errors中提取type/description/suggestion构建查询文本"""
        parts = []
        for e in errors:
            try:
                type_val = e.get("type", "") or e.get("error_category", "") or ""
                desc_val = e.get("description", "") or e.get("title", "") or ""
                sugg_val = e.get("suggestion", "") or ""
                combined = f"{type_val} {desc_val} {sugg_val}".strip()
                if combined:
                    parts.append(combined)
            except Exception:
                continue
        return " ".join(parts)

    def _score_cases(self, query: str) -> List[tuple]:
        """对每个case计算相关度分数"""
        scored = []
        query_lower = query.lower()

        for case in self.cases:
            try:
                if not case.get("llm_context_enabled", True):
                    continue
                if case.get("qa_status") not in ("reviewed", "published"):
                    continue

                score = 0

                for kw in case.get("keywords", []):
                    if kw and kw in query_lower:
                        score += 1

                cat = case.get("v2_error_category", "")
                if cat and cat in query:
                    score += 2

                fig = case.get("figure_no", "")
                if fig and fig in query:
                    score += 1

                case_name = case.get("case_name", "")
                if case_name and case_name in query:
                    score += 1

                source_text = case.get("source_text", "")
                if source_text:
                    source_words = source_text.split()
                    for w in source_words:
                        if len(w) >= 2 and w in query:
                            score += 0.5
                            break

                if score > 0:
                    scored.append((score, case))
            except Exception:
                continue

        return scored
