import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

from .atlas_registry import AtlasRegistry
from .atlas_case_schema import AtlasCase

logger = logging.getLogger("AtlasRetriever")

try:
    from config_loader import ATLAS_MAX_CONTEXT_CASES as _MAX_CASES
    MAX_CONTEXT_CASES = _MAX_CASES
except ImportError:
    MAX_CONTEXT_CASES = 3


class AtlasRetriever:
    def __init__(self, registry: AtlasRegistry, max_cases: int = 0):
        self.registry = registry
        self.max_cases = max_cases or MAX_CONTEXT_CASES

    def retrieve_by_error_type(self, error_type: str,
                                top_k: Optional[int] = None) -> List[Tuple[AtlasCase, float]]:
        k = top_k or self.max_cases
        all_cases = self.registry.get_all_cases()

        scored = []
        for case in all_cases:
            score = self._compute_relevance(case, error_type, {})
            if score > 0:
                scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def retrieve_by_context(self, error_type: str, context: Dict,
                             top_k: Optional[int] = None) -> List[Tuple[AtlasCase, float]]:
        k = top_k or self.max_cases
        all_cases = self.registry.get_all_cases()

        scored = []
        for case in all_cases:
            score = self._compute_relevance(case, error_type, context)
            if score > 0:
                scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def retrieve_similar(self, analysis_result: Dict,
                          top_k: Optional[int] = None) -> List[Tuple[AtlasCase, float]]:
        k = top_k or self.max_cases
        all_cases = self.registry.get_all_cases()
        errors = analysis_result.get('errors', [])

        error_types = set(e.get('type', '') for e in errors)

        scored = []
        for case in all_cases:
            score = 0.0
            if case.error_type in error_types:
                score += 0.5
            for tag in case.tags:
                if tag in error_types:
                    score += 0.2

            geo = analysis_result.get('geo_result', {})
            if geo:
                case_features = case.drawing_features
                if case_features:
                    score += self._feature_similarity(geo, case_features) * 0.3

            score *= case.confidence
            positive_ratio = case.get_positive_feedback_ratio()
            score *= (0.5 + 0.5 * positive_ratio)

            if score > 0:
                scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _compute_relevance(self, case: AtlasCase, error_type: str,
                            context: Dict) -> float:
        score = 0.0

        if case.error_type == error_type:
            score += 0.6
        elif error_type in case.tags:
            score += 0.3

        if context:
            feature_overlap = 0
            total_features = 0
            for key, value in context.items():
                total_features += 1
                if key in case.drawing_features:
                    if case.drawing_features[key] == value:
                        feature_overlap += 1
            if total_features > 0:
                score += (feature_overlap / total_features) * 0.3

        score *= case.confidence
        positive_ratio = case.get_positive_feedback_ratio()
        score *= (0.5 + 0.5 * positive_ratio)

        return score

    def _feature_similarity(self, geo: Dict, case_features: Dict) -> float:
        similarity = 0.0
        comparisons = 0

        for key in ['line_count', 'circle_count', 'arrow_count']:
            if key in case_features:
                comparisons += 1
                current_val = 0
                if key == 'line_count':
                    current_val = len(geo.get('lines', []))
                elif key == 'circle_count':
                    current_val = len(geo.get('circles', []))
                elif key == 'arrow_count':
                    current_val = len(geo.get('arrows', []))

                case_val = case_features[key]
                if case_val > 0:
                    similarity += 1.0 - abs(current_val - case_val) / max(current_val, case_val)
                elif current_val == 0:
                    similarity += 1.0

        return similarity / max(comparisons, 1)
