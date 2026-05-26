import logging
from typing import Dict, List, Optional, Tuple

from .atlas_registry import AtlasRegistry
from .atlas_case_schema import AtlasCase
from .atlas_rule_schema import AtlasRule

logger = logging.getLogger("AtlasMatcher")


class AtlasMatcher:
    def __init__(self, registry: AtlasRegistry):
        self.registry = registry

    def match_errors(self, errors: List[Dict],
                      context: Optional[Dict] = None) -> List[Dict]:
        context = context or {}
        matched = []

        for error in errors:
            error_type = error.get('type', '')
            description = error.get('description', '')

            case_matches = self._match_cases(error_type, description, context)
            rule_matches = self._match_rules(error_type, context)

            match_info = {
                'error': error,
                'case_matches': case_matches[:3],
                'rule_matches': rule_matches[:3],
                'best_match': self._select_best_match(case_matches, rule_matches)
            }
            matched.append(match_info)

        return matched

    def _match_cases(self, error_type: str, description: str,
                      context: Dict) -> List[Dict]:
        all_cases = self.registry.get_all_cases()
        matches = []

        for case in all_cases:
            score = 0.0

            if case.error_type == error_type:
                score += 0.5

            for tag in case.tags:
                if tag == error_type or tag in description:
                    score += 0.2

            for case_error in case.errors_found:
                if case_error.get('description', '') == description:
                    score += 0.3
                    break
                case_desc_words = set(case_error.get('description', '').split())
                desc_words = set(description.split())
                overlap = len(case_desc_words & desc_words)
                if overlap > 0:
                    score += 0.1 * (overlap / max(len(desc_words), 1))

            score *= case.confidence
            positive_ratio = case.get_positive_feedback_ratio()
            score *= (0.5 + 0.5 * positive_ratio)

            if score > 0.1:
                matches.append({
                    'case_id': case.case_id,
                    'category': case.category,
                    'error_type': case.error_type,
                    'score': round(score, 3),
                    'corrections': case.corrections,
                    'gb_references': case.gb_references
                })

        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches

    def _match_rules(self, error_type: str, context: Dict) -> List[Dict]:
        all_rules = self.registry.get_all_rules()
        matches = []

        for rule in all_rules:
            if not rule.enabled:
                continue

            if rule.matches(error_type, context):
                matches.append({
                    'rule_id': rule.rule_id,
                    'category': rule.category,
                    'error_type': rule.error_type,
                    'action': rule.action,
                    'priority': rule.priority,
                    'confidence': rule.confidence,
                    'gb_reference': rule.gb_reference,
                    'accuracy': rule.get_accuracy()
                })

        matches.sort(key=lambda x: (x['priority'], x['confidence']), reverse=True)
        return matches

    def _select_best_match(self, case_matches: List[Dict],
                            rule_matches: List[Dict]) -> Optional[Dict]:
        best_case = case_matches[0] if case_matches else None
        best_rule = rule_matches[0] if rule_matches else None

        if best_case is None and best_rule is None:
            return None

        if best_case is None:
            return {'type': 'rule', **best_rule}

        if best_rule is None:
            return {'type': 'case', **best_case}

        case_score = best_case.get('score', 0)
        rule_score = best_rule.get('confidence', 0) * best_rule.get('accuracy', 0.5)

        if case_score >= rule_score:
            return {'type': 'case', **best_case}
        else:
            return {'type': 'rule', **best_rule}
