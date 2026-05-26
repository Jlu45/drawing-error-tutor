import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .atlas_registry import AtlasRegistry
from .atlas_rule_schema import AtlasRule

logger = logging.getLogger("AtlasRulePack")

try:
    from config_loader import ATLAS_RULE_MODE as _RULE_MODE
    RULE_MODE = _RULE_MODE
except ImportError:
    RULE_MODE = 'safe'


class AtlasRulePack:
    def __init__(self, registry: AtlasRegistry, mode: str = ""):
        self.registry = registry
        self.mode = mode or RULE_MODE
        self._active_rules: List[AtlasRule] = []

    def load_rules(self):
        all_rules = self.registry.get_all_rules()
        self._active_rules = [r for r in all_rules if r.enabled]
        self._active_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"[AtlasRulePack] Loaded {len(self._active_rules)} active rules "
                     f"(mode={self.mode})")

    def evaluate(self, error_type: str, context: Dict) -> List[Dict]:
        if not self._active_rules:
            self.load_rules()

        results = []
        for rule in self._active_rules:
            if rule.matches(error_type, context):
                result = {
                    'rule_id': rule.rule_id,
                    'category': rule.category,
                    'error_type': rule.error_type,
                    'action': rule.action,
                    'priority': rule.priority,
                    'confidence': rule.confidence,
                    'gb_reference': rule.gb_reference,
                    'description': rule.description
                }

                if self.mode == 'safe' and rule.confidence < 0.5:
                    continue
                elif self.mode == 'aggressive' and rule.confidence < 0.2:
                    continue

                results.append(result)

        results.sort(key=lambda x: x['priority'], reverse=True)
        return results

    def apply_rules(self, errors: List[Dict], context: Dict) -> List[Dict]:
        if not self._active_rules:
            self.load_rules()

        enhanced_errors = []
        for error in errors:
            error_type = error.get('type', '')
            rule_results = self.evaluate(error_type, context)

            enhanced = dict(error)
            if rule_results:
                best_rule = rule_results[0]
                enhanced['atlas_rule_id'] = best_rule['rule_id']
                enhanced['atlas_confidence'] = best_rule['confidence']
                if best_rule.get('gb_reference'):
                    enhanced['gb_reference'] = best_rule['gb_reference']
                if best_rule['action'].get('suggestion'):
                    enhanced['suggestion'] = best_rule['action']['suggestion']
                if best_rule['action'].get('severity'):
                    enhanced['severity'] = best_rule['action']['severity']

            enhanced_errors.append(enhanced)

        additional_errors = self._find_additional_errors(context)
        for add_error in additional_errors:
            desc = add_error.get('description', '')
            if not any(e.get('description', '') == desc for e in enhanced_errors):
                enhanced_errors.append(add_error)

        return enhanced_errors

    def record_feedback(self, rule_id: str, confirmed: bool):
        rule = self.registry.get_rule(rule_id)
        if rule:
            rule.record_hit(confirmed)
            logger.debug(f"[AtlasRulePack] Rule {rule_id} feedback: "
                          f"{'confirmed' if confirmed else 'dismissed'}")

    def _find_additional_errors(self, context: Dict) -> List[Dict]:
        additional = []
        for rule in self._active_rules:
            if rule.error_type == '*' or rule.error_type == '__any__':
                if rule.matches('*', context):
                    additional.append({
                        'type': rule.category,
                        'description': rule.description,
                        'suggestion': rule.action.get('suggestion', ''),
                        'severity': rule.action.get('severity', '中'),
                        'gb_reference': rule.gb_reference,
                        'source': 'atlas_rule',
                        'atlas_rule_id': rule.rule_id,
                        'atlas_confidence': rule.confidence
                    })
        return additional

    def get_stats(self) -> Dict:
        total = len(self._active_rules)
        by_category = {}
        for rule in self._active_rules:
            by_category[rule.category] = by_category.get(rule.category, 0) + 1
        return {
            'total_active_rules': total,
            'mode': self.mode,
            'by_category': by_category
        }
