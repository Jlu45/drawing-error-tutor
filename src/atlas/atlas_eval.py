import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .atlas_registry import AtlasRegistry
from .atlas_case_schema import AtlasCase
from .atlas_rule_schema import AtlasRule

logger = logging.getLogger("AtlasEval")

try:
    from config_loader import ATLAS_EVAL_PATH as _EVAL_PATH
    EVAL_PATH = _EVAL_PATH
except ImportError:
    EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'atlas', 'atlas_eval_cases.jsonl')


@dataclass
class EvalCase:
    eval_id: str
    image_description: str
    expected_errors: List[Dict]
    expected_score_range: Tuple[int, int] = (0, 100)
    category: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvalResult:
    eval_id: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    score_in_range: bool = False
    details: Dict = field(default_factory=dict)


class AtlasEvaluator:
    def __init__(self, registry: AtlasRegistry, eval_path: str = ""):
        self.registry = registry
        self.eval_path = eval_path or EVAL_PATH
        self._eval_cases: List[EvalCase] = []
        self._load_eval_cases()

    def evaluate(self, analysis_result: Dict, eval_case: Optional[EvalCase] = None) -> EvalResult:
        if eval_case is None:
            if not self._eval_cases:
                return EvalResult(eval_id="no_eval", precision=0.0, recall=0.0, f1=0.0)
            eval_case = self._eval_cases[0]

        detected_errors = analysis_result.get('errors', [])
        expected_errors = eval_case.expected_errors

        detected_set = set()
        for e in detected_errors:
            key = f"{e.get('type', '')}:{e.get('description', '')[:30]}"
            detected_set.add(key)

        expected_set = set()
        for e in expected_errors:
            key = f"{e.get('type', '')}:{e.get('description', '')[:30]}"
            expected_set.add(key)

        true_positives = len(detected_set & expected_set)
        false_positives = len(detected_set - expected_set)
        false_negatives = len(expected_set - detected_set)

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        report = analysis_result.get('report', {})
        score = report.get('overall_score', 0)
        score_in_range = eval_case.expected_score_range[0] <= score <= eval_case.expected_score_range[1]

        return EvalResult(
            eval_id=eval_case.eval_id,
            precision=round(precision, 3),
            recall=round(recall, 3),
            f1=round(f1, 3),
            score_in_range=score_in_range,
            details={
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'detected_count': len(detected_set),
                'expected_count': len(expected_set)
            }
        )

    def evaluate_all(self, analysis_results: Dict[str, Dict]) -> Dict[str, EvalResult]:
        results = {}
        for eval_case in self._eval_cases:
            result = analysis_results.get(eval_case.eval_id)
            if result:
                results[eval_case.eval_id] = self.evaluate(result, eval_case)
        return results

    def evaluate_rule_accuracy(self) -> Dict[str, Dict]:
        all_rules = self.registry.get_all_rules()
        results = {}
        for rule in all_rules:
            accuracy = rule.get_accuracy()
            results[rule.rule_id] = {
                'rule_id': rule.rule_id,
                'category': rule.category,
                'error_type': rule.error_type,
                'accuracy': round(accuracy, 3),
                'hit_count': rule.hit_count,
                'confirm_count': rule.confirm_count,
                'dismiss_count': rule.dismiss_count,
                'confidence': rule.confidence
            }
        return results

    def get_summary(self, results: Dict[str, EvalResult]) -> Dict:
        if not results:
            return {'total': 0}

        precisions = [r.precision for r in results.values()]
        recalls = [r.recall for r in results.values()]
        f1s = [r.f1 for r in results.values()]
        score_accuracy = sum(1 for r in results.values() if r.score_in_range) / len(results)

        return {
            'total': len(results),
            'avg_precision': round(sum(precisions) / len(precisions), 3),
            'avg_recall': round(sum(recalls) / len(recalls), 3),
            'avg_f1': round(sum(f1s) / len(f1s), 3),
            'score_accuracy': round(score_accuracy, 3)
        }

    def _load_eval_cases(self):
        if os.path.exists(self.eval_path):
            try:
                with open(self.eval_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            eval_case = EvalCase(
                                eval_id=data.get('eval_id', ''),
                                image_description=data.get('image_description', ''),
                                expected_errors=data.get('expected_errors', []),
                                expected_score_range=tuple(data.get('expected_score_range', [0, 100])),
                                category=data.get('category', ''),
                                metadata=data.get('metadata', {})
                            )
                            self._eval_cases.append(eval_case)
                logger.info(f"[AtlasEval] Loaded {len(self._eval_cases)} eval cases")
            except Exception as e:
                logger.warning(f"[AtlasEval] Load failed: {e}")
