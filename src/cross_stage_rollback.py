import copy
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from src.connector_contract import StageName, ContractStatus, StageOutput

logger = logging.getLogger("CrossStageRollback")

try:
    from config_loader import ROLLBACK_MAX_RETRIES as _ROLLBACK_CONFIG
    DEFAULT_MAX_RETRIES = _ROLLBACK_CONFIG
except ImportError:
    DEFAULT_MAX_RETRIES = {
        'ocr': 2,
        'geometry': 2,
        'structure': 1,
        'rule_check': 1,
        'llm': 1
    }


@dataclass
class Checkpoint:
    stage_name: str
    output: StageOutput
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0


@dataclass
class RollbackResult:
    success: bool
    rolled_back_stages: List[str] = field(default_factory=list)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    reason: str = ""


class RollbackStrategy:
    def should_rollback(self, failed_stage: str, error_info: Dict) -> bool:
        raise NotImplementedError

    def get_rollback_stages(self, failed_stage: str) -> List[str]:
        raise NotImplementedError


class ConfidenceBasedStrategy(RollbackStrategy):
    CONFIDENCE_THRESHOLD = 0.3

    def should_rollback(self, failed_stage: str, error_info: Dict) -> bool:
        confidence = error_info.get('confidence', 0.0)
        return confidence < self.CONFIDENCE_THRESHOLD

    def get_rollback_stages(self, failed_stage: str) -> List[str]:
        upstream = {
            'rule_check': ['ocr', 'geometry', 'structure'],
            'llm': ['rule_check', 'ocr', 'geometry', 'structure'],
            'vlm_judge': ['llm', 'rule_check']
        }
        return upstream.get(failed_stage, [])


class ErrorCountStrategy(RollbackStrategy):
    MAX_ERRORS = 5

    def should_rollback(self, failed_stage: str, error_info: Dict) -> bool:
        error_count = error_info.get('error_count', 0)
        return error_count > self.MAX_ERRORS

    def get_rollback_stages(self, failed_stage: str) -> List[str]:
        direct_upstream = {
            'rule_check': ['ocr', 'geometry', 'structure'],
            'llm': ['rule_check'],
            'vlm_judge': ['llm']
        }
        return direct_upstream.get(failed_stage, [])


STAGE_DEPENDENCIES = {
    'ocr': [],
    'geometry': [],
    'structure': [],
    'rule_check': ['ocr', 'geometry', 'structure'],
    'llm': ['ocr', 'geometry', 'structure', 'rule_check'],
    'vlm_judge': ['ocr', 'rule_check', 'llm']
}


class CrossStageRollback:
    def __init__(self, max_retries: Optional[Dict[str, int]] = None):
        self._max_retries = max_retries or dict(DEFAULT_MAX_RETRIES)
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._strategies: List[RollbackStrategy] = [
            ConfidenceBasedStrategy(),
            ErrorCountStrategy()
        ]
        self._rollback_history: List[RollbackResult] = []

    def save_checkpoint(self, stage_name: str, output: StageOutput):
        self._checkpoints[stage_name] = Checkpoint(
            stage_name=stage_name,
            output=copy.deepcopy(output),
            timestamp=time.time(),
            retry_count=self._checkpoints.get(stage_name, Checkpoint(stage_name, output)).retry_count
        )
        logger.debug(f"[Rollback] Checkpoint saved: {stage_name}")

    def should_rollback(self, failed_stage: str, error_info: Dict) -> bool:
        for strategy in self._strategies:
            if strategy.should_rollback(failed_stage, error_info):
                logger.info(f"[Rollback] Strategy {strategy.__class__.__name__} "
                            f"recommends rollback for {failed_stage}")
                return True
        return False

    def execute_rollback(self, failed_stage: str, error_info: Dict) -> RollbackResult:
        rollback_stages = self._get_rollback_targets(failed_stage, error_info)
        if not rollback_stages:
            return RollbackResult(success=False, reason="No rollback targets found")

        retry_counts = {}
        actually_rolled = []

        for stage in rollback_stages:
            checkpoint = self._checkpoints.get(stage)
            if checkpoint is None:
                continue

            max_retries = self._max_retries.get(stage, 1)
            if checkpoint.retry_count >= max_retries:
                logger.warning(f"[Rollback] Stage {stage} exceeded max retries "
                               f"({checkpoint.retry_count}/{max_retries})")
                continue

            checkpoint.retry_count += 1
            checkpoint.output.status = ContractStatus.ROLLED_BACK
            retry_counts[stage] = checkpoint.retry_count
            actually_rolled.append(stage)
            logger.info(f"[Rollback] Rolled back {stage} "
                        f"(retry {checkpoint.retry_count}/{max_retries})")

        result = RollbackResult(
            success=len(actually_rolled) > 0,
            rolled_back_stages=actually_rolled,
            retry_counts=retry_counts,
            reason=f"Rolled back {len(actually_rolled)} stages due to failure in {failed_stage}"
        )
        self._rollback_history.append(result)
        return result

    def get_checkpoint(self, stage_name: str) -> Optional[Checkpoint]:
        return self._checkpoints.get(stage_name)

    def get_stage_output(self, stage_name: str) -> Optional[StageOutput]:
        cp = self._checkpoints.get(stage_name)
        return cp.output if cp else None

    def get_retry_count(self, stage_name: str) -> int:
        cp = self._checkpoints.get(stage_name)
        return cp.retry_count if cp else 0

    def can_retry(self, stage_name: str) -> bool:
        current = self.get_retry_count(stage_name)
        maximum = self._max_retries.get(stage_name, 1)
        return current < maximum

    def clear(self):
        self._checkpoints.clear()
        self._rollback_history.clear()

    def get_history(self) -> List[RollbackResult]:
        return list(self._rollback_history)

    def _get_rollback_targets(self, failed_stage: str, error_info: Dict) -> List[str]:
        targets = set()
        for strategy in self._strategies:
            if strategy.should_rollback(failed_stage, error_info):
                for stage in strategy.get_rollback_stages(failed_stage):
                    if stage in self._checkpoints:
                        targets.add(stage)

        if not targets:
            deps = STAGE_DEPENDENCIES.get(failed_stage, [])
            for dep in deps:
                if dep in self._checkpoints and self.can_retry(dep):
                    targets.add(dep)

        ordered = []
        for stage in ['ocr', 'geometry', 'structure', 'rule_check', 'llm', 'vlm_judge']:
            if stage in targets:
                ordered.append(stage)
        return ordered
