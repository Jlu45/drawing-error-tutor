import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("ConnectorContract")


class StageName(str, Enum):
    OCR = "ocr"
    GEOMETRY = "geometry"
    STRUCTURE = "structure"
    RULE_CHECK = "rule_check"
    LLM = "llm"
    VLM_JUDGE = "vlm_judge"


class ContractStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class StageInput:
    stage_name: StageName
    image_path: str = ""
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.kwargs.get(key, default)


@dataclass
class StageOutput:
    stage_name: StageName
    status: ContractStatus
    data: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    confidence: float = 0.0
    execution_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConnectorContract:
    source_stage: StageName
    target_stage: StageName
    required_outputs: List[str] = field(default_factory=list)
    optional_outputs: List[str] = field(default_factory=list)
    transform_fn: Optional[str] = None

    def validate_output(self, output: StageOutput) -> bool:
        if output.status != ContractStatus.COMPLETED:
            return False
        for key in self.required_outputs:
            if key not in output.data:
                logger.warning(f"[Contract] Missing required output '{key}' "
                               f"from {output.stage_name.value}")
                return False
        return True


STAGE_CONTRACTS = {
    ("ocr", "rule_check"): ConnectorContract(
        source_stage=StageName.OCR,
        target_stage=StageName.RULE_CHECK,
        required_outputs=["texts", "total_count"],
        optional_outputs=["high_confidence_count"]
    ),
    ("geometry", "rule_check"): ConnectorContract(
        source_stage=StageName.GEOMETRY,
        target_stage=StageName.RULE_CHECK,
        required_outputs=["lines", "circles"],
        optional_outputs=["arrows", "line_types", "dimension_structures", "contours"]
    ),
    ("structure", "rule_check"): ConnectorContract(
        source_stage=StageName.STRUCTURE,
        target_stage=StageName.RULE_CHECK,
        required_outputs=["image_size"],
        optional_outputs=["title_block", "view_areas", "has_border", "regions"]
    ),
    ("ocr", "llm"): ConnectorContract(
        source_stage=StageName.OCR,
        target_stage=StageName.LLM,
        required_outputs=["texts"],
        optional_outputs=["total_count", "high_confidence_count"]
    ),
    ("geometry", "llm"): ConnectorContract(
        source_stage=StageName.GEOMETRY,
        target_stage=StageName.LLM,
        required_outputs=["lines", "circles"],
        optional_outputs=["arrows", "line_types", "dimension_structures"]
    ),
    ("structure", "llm"): ConnectorContract(
        source_stage=StageName.STRUCTURE,
        target_stage=StageName.LLM,
        required_outputs=["image_size"],
        optional_outputs=["title_block", "view_areas", "has_border"]
    ),
    ("rule_check", "llm"): ConnectorContract(
        source_stage=StageName.RULE_CHECK,
        target_stage=StageName.LLM,
        required_outputs=["errors"],
        optional_outputs=["total_errors", "high_severity", "medium_severity", "low_severity"]
    ),
    ("ocr", "vlm_judge"): ConnectorContract(
        source_stage=StageName.OCR,
        target_stage=StageName.VLM_JUDGE,
        required_outputs=["texts"],
        optional_outputs=[]
    ),
    ("rule_check", "vlm_judge"): ConnectorContract(
        source_stage=StageName.RULE_CHECK,
        target_stage=StageName.VLM_JUDGE,
        required_outputs=["errors"],
        optional_outputs=["total_errors"]
    ),
    ("llm", "vlm_judge"): ConnectorContract(
        source_stage=StageName.LLM,
        target_stage=StageName.VLM_JUDGE,
        required_outputs=["raw_response"],
        optional_outputs=["model", "usage"]
    ),
}


class ContractValidator:
    def __init__(self):
        self._contracts = dict(STAGE_CONTRACTS)

    def validate(self, source: str, target: str, output: StageOutput) -> bool:
        contract_key = (source, target)
        contract = self._contracts.get(contract_key)
        if contract is None:
            logger.debug(f"[ContractValidator] No contract for {source}->{target}, skipping")
            return True
        return contract.validate_output(output)

    def validate_all(self, outputs: Dict[str, StageOutput], target: str) -> Dict[str, bool]:
        results = {}
        for source_name, output in outputs.items():
            results[source_name] = self.validate(source_name, target, output)
        return results

    def get_missing_fields(self, source: str, target: str, output: StageOutput) -> List[str]:
        contract_key = (source, target)
        contract = self._contracts.get(contract_key)
        if contract is None:
            return []
        missing = []
        for key in contract.required_outputs:
            if key not in output.data:
                missing.append(key)
        return missing
