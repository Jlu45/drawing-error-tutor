import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AtlasCase:
    case_id: str = ""
    category: str = ""
    sub_category: str = ""
    error_type: str = ""
    description: str = ""
    drawing_features: Dict = field(default_factory=dict)
    errors_found: List[Dict] = field(default_factory=list)
    corrections: List[Dict] = field(default_factory=list)
    gb_references: List[str] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)
    confidence: float = 0.0
    image_hash: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'AtlasCase':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def create(cls, category: str, error_type: str, description: str,
               errors_found: List[Dict], corrections: List[Dict],
               gb_references: Optional[List[str]] = None,
               drawing_features: Optional[Dict] = None,
               tags: Optional[List[str]] = None) -> 'AtlasCase':
        key_parts = [category, error_type, description[:50]]
        case_id = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
        return cls(
            case_id=case_id,
            category=category,
            error_type=error_type,
            description=description,
            errors_found=errors_found,
            corrections=corrections,
            gb_references=gb_references or [],
            drawing_features=drawing_features or {},
            tags=tags or [],
            timestamp=time.time()
        )

    def add_feedback(self, feedback_type: str):
        self.feedback_history.append(feedback_type)
        if feedback_type == 'confirmed':
            self.confidence = min(1.0, self.confidence + 0.1)
        elif feedback_type == 'dismissed':
            self.confidence = max(0.0, self.confidence - 0.2)

    def matches_error_type(self, error_type: str) -> bool:
        return self.error_type == error_type or error_type in self.tags

    def get_positive_feedback_ratio(self) -> float:
        if not self.feedback_history:
            return 0.5
        positive = sum(1 for f in self.feedback_history if f in ('confirmed', 'useful_guidance'))
        return positive / len(self.feedback_history)
