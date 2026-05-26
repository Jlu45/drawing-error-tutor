import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class AtlasRule:
    rule_id: str = ""
    category: str = ""
    error_type: str = ""
    condition: Dict = field(default_factory=dict)
    action: Dict = field(default_factory=dict)
    priority: int = 5
    confidence: float = 0.5
    gb_reference: str = ""
    description: str = ""
    enabled: bool = True
    hit_count: int = 0
    confirm_count: int = 0
    dismiss_count: int = 0
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'AtlasRule':
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def create(cls, category: str, error_type: str, condition: Dict,
               action: Dict, gb_reference: str = "",
               description: str = "", priority: int = 5,
               tags: Optional[List[str]] = None) -> 'AtlasRule':
        key_parts = [category, error_type, str(condition)]
        rule_id = hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
        return cls(
            rule_id=rule_id,
            category=category,
            error_type=error_type,
            condition=condition,
            action=action,
            priority=priority,
            gb_reference=gb_reference,
            description=description,
            tags=tags or [],
            timestamp=time.time()
        )

    def record_hit(self, confirmed: bool):
        self.hit_count += 1
        if confirmed:
            self.confirm_count += 1
            self.confidence = min(1.0, self.confidence + 0.05)
        else:
            self.dismiss_count += 1
            self.confidence = max(0.0, self.confidence - 0.1)

    def matches(self, error_type: str, context: Dict) -> bool:
        if not self.enabled:
            return False
        if self.error_type != error_type and error_type not in self.tags:
            return False
        for key, value in self.condition.items():
            if key in context:
                if isinstance(value, (list, tuple)):
                    if context[key] not in value:
                        return False
                elif context[key] != value:
                    return False
        return True

    def get_accuracy(self) -> float:
        total = self.confirm_count + self.dismiss_count
        if total == 0:
            return 0.5
        return self.confirm_count / total
