"""
Atlas 契约扩展器 (Phase 0)
============================
在 Planning 阶段，将 atlas 规则追加到 ErrorContract.metadata["atlas_subchecks"]，
使后续 Phase 1/2 可以感知图册驱动的检测需求。
"""

import logging
from typing import Dict, List, Optional

from atlas.atlas_registry import AtlasRegistry

logger = logging.getLogger("AtlasContractExtender")

PRIORITY_FILTERS = {
    "default": {"P0", "P1", "P2"},
    "strict": {"P0", "P1"},
    "lenient": {"P0"},
}


class AtlasContractExtender:
    """Phase 0 契约扩展：将 atlas 规则注入 ErrorContract"""

    def __init__(self, atlas_registry: AtlasRegistry):
        self.registry = atlas_registry

    def extend(
        self,
        contracts: List,
        drawing_type: str = "part",
        teacher_profile: str = "default",
    ) -> List:
        allowed_priorities = PRIORITY_FILTERS.get(teacher_profile, PRIORITY_FILTERS["default"])
        extended_count = 0
        for contract in contracts:
            try:
                v2_cat = AtlasRegistry.v2_category_from_error_category(
                    contract.error_category.value
                )
                matching_rules = self.registry.get_rules_by_v2_category(
                    v2_cat, enabled_only=True, qa_filter=True
                )
                filtered = [
                    r for r in matching_rules if r.get("priority", "P2") in allowed_priorities
                ]
                if not filtered:
                    continue
                metadata = self._get_metadata(contract)
                subchecks = []
                for rule in filtered:
                    subcheck = {
                        "rule_id": rule.get("rule_id", ""),
                        "priority": rule.get("priority", "P2"),
                        "source_case_id": rule.get("source_case_id", ""),
                        "check_type": rule.get("check_type", ""),
                    }
                    subchecks.append(subcheck)
                metadata["atlas_subchecks"] = subchecks
                extended_count += len(subchecks)
            except Exception as e:
                logger.error(
                    f"[AtlasContractExtender] 扩展契约 {getattr(contract, 'contract_id', '?')} "
                    f"失败: {e}"
                )
        logger.info(
            f"[AtlasContractExtender] 扩展完成: {extended_count} 条atlas子检查 "
            f"注入到 {len(contracts)} 个契约 (profile={teacher_profile})"
        )
        return contracts

    def _get_metadata(self, contract) -> Dict:
        if hasattr(contract, "metadata") and isinstance(contract.metadata, dict):
            return contract.metadata
        if not hasattr(contract, "metadata"):
            try:
                contract.metadata = {}
            except (AttributeError, TypeError):
                return {}
        if not isinstance(contract.metadata, dict):
            try:
                contract.metadata = {}
            except (AttributeError, TypeError):
                return {}
        return contract.metadata
