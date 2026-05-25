"""
Atlas 图册纠错模块
==================
基于《机械制图常见错误经典示例图册》的规则驱动纠错引擎。

核心类：
- AtlasRegistry: 图册案例/规则统一加载与索引
- AtlasContractExtender: Phase 0 契约扩展
- AtlasFeatureAdapter: Phase 1 特征适配
- AtlasRulePlugin: Phase 2 规则插件
- AtlasContextRetriever: Phase 3 LLM 图册上下文检索
- AtlasVLMFewshotProvider: Phase 5 VLM Judge few-shot 提供器
- AtlasEvidenceFusion: Phase 4 结果融合器
"""

from atlas.atlas_registry import AtlasRegistry
from atlas.atlas_contract_extender import AtlasContractExtender
from atlas.atlas_feature_adapter import AtlasFeatureAdapter
from atlas.atlas_rule_plugin import AtlasRulePlugin
from atlas.atlas_context_retriever import AtlasContextRetriever
from atlas.atlas_vlm_fewshot import AtlasVLMFewshotProvider
from atlas.atlas_fusion import AtlasEvidenceFusion

__all__ = [
    "AtlasRegistry",
    "AtlasContractExtender",
    "AtlasFeatureAdapter",
    "AtlasRulePlugin",
    "AtlasContextRetriever",
    "AtlasVLMFewshotProvider",
    "AtlasEvidenceFusion",
]
