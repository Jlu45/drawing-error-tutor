"""
跨阶段回滚控制器 (Cross-Stage Rollback Controller)
====================================================
借鉴 ArtiCAD 的 Cross-Stage Rollback Mechanism。

核心思想：当检测/分析失败时，不是全流程重试，而是：
1. 精确分类错误类型（DETECTION / ANALYSIS）
2. 定向路由到负责的Agent
3. 保留无故障中间结果，仅重新执行受影响部分

错误分类：
- DETECTION 错误：检测器本身失败（OCR未识别、几何检测异常）
  → 回滚到对应感知Agent，调整参数后重试
- ANALYSIS 错误：分析阶段误判（规则检查假阳性、LLM幻觉）
  → 回滚到RuleCheck/LLM Agent，用经验库修正
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger("RollbackController")


class ErrorType(Enum):
    """错误类型分类"""
    DETECTION = "detection"    # 检测器失败
    ANALYSIS = "analysis"      # 分析误判
    INFRASTRUCTURE = "infra"   # 基础设施错误（网络、API等）


class RollbackAction(Enum):
    """回滚动作"""
    RETRY_AGENT = "retry_agent"           # 重试单个Agent
    ADJUST_PARAMS = "adjust_params"       # 调整参数后重试
    USE_FALLBACK = "use_fallback"         # 使用降级方案
    SKIP = "skip"                         # 跳过该步骤
    FULL_RETRY = "full_retry"             # 全流程重试（最后手段）


@dataclass
class RollbackDecision:
    """回滚决策"""
    error_type: ErrorType
    source_agent: str
    error_message: str
    action: RollbackAction
    target_agent: str = ""           # 回滚目标Agent
    adjusted_params: Dict = field(default_factory=dict)
    keep_results: List[str] = field(default_factory=list)   # 保留的中间结果
    regenerate_results: List[str] = field(default_factory=list)  # 需要重新生成的结果
    reason: str = ""
    max_retries: int = 1


@dataclass
class IntermediateResult:
    """中间结果快照，用于回滚时恢复"""
    agent_name: str
    data: Dict
    timestamp: float = field(default_factory=time.time)
    version: int = 1
    is_valid: bool = True


class ResultSnapshot:
    """
    中间结果快照管理器

    在流水线每个阶段完成后保存快照，支持回滚时恢复到任意阶段。
    类比 ArtiCAD 中将零件分为 keep/regenerate/newly introduced 三组。
    """

    def __init__(self, max_snapshots: int = 20):
        self._snapshots: Dict[str, List[IntermediateResult]] = {}
        self._max_snapshots = max_snapshots

    def save(self, agent_name: str, data: Dict) -> int:
        """保存一个中间结果快照"""
        if agent_name not in self._snapshots:
            self._snapshots[agent_name] = []
        version = len(self._snapshots[agent_name]) + 1
        snapshot = IntermediateResult(
            agent_name=agent_name,
            data=data,
            version=version
        )
        self._snapshots[agent_name].append(snapshot)

        # 限制快照数量
        if len(self._snapshots[agent_name]) > self._max_snapshots:
            self._snapshots[agent_name] = self._snapshots[agent_name][-self._max_snapshots:]

        logger.debug(f"[快照] 保存 {agent_name} v{version}")
        return version

    def restore(self, agent_name: str, version: int = -1) -> Optional[Dict]:
        """恢复中间结果"""
        snapshots = self._snapshots.get(agent_name, [])
        if not snapshots:
            return None
        snapshot = snapshots[version]
        logger.debug(f"[快照] 恢复 {agent_name} v{snapshot.version}")
        return snapshot.data

    def get_latest(self, agent_name: str) -> Optional[Dict]:
        """获取最新快照"""
        return self.restore(agent_name, -1)

    def invalidate(self, agent_name: str):
        """标记某个Agent的结果为无效"""
        snapshots = self._snapshots.get(agent_name, [])
        for s in snapshots:
            s.is_valid = False

    def get_valid_results(self) -> Dict[str, Dict]:
        """获取所有有效的中间结果"""
        valid = {}
        for agent_name, snapshots in self._snapshots.items():
            for s in reversed(snapshots):
                if s.is_valid:
                    valid[agent_name] = s.data
                    break
        return valid

    def classify_results(self, failed_agents: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        将中间结果分为三组：keep / regenerate / adjust

        借鉴 ArtiCAD 的定向修复策略：
        - keep: 无故障组件，直接复用
        - regenerate: 需要完全重新生成
        - adjust: 需要调整参数后重新执行

        Returns:
            (keep_agents, regenerate_agents, adjust_agents)
        """
        keep = []
        regenerate = []
        adjust = []

        all_agents = list(self._snapshots.keys())
        for agent in all_agents:
            snapshots = self._snapshots[agent]
            has_valid = any(s.is_valid for s in snapshots)
            if agent in failed_agents:
                if has_valid:
                    adjust.append(agent)  # 有历史有效结果，可尝试调整参数
                else:
                    regenerate.append(agent)  # 从未成功过，需要完全重新生成
            else:
                keep.append(agent)

        logger.info(f"[快照分类] keep={keep}, regenerate={regenerate}, adjust={adjust}")
        return keep, regenerate, adjust

    def clear(self):
        """清空所有快照"""
        self._snapshots.clear()


class RollbackController:
    """
    跨阶段回滚控制器

    负责：
    1. 错误分类（DETECTION / ANALYSIS / INFRASTRUCTURE）
    2. 回滚决策（重试/调参/降级/跳过/全量重试）
    3. 中间结果管理（保存/恢复/分类）
    4. 回滚执行（定向修复）
    """

    # Agent失败时的默认处理策略
    DEFAULT_STRATEGIES = {
        'ocr': {
            'error_type': ErrorType.DETECTION,
            'primary_action': RollbackAction.ADJUST_PARAMS,
            'fallback_action': RollbackAction.USE_FALLBACK,
            'adjustable_params': {
                'preprocess_mode': ['ocr', 'ocr_aggressive', 'ocr_conservative'],
                'threshold': [0.5, 0.3, 0.7]
            },
            'max_retries': 2
        },
        'geometry': {
            'error_type': ErrorType.DETECTION,
            'primary_action': RollbackAction.ADJUST_PARAMS,
            'fallback_action': RollbackAction.USE_FALLBACK,
            'adjustable_params': {
                'canny_low': [50, 30, 70],
                'canny_high': [150, 100, 200],
                'hough_threshold': [80, 50, 100]
            },
            'max_retries': 2
        },
        'structure': {
            'error_type': ErrorType.DETECTION,
            'primary_action': RollbackAction.RETRY_AGENT,
            'fallback_action': RollbackAction.USE_FALLBACK,
            'max_retries': 1
        },
        'rule_check': {
            'error_type': ErrorType.ANALYSIS,
            'primary_action': RollbackAction.ADJUST_PARAMS,
            'fallback_action': RollbackAction.SKIP,
            'adjustable_params': {
                'confidence_threshold': [0.3, 0.5, 0.1]
            },
            'max_retries': 1
        },
        'llm': {
            'error_type': ErrorType.ANALYSIS,
            'primary_action': RollbackAction.USE_FALLBACK,
            'fallback_action': RollbackAction.SKIP,
            'max_retries': 1
        }
    }

    def __init__(self):
        self.snapshot = ResultSnapshot()
        self._retry_counts: Dict[str, int] = {}
        self._rollback_history: List[RollbackDecision] = []
        self._total_rollbacks = 0

    def save_result(self, agent_name: str, data: Dict):
        """保存Agent执行结果快照"""
        self.snapshot.save(agent_name, data)
        # 成功执行后重置重试计数
        self._retry_counts[agent_name] = 0

    def classify_error(self, agent_name: str, error: Exception) -> ErrorType:
        """
        分类错误类型

        DETECTION: 检测器本身的技术故障
        ANALYSIS: 分析结果的语义错误
        INFRASTRUCTURE: 外部依赖故障
        """
        error_msg = str(error).lower()

        # 基础设施错误
        infra_keywords = ['timeout', 'connection', 'network', 'api', 'key', 'auth',
                         'rate limit', '500', '502', '503', '504']
        if any(kw in error_msg for kw in infra_keywords):
            return ErrorType.INFRASTRUCTURE

        # 根据Agent类型分类
        strategy = self.DEFAULT_STRATEGIES.get(agent_name, {})
        if strategy:
            return strategy['error_type']

        # 默认分类
        if agent_name in ('ocr', 'geometry', 'structure'):
            return ErrorType.DETECTION
        return ErrorType.ANALYSIS

    def decide_rollback(self, agent_name: str, error: Exception) -> RollbackDecision:
        """
        做出回滚决策

        借鉴 ArtiCAD 的两级错误分类和定向修复：
        1. 分类错误类型
        2. 查找对应策略
        3. 确定保留/重新生成/调整的组件
        4. 生成回滚决策
        """
        error_type = self.classify_error(agent_name, error)
        strategy = self.DEFAULT_STRATEGIES.get(agent_name, {})
        retry_count = self._retry_counts.get(agent_name, 0)

        # 确定回滚动作
        if retry_count >= strategy.get('max_retries', 1):
            action = strategy.get('fallback_action', RollbackAction.SKIP)
            reason = f"已达最大重试次数 ({retry_count})"
        elif error_type == ErrorType.INFRASTRUCTURE:
            action = RollbackAction.USE_FALLBACK
            reason = "基础设施错误，使用降级方案"
        else:
            action = strategy.get('primary_action', RollbackAction.RETRY_AGENT)
            reason = f"{error_type.value}错误，执行{action.value}"

        # 分类中间结果
        keep, regenerate, adjust = self.snapshot.classify_results([agent_name])
        if action == RollbackAction.ADJUST_PARAMS:
            adjust.append(agent_name)
            if agent_name in regenerate:
                regenerate.remove(agent_name)
        elif action == RollbackAction.RETRY_AGENT:
            regenerate.append(agent_name)
            if agent_name in adjust:
                adjust.remove(agent_name)

        # 生成调整参数
        adjusted_params = {}
        if action == RollbackAction.ADJUST_PARAMS and 'adjustable_params' in strategy:
            params = strategy['adjustable_params']
            for param_name, values in params.items():
                idx = retry_count % len(values)
                adjusted_params[param_name] = values[idx]

        decision = RollbackDecision(
            error_type=error_type,
            source_agent=agent_name,
            error_message=str(error),
            action=action,
            target_agent=agent_name,
            adjusted_params=adjusted_params,
            keep_results=keep,
            regenerate_results=regenerate,
            reason=reason,
            max_retries=strategy.get('max_retries', 1)
        )

        self._retry_counts[agent_name] = retry_count + 1
        self._rollback_history.append(decision)
        self._total_rollbacks += 1

        logger.info(f"[回滚] {agent_name}: {action.value} (第{retry_count+1}次) "
                    f"原因: {reason}")
        logger.info(f"[回滚] keep={keep}, regenerate={regenerate}, adjust={adjust}")

        return decision

    def get_valid_results(self) -> Dict[str, Dict]:
        """获取所有有效的中间结果"""
        return self.snapshot.get_valid_results()

    def get_retry_count(self, agent_name: str) -> int:
        return self._retry_counts.get(agent_name, 0)

    @property
    def stats(self) -> Dict:
        return {
            'total_rollbacks': self._total_rollbacks,
            'retry_counts': dict(self._retry_counts),
            'history': [
                {
                    'agent': d.source_agent,
                    'type': d.error_type.value,
                    'action': d.action.value,
                    'reason': d.reason
                }
                for d in self._rollback_history[-10:]  # 最近10条
            ]
        }

    def reset(self):
        """重置回滚控制器状态"""
        self.snapshot.clear()
        self._retry_counts.clear()
        self._rollback_history.clear()
        self._total_rollbacks = 0
