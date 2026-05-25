"""
检测预检系统 (Pre-check Protocol System)
====================================
借鉴 ArtiCAD 的 Connector Contract 设计思想。

核心创新：在感知阶段之前，由 Planning Agent 定义"检测预检单"，
明确每个检测器的职责、输出格式和依赖关系，使各检测器条件独立。

数学保证（马尔可夫毯）：
    P(D_1, ..., D_N | C) = ∏ P(D_i | c_i)
    其中 D_i 是第i个检测器的结果，c_i 是其对应的预检项。
"""

import uuid
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger("ErrorContract")


class ErrorCategory(Enum):
    """错误类别枚举，对应GB标准的不同检查维度"""
    DIMENSION = "尺寸标注"       # GB/T 4458.4
    LINE_TYPE = "线型"           # GB/T 4457.4
    TOLERANCE = "公差"           # GB/T 1800.1
    TITLE_BLOCK = "标题栏"       # GB/T 10609.1
    SYMBOL = "符号"              # GB/T 131
    GEOMETRY = "几何完整性"       # 综合判断
    STRUCTURE = "图纸结构"        # GB/T 14665
    SURFACE = "表面粗糙度"       # GB/T 131
    WELD = "焊接符号"            # GB/T 324
    SHEET = "图幅规范"           # GB/T 14689
    VIEW = "视图标注"            # GB/T 17451
    GENERAL = "其他"


class ContractStatus(Enum):
    """预检执行状态"""
    PENDING = "pending"           # 待执行
    IN_PROGRESS = "in_progress"   # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"             # 执行失败
    SKIPPED = "skipped"           # 跳过（如不适用）
    DEGRADED = "degraded"         # 降级执行


@dataclass
class ErrorContract:
    """
    检测预检单 —— 类比 ArtiCAD 的 Connector Contract

    在检测开始前定义好每个检测器的"接口预检"：
    - 检测什么（scope）
    - 输出什么格式（output_schema）
    - 依赖什么（dependencies，通常为空表示条件独立）
    - 对应什么国标（gb_reference）
    """
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    detector_name: str = ""
    error_category: ErrorCategory = ErrorCategory.GENERAL
    detection_scope: List[str] = field(default_factory=list)
    output_schema: Dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他预检项ID
    priority: int = 5              # 1-10，数字越小优先级越高
    gb_reference: str = ""
    description: str = ""
    status: ContractStatus = ContractStatus.PENDING
    result: Optional[Dict] = None
    execution_time_ms: float = 0.0
    confidence: float = 0.0
    error_message: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['error_category'] = self.error_category.value
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ErrorContract':
        d = d.copy()
        if 'error_category' in d and isinstance(d['error_category'], str):
            d['error_category'] = ErrorCategory(d['error_category'])
        if 'status' in d and isinstance(d['status'], str):
            d['status'] = ContractStatus(d['status'])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def mark_completed(self, result: Dict, confidence: float = 1.0, exec_time: float = 0.0):
        self.status = ContractStatus.COMPLETED
        self.result = result
        self.confidence = confidence
        self.execution_time_ms = exec_time

    def mark_failed(self, error_msg: str):
        self.status = ContractStatus.FAILED
        self.error_message = error_msg

    def mark_skipped(self, reason: str = ""):
        self.status = ContractStatus.SKIPPED
        self.error_message = reason

    def mark_degraded(self, result: Dict, confidence: float = 0.3):
        self.status = ContractStatus.DEGRADED
        self.result = result
        self.confidence = confidence


class ContractRegistry:
    """
    预检注册表 —— 管理所有检测预检单的生命周期

    功能：
    1. 注册/注销预检项
    2. 拓扑排序确定执行顺序（处理依赖关系）
    3. 查询预检状态
    4. 验证预检完整性
    """

    def __init__(self):
        self._contracts: Dict[str, ErrorContract] = {}
        self._execution_order: List[str] = []

    def register(self, contract: ErrorContract) -> str:
        """注册一个新预检项"""
        if contract.contract_id in self._contracts:
            logger.warning(f"预检项 {contract.contract_id} 已存在，将被覆盖")
        self._contracts[contract.contract_id] = contract
        self._execution_order = self._topological_sort()
        logger.info(f"[预检注册] {contract.contract_id} ({contract.detector_name}) "
                    f"→ {contract.error_category.value}, 优先级={contract.priority}")
        return contract.contract_id

    def unregister(self, contract_id: str):
        """注销预检项"""
        if contract_id in self._contracts:
            del self._contracts[contract_id]
            self._execution_order = self._topological_sort()

    def get(self, contract_id: str) -> Optional[ErrorContract]:
        return self._contracts.get(contract_id)

    def get_by_detector(self, detector_name: str) -> List[ErrorContract]:
        return [c for c in self._contracts.values() if c.detector_name == detector_name]

    def get_by_category(self, category: ErrorCategory) -> List[ErrorContract]:
        return [c for c in self._contracts.values() if c.error_category == category]

    def get_pending(self) -> List[ErrorContract]:
        return [c for c in self._contracts.values() if c.status == ContractStatus.PENDING]

    def get_failed(self) -> List[ErrorContract]:
        return [c for c in self._contracts.values() if c.status == ContractStatus.FAILED]

    def get_completed(self) -> List[ErrorContract]:
        return [c for c in self._contracts.values() if c.status == ContractStatus.COMPLETED]

    @property
    def execution_order(self) -> List[str]:
        """返回拓扑排序后的预检执行顺序"""
        return self._execution_order

    @property
    def all_contracts(self) -> List[ErrorContract]:
        return list(self._contracts.values())

    @property
    def stats(self) -> Dict:
        status_counts = {}
        for c in self._contracts.values():
            s = c.status.value
            status_counts[s] = status_counts.get(s, 0) + 1
        return {
            'total': len(self._contracts),
            'status_counts': status_counts,
            'avg_confidence': sum(c.confidence for c in self._contracts.values()
                                  if c.status == ContractStatus.COMPLETED) /
                               max(1, len(self.get_completed())),
            'total_exec_time_ms': sum(c.execution_time_ms for c in self._contracts.values())
        }

    def _topological_sort(self) -> List[str]:
        """
        拓扑排序：根据依赖关系确定预检执行顺序
        无依赖的预检项可并行执行（同一层级）
        """
        # 构建邻接表
        in_degree = {cid: 0 for cid in self._contracts}
        graph = {cid: [] for cid in self._contracts}

        for cid, contract in self._contracts.items():
            for dep_id in contract.dependencies:
                if dep_id in self._contracts:
                    graph[dep_id].append(cid)
                    in_degree[cid] += 1

        # Kahn算法
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        queue.sort(key=lambda cid: self._contracts[cid].priority)  # 同级按优先级排序
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    queue.sort(key=lambda cid: self._contracts[cid].priority)

        if len(order) != len(self._contracts):
            logger.warning("[预检注册表] 检测到循环依赖！部分预检项可能无法执行")
            # 添加未被排序的预检项
            for cid in self._contracts:
                if cid not in order:
                    order.append(cid)

        return order

    def validate(self) -> List[str]:
        """验证所有预检项的完整性"""
        issues = []
        for cid, contract in self._contracts.items():
            # 检查依赖是否存在
            for dep_id in contract.dependencies:
                if dep_id not in self._contracts:
                    issues.append(f"预检项 {cid} 依赖不存在的预检项 {dep_id}")
            # 检查输出格式是否定义
            if not contract.output_schema:
                issues.append(f"预检项 {cid} 未定义输出格式 (output_schema)")
            # 检查检测范围是否为空
            if not contract.detection_scope:
                issues.append(f"预检项 {cid} 检测范围 (detection_scope) 为空")
        return issues

    def to_dict(self) -> Dict:
        return {
            'contracts': [c.to_dict() for c in self._contracts.values()],
            'execution_order': self._execution_order,
            'stats': self.stats
        }


class ContractFactory:
    """
    预检工厂 —— 根据图纸类型和扫描结果动态生成预检集

    借鉴 ArtiCAD 的 Design Agent 思想：
    在检测前先做"检测规划"，根据图纸特征动态调整检测策略。
    """

    # 零件图标准预检模板
    PART_DRAWING_CONTRACTS = [
        {
            'detector_name': 'ocr',
            'error_category': ErrorCategory.TITLE_BLOCK,
            'detection_scope': ['标题栏文字识别', '图名提取', '比例提取', '材料标注提取'],
            'output_schema': {
                'title_texts': 'List[Dict]', 'drawing_name': 'str',
                'scale': 'str', 'material': 'str'
            },
            'gb_reference': 'GB/T 10609.1',
            'description': '识别标题栏中的关键信息',
            'priority': 1
        },
        {
            'detector_name': 'ocr',
            'error_category': ErrorCategory.DIMENSION,
            'detection_scope': ['尺寸数值识别', '公差标注识别', '表面粗糙度数值识别'],
            'output_schema': {
                'dimension_texts': 'List[Dict]', 'tolerance_texts': 'List[Dict]',
                'surface_roughness': 'List[Dict]'
            },
            'gb_reference': 'GB/T 4458.4',
            'description': '识别尺寸标注相关文字',
            'priority': 2
        },
        {
            'detector_name': 'geometry',
            'error_category': ErrorCategory.LINE_TYPE,
            'detection_scope': [
                '实线检测', '虚线检测', '点画线检测',
                '线宽分类', '线型连续性检查'
            ],
            'output_schema': {
                'line_types': 'Dict', 'solid_lines': 'List', 'dashed_lines': 'List',
                'center_lines': 'List', 'width_distribution': 'Dict'
            },
            'gb_reference': 'GB/T 4457.4',
            'description': '检测各类线型及其使用规范',
            'priority': 2
        },
        {
            'detector_name': 'geometry',
            'error_category': ErrorCategory.GEOMETRY,
            'detection_scope': [
                '直线检测', '圆/圆弧检测', '箭头检测',
                '尺寸线结构检测', '轮廓完整性检测'
            ],
            'output_schema': {
                'lines': 'List', 'circles': 'List', 'arrows': 'List',
                'dimension_structures': 'List', 'contours': 'List'
            },
            'gb_reference': '综合',
            'description': '检测几何元素的完整性和规范性',
            'priority': 3
        },
        {
            'detector_name': 'structure',
            'error_category': ErrorCategory.STRUCTURE,
            'detection_scope': [
                '图框检测', '标题栏区域定位', '视图区域分割',
                '比例一致性检查'
            ],
            'output_schema': {
                'has_border': 'bool', 'title_block': 'Dict',
                'view_areas': 'List', 'image_size': 'Dict'
            },
            'gb_reference': 'GB/T 14665',
            'description': '分析图纸整体结构和布局',
            'priority': 1
        },
        {
            'detector_name': 'rule_check',
            'error_category': ErrorCategory.DIMENSION,
            'detection_scope': [
                '尺寸标注完整性', '尺寸线终端形式', '尺寸数值合理性',
                'Φ/R符号使用', '重复尺寸检查'
            ],
            'output_schema': {
                'errors': 'List[Dict]', 'total_errors': 'int',
                'severity_distribution': 'Dict'
            },
            'gb_reference': 'GB/T 4458.4',
            'description': '校验尺寸标注是否符合国标',
            'priority': 5,
            'dependencies': ['ocr_dim', 'geometry_geo']  # 依赖OCR和几何检测结果
        },
        {
            'detector_name': 'rule_check',
            'error_category': ErrorCategory.TOLERANCE,
            'detection_scope': [
                '配合尺寸公差标注', '形位公差标注', '公差值合理性'
            ],
            'output_schema': {
                'errors': 'List[Dict]', 'total_errors': 'int'
            },
            'gb_reference': 'GB/T 1800.1',
            'description': '校验公差标注',
            'priority': 6,
            'dependencies': ['ocr_dim']
        },
        {
            'detector_name': 'rule_check',
            'error_category': ErrorCategory.SYMBOL,
            'detection_scope': [
                '表面粗糙度符号', '形位公差框格', '基准符号',
                '焊接符号（如适用）'
            ],
            'output_schema': {
                'errors': 'List[Dict]', 'total_errors': 'int'
            },
            'gb_reference': 'GB/T 131',
            'description': '校验技术要求符号标注',
            'priority': 6,
            'dependencies': ['ocr_dim']
        },
        {
            'detector_name': 'llm',
            'error_category': ErrorCategory.GENERAL,
            'detection_scope': [
                '综合错误分析', '苏格拉底式引导生成',
                '学习要点提取', '修正建议生成'
            ],
            'output_schema': {
                'drawing_type': 'str', 'content_summary': 'str',
                'errors': 'List[Dict]', 'overall_score': 'int',
                'summary': 'str', 'learning_points': 'List[str]'
            },
            'gb_reference': '综合',
            'description': 'LLM深度分析与导学',
            'priority': 8,
            'dependencies': ['rule_dim', 'rule_tol', 'rule_sym']
        },
    ]

    # 装配图额外预检
    ASSEMBLY_EXTRA_CONTRACTS = [
        {
            'detector_name': 'ocr',
            'error_category': ErrorCategory.TITLE_BLOCK,
            'detection_scope': ['零件编号识别', '明细栏识别', '装配关系文字'],
            'output_schema': {
                'part_numbers': 'List[Dict]', 'bill_of_materials': 'List[Dict]'
            },
            'gb_reference': 'GB/T 10609.2',
            'description': '识别装配图特有的标题栏和明细栏信息',
            'priority': 1
        },
        {
            'detector_name': 'rule_check',
            'error_category': ErrorCategory.GENERAL,
            'detection_scope': [
                '配合代号标注', '装配尺寸链', '零件编号与明细栏一致性',
                '剖面线方向一致性'
            ],
            'output_schema': {
                'errors': 'List[Dict]', 'total_errors': 'int'
            },
            'gb_reference': '综合',
            'description': '校验装配图特有规范',
            'priority': 7,
            'dependencies': ['ocr_title', 'geometry_geo']
        },
    ]

    @classmethod
    def create_contracts(cls, drawing_type: str = "auto",
                         initial_scan: Optional[Dict] = None) -> ContractRegistry:
        """
        根据图纸类型创建预检集

        Args:
            drawing_type: 图纸类型 ("part"=零件图, "assembly"=装配图, "auto"=自动判断)
            initial_scan: 初始扫描结果（可选，用于动态调整预检）

        Returns:
            ContractRegistry: 已注册所有预检项的注册表
        """
        registry = ContractRegistry()

        # 创建标准零件图预检
        for template in cls.PART_DRAWING_CONTRACTS:
            contract = ErrorContract(**{k: v for k, v in template.items()
                                        if k != 'dependencies'})
            registry.register(contract)

        # 如果是装配图，添加额外预检
        if drawing_type == "assembly":
            for template in cls.ASSEMBLY_EXTRA_CONTRACTS:
                contract = ErrorContract(**{k: v for k, v in template.items()
                                            if k != 'dependencies'})
                registry.register(contract)

        # 根据初始扫描结果动态调整
        if initial_scan:
            cls._adjust_contracts(registry, initial_scan)

        # 验证预检完整性
        issues = registry.validate()
        if issues:
            for issue in issues:
                logger.warning(f"[预检工厂] {issue}")

        return registry

    @classmethod
    def _adjust_contracts(cls, registry: ContractRegistry, scan: Dict):
        """根据初始扫描结果动态调整预检"""
        # 如果检测到大圆但无中心线检测需求，提高线型检测优先级
        if scan.get('large_circles', 0) > 0:
            for c in registry.get_by_category(ErrorCategory.LINE_TYPE):
                if '中心线' not in c.detection_scope:
                    c.detection_scope.append('中心线检测')
                    logger.info(f"[预检调整] {c.contract_id}: 增加'中心线检测'范围")

        # 如果检测到焊接相关文字，添加焊接符号检测预检
        ocr_texts = scan.get('ocr_texts', [])
        if any('焊' in t for t in ocr_texts):
            weld_contract = ErrorContract(
                detector_name='rule_check',
                error_category=ErrorCategory.WELD,
                detection_scope=['焊接符号标注', '焊缝代号', '焊接方法代号'],
                output_schema={'errors': 'List[Dict]', 'total_errors': 'int'},
                gb_reference='GB/T 324',
                description='校验焊接符号标注',
                priority=6
            )
            registry.register(weld_contract)

        # 如果OCR置信度低，添加OCR增强预检
        if scan.get('ocr_confidence', 1.0) < 0.5:
            enhance_contract = ErrorContract(
                detector_name='ocr',
                error_category=ErrorCategory.TITLE_BLOCK,
                detection_scope=['标题栏区域增强OCR', '低置信度区域重识别'],
                output_schema={'enhanced_texts': 'List[Dict]', 'new_count': 'int'},
                gb_reference='',
                description='对低置信度区域进行增强OCR',
                priority=4
            )
            registry.register(enhance_contract)
