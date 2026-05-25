"""
规则校验Agent
=============
基于GB标准的确定性规则校验。
6类规则：尺寸标注、线型、公差、标题栏、符号、几何完整性。
"""

import logging
from typing import Dict, List, Optional, Any

from agents.base import BaseAgent, AgentResult

logger = logging.getLogger("RuleCheckAgent")


class RuleCheckAgent(BaseAgent):
    """规则校验Agent"""

    def __init__(self):
        super().__init__("RuleCheck", max_retries=1)
        self.initialize()

    def _do_initialize(self) -> bool:
        return True

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        """
        执行规则校验

        注意：此Agent不直接读取图像，而是依赖感知Agent的结果。
        这体现了"预检机制"的设计——RuleCheck只关心预检约定的接口数据。
        """
        ocr_result = kwargs.get('ocr_result')
        geometry_result = kwargs.get('geometry_result')
        structure_result = kwargs.get('structure_result')

        # 提取数据（通过预检接口）
        ocr_data = ocr_result.data if ocr_result and ocr_result.success else {}
        geo_data = geometry_result.data if geometry_result and geometry_result.success else {}
        struct_data = structure_result.data if structure_result and structure_result.success else {}

        # 支持通过kwargs调整置信度阈值
        confidence_threshold = kwargs.get('confidence_threshold', 0.3)

        all_errors = []

        # 6类规则校验
        all_errors.extend(self._check_dimension_rules(ocr_data, geo_data))
        all_errors.extend(self._check_tolerance_rules(ocr_data))
        all_errors.extend(self._check_title_block_rules(ocr_data, struct_data))
        all_errors.extend(self._check_symbol_rules(ocr_data))
        all_errors.extend(self._check_line_type_rules(geo_data))
        all_errors.extend(self._check_geometry_completeness(geo_data))
        all_errors.extend(self._check_structure_rules(struct_data))

        # 过滤低置信度结果
        filtered_errors = [e for e in all_errors
                          if self._severity_to_confidence(e.get('severity', '中')) >= confidence_threshold]

        high = sum(1 for e in filtered_errors if e.get('severity') == '高')
        medium = sum(1 for e in filtered_errors if e.get('severity') == '中')
        low = sum(1 for e in filtered_errors if e.get('severity') == '低')

        return AgentResult("RuleCheck", True, {
            'errors': filtered_errors,
            'total_errors': len(filtered_errors),
            'high_severity': high,
            'medium_severity': medium,
            'low_severity': low,
            'severity_distribution': {'高': high, '中': medium, '低': low}
        }, confidence=0.8)

    def _severity_to_confidence(self, severity: str) -> float:
        return {'高': 0.9, '中': 0.6, '低': 0.3}.get(severity, 0.5)

    def _check_dimension_rules(self, ocr_data, geo_data):
        errors = []
        texts = ocr_data.get('texts', [])
        dim_structs = geo_data.get('dimension_structures', [])

        # 检查是否有尺寸数值
        has_dimension = any(self._is_dimension_text(t['text']) for t in texts)
        if not has_dimension and len(texts) > 3:
            errors.append({
                'type': '尺寸标注',
                'description': '未检测到明显的尺寸数值标注',
                'suggestion': '工程图纸应标注关键尺寸（如长度、直径、角度等）',
                'severity': '高',
                'gb_reference': 'GB/T 4458.4'
            })

        # 检查圆形特征是否有Φ符号
        circles = geo_data.get('circles', [])
        large_circles = [c for c in circles if c.get('is_large', False)]
        if large_circles:
            has_phi = any('Φ' in t['text'] or 'φ' in t['text'] or 'phi' in t['text'].lower()
                         for t in texts)
            if not has_phi:
                errors.append({
                    'type': '尺寸标注',
                    'description': f'检测到{len(large_circles)}个大圆但未检测到Φ直径符号',
                    'suggestion': '圆形特征的直径尺寸应使用Φ符号标注',
                    'severity': '中',
                    'gb_reference': 'GB/T 4458.4'
                })

        # 检查尺寸线结构是否充足
        if len(texts) > 5 and len(dim_structs) < 2:
            errors.append({
                'type': '尺寸标注',
                'description': '文字标注较多但尺寸线结构不足',
                'suggestion': '检查尺寸标注是否完整（应有尺寸线、尺寸界线和箭头）',
                'severity': '低',
                'gb_reference': 'GB/T 4458.4'
            })

        return errors

    def _check_tolerance_rules(self, ocr_data):
        errors = []
        texts = ocr_data.get('texts', [])
        has_tolerance = any('±' in t['text'] or '+0' in t['text'] or '-0' in t['text']
                           or 'H7' in t['text'] or 'h6' in t['text']
                           or 'IT' in t['text'] for t in texts)
        if not has_tolerance and len(texts) > 5:
            errors.append({
                'type': '公差',
                'description': '未检测到公差标注',
                'suggestion': '关键配合尺寸应标注尺寸公差或配合代号（如H7/h6）',
                'severity': '中',
                'gb_reference': 'GB/T 1800.1'
            })
        return errors

    def _check_title_block_rules(self, ocr_data, struct_data):
        errors = []
        title_detected = struct_data.get('title_block', {}).get('detected', False)
        if not title_detected:
            errors.append({
                'type': '标题栏',
                'description': '未检测到标准标题栏结构',
                'suggestion': '按GB/T 10609.1标准添加标题栏，包含图名、比例、材料等信息',
                'severity': '中',
                'gb_reference': 'GB/T 10609.1'
            })
        return errors

    def _check_symbol_rules(self, ocr_data):
        errors = []
        texts = ocr_data.get('texts', [])
        if not any('Ra' in t['text'] or '粗糙度' in t['text'] for t in texts):
            errors.append({
                'type': '符号',
                'description': '未检测到表面粗糙度标注',
                'suggestion': '零件图应标注表面粗糙度要求（如Ra3.2）',
                'severity': '中',
                'gb_reference': 'GB/T 131'
            })
        return errors

    def _check_line_type_rules(self, geo_data):
        errors = []
        lt = geo_data.get('line_types', {})
        total = lt.get('total_lines', 0)
        solid = lt.get('solid_count', 0)
        center = lt.get('center_line_count', 0)
        circles = geo_data.get('circles', [])
        large_circles = [c for c in circles if c.get('is_large', False)]
        if large_circles and center == 0:
            errors.append({
                'type': '线型',
                'description': f'检测到{len(large_circles)}个大圆但未检测到中心线(点画线)',
                'suggestion': '圆心位置应使用细点画线绘制中心线',
                'severity': '中',
                'gb_reference': 'GB/T 4457.4'
            })
        if total > 0 and solid < total * 0.2:
            errors.append({
                'type': '线型',
                'description': f'实线比例偏低({solid}/{total})',
                'suggestion': '可见轮廓线应使用粗实线',
                'severity': '低',
                'gb_reference': 'GB/T 4457.4'
            })
        return errors

    def _check_geometry_completeness(self, geo_data):
        errors = []
        lines = geo_data.get('lines', [])
        circles = geo_data.get('circles', [])
        arrows = geo_data.get('arrows', [])
        if len(lines) < 5 and len(circles) < 2:
            errors.append({
                'type': '几何完整性',
                'description': '检测到的几何元素过少，图纸可能不清晰或分辨率不足',
                'suggestion': '建议上传更高分辨率的图纸',
                'severity': '高'
            })
        if len(arrows) < 2:
            errors.append({
                'type': '几何完整性',
                'description': '箭头/尺寸终端检测不足',
                'suggestion': '检查尺寸标注的终端形式是否完整',
                'severity': '低'
            })
        return errors

    def _check_structure_rules(self, struct_data):
        errors = []
        if not struct_data.get('title_block', {}).get('detected', False):
            errors.append({
                'type': '结构',
                'description': '未检测到标准标题栏结构',
                'suggestion': '按GB/T 10609.1标准添加标题栏',
                'severity': '中',
                'gb_reference': 'GB/T 10609.1'
            })
        if not struct_data.get('has_border', True):
            errors.append({
                'type': '结构',
                'description': '未检测到图框线',
                'suggestion': '图纸应包含标准图框',
                'severity': '低',
                'gb_reference': 'GB/T 14665'
            })
        return errors

    def _is_dimension_text(self, text: str) -> bool:
        """判断文本是否为尺寸数值"""
        import re
        return bool(re.search(r'\d+\.?\d*', text)) and len(text) < 15
