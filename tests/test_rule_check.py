import pytest
import numpy as np
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_agent_system import (
    RuleCheckAgent, AgentResult, DrawingRegion,
    ImageCache, PreprocessPipeline
)


class TestRuleCheckAgent:
    def setup_method(self):
        self.agent = RuleCheckAgent()

    def test_agent_initialization(self):
        assert self.agent.name == "RuleCheck"
        assert self.agent._initialized is True

    def test_check_dimension_rules_no_numbers(self):
        ocr_data = {'texts': [{'text': 'hello', 'confidence': 0.9}]}
        geo_data = {}
        errors = self.agent._check_dimension_rules(ocr_data, geo_data)
        assert len(errors) > 0
        assert errors[0]['type'] == '尺寸标注'
        assert errors[0]['severity'] == '高'

    def test_check_dimension_rules_circles_no_phi(self):
        ocr_data = {'texts': [{'text': '50', 'confidence': 0.9}]}
        geo_data = {'circles': [{'center': [100, 100], 'radius': 50, 'is_large': True}]}
        errors = self.agent._check_dimension_rules(ocr_data, geo_data)
        phi_errors = [e for e in errors if 'Φ' in e['description'] or '直径' in e['suggestion']]
        assert len(phi_errors) > 0

    def test_check_tolerance_rules_missing(self):
        ocr_data = {'texts': [{'text': '50', 'confidence': 0.9}]}
        errors = self.agent._check_tolerance_rules(ocr_data)
        assert len(errors) > 0
        assert errors[0]['type'] == '公差'

    def test_check_tolerance_rules_present(self):
        ocr_data = {'texts': [{'text': 'Φ50±0.02', 'confidence': 0.9}]}
        errors = self.agent._check_tolerance_rules(ocr_data)
        assert len(errors) == 0

    def test_check_title_rules_incomplete(self):
        ocr_data = {'texts': [{'text': '名称', 'confidence': 0.9}]}
        errors = self.agent._check_title_rules(ocr_data)
        assert len(errors) > 0
        assert errors[0]['type'] == '标题栏'

    def test_check_title_rules_complete(self):
        ocr_data = {'texts': [
            {'text': '名称', 'confidence': 0.9},
            {'text': '比例', 'confidence': 0.9},
            {'text': '材料', 'confidence': 0.9},
        ]}
        errors = self.agent._check_title_rules(ocr_data)
        assert len(errors) == 0

    def test_check_symbol_rules_missing(self):
        ocr_data = {'texts': [{'text': '50', 'confidence': 0.9}]}
        errors = self.agent._check_symbol_rules(ocr_data)
        assert len(errors) > 0
        assert errors[0]['type'] == '符号'

    def test_check_symbol_rules_present(self):
        ocr_data = {'texts': [{'text': 'Ra3.2', 'confidence': 0.9}]}
        errors = self.agent._check_symbol_rules(ocr_data)
        assert len(errors) == 0

    def test_check_line_type_rules_missing_center_line(self):
        geo_data = {
            'line_types': {'total_lines': 20, 'solid_count': 10, 'center_line_count': 0},
            'circles': [{'center': [100, 100], 'radius': 80, 'is_large': True}]
        }
        errors = self.agent._check_line_type_rules(geo_data)
        center_errors = [e for e in errors if '中心线' in e['description']]
        assert len(center_errors) > 0

    def test_check_geometry_completeness_insufficient(self):
        geo_data = {
            'lines': [{'start': (0, 0), 'end': (10, 10)}],
            'circles': [],
            'arrows': []
        }
        errors = self.agent._check_geometry_completeness(geo_data)
        assert len(errors) > 0

    def test_full_analysis(self):
        ocr_result = AgentResult("OCR", True, {
            'texts': [{'text': '50', 'confidence': 0.9}],
            'total_count': 1,
            'high_confidence_count': 1
        })
        geometry_result = AgentResult("Geometry", True, {
            'lines': [],
            'circles': [{'center': [100, 100], 'radius': 50, 'is_large': True}],
            'arrows': [],
            'contours': [],
            'line_types': {'total_lines': 0, 'solid_count': 0, 'center_line_count': 0},
            'dimension_structures': []
        })
        structure_result = AgentResult("Structure", True, {
            'title_block': {'detected': False},
            'has_border': True
        })

        result = self.agent.analyze("", ocr_result=ocr_result,
                                     geometry_result=geometry_result,
                                     structure_result=structure_result)
        assert result.success is True
        assert result.data['total_errors'] > 0


class TestAgentResult:
    def test_creation(self):
        result = AgentResult("Test", True, {"key": "value"})
        assert result.agent_name == "Test"
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.errors == []
        assert result.confidence == 0.0

    def test_with_errors(self):
        result = AgentResult("Test", False, {}, ["error1", "error2"])
        assert len(result.errors) == 2


class TestDrawingRegion:
    def test_creation(self):
        region = DrawingRegion("标题栏区域", 520, 510, 280, 90)
        assert region.name == "标题栏区域"
        assert region.x == 520
        assert region.w == 280


class TestPreprocessPipeline:
    def test_ocr_mode(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = PreprocessPipeline.run(img, "ocr")
        assert result is not None
        assert result.dtype == np.uint8

    def test_geometry_mode(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = PreprocessPipeline.run(img, "geometry")
        assert result is not None

    def test_none_input(self):
        result = PreprocessPipeline.run(None, "ocr")
        assert result is None
