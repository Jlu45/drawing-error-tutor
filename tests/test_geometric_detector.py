import pytest
import numpy as np
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometric_detector import GeometricElementDetector


class TestGeometricElementDetector:
    def setup_method(self):
        self.detector = GeometricElementDetector()
        self.test_dir = tempfile.mkdtemp()

    def _create_test_image(self, width=800, height=600, color=(255, 255, 255)):
        import cv2
        img = np.ones((height, width, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
        path = os.path.join(self.test_dir, 'test_drawing.png')
        cv2.imwrite(path, img)
        return path

    def _create_line_image(self):
        import cv2
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.line(img, (100, 300), (700, 300), (0, 0, 0), 2)
        cv2.line(img, (400, 50), (400, 550), (0, 0, 0), 2)
        path = os.path.join(self.test_dir, 'test_lines.png')
        cv2.imwrite(path, img)
        return path

    def _create_circle_image(self):
        import cv2
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.circle(img, (400, 300), 100, (0, 0, 0), 2)
        path = os.path.join(self.test_dir, 'test_circles.png')
        cv2.imwrite(path, img)
        return path

    def test_detector_initialization(self):
        assert self.detector is not None

    def test_detect_empty_result_for_missing_file(self):
        result = self.detector.detect('nonexistent_file.png')
        assert result is not None
        assert 'lines' in result
        assert 'circles' in result
        assert 'arrows' in result
        assert len(result['lines']) == 0

    def test_detect_blank_image(self):
        path = self._create_test_image()
        result = self.detector.detect(path)
        assert result is not None
        assert isinstance(result['lines'], list)
        assert isinstance(result['circles'], list)

    def test_detect_lines(self):
        path = self._create_line_image()
        result = self.detector.detect(path)
        assert len(result['lines']) > 0

    def test_detect_circles(self):
        path = self._create_circle_image()
        result = self.detector.detect(path)
        assert len(result['circles']) > 0

    def test_line_types_structure(self):
        path = self._create_test_image()
        result = self.detector.detect(path)
        lt = result['line_types']
        assert 'solid_count' in lt
        assert 'dashed_count' in lt
        assert 'center_line_count' in lt
        assert 'total_lines' in lt

    def test_convert_to_yolo_format(self):
        path = self._create_line_image()
        result = self.detector.detect(path)
        yolo_results = self.detector.convert_to_yolo_format(result)
        assert isinstance(yolo_results, list)
        for item in yolo_results:
            assert 'class' in item
            assert 'bbox' in item
            assert 'confidence' in item
            assert 'source' in item
            assert item['source'] == 'geometric'

    def test_generate_detection_summary(self):
        path = self._create_test_image()
        result = self.detector.detect(path)
        summary = self.detector.generate_detection_summary(result)
        assert isinstance(summary, str)
        assert len(summary) > 0
