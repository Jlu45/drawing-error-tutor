import os
import cv2
import numpy as np
from typing import List, Dict, Tuple

class GeometricElementDetector:

    def __init__(self):
        pass

    def detect(self, image_path: str) -> Dict:
        img = cv2.imread(image_path)
        if img is None:
            return self._empty_result()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = {
            'lines': self._detect_lines(gray, img),
            'circles': self._detect_circles(gray),
            'arrows': self._detect_arrows(gray, img),
            'dimension_lines': self._detect_dimension_elements(gray, img),
            'text_regions': self._detect_text_regions(gray),
            'line_types': self._classify_line_types(img, gray)
        }

        return result

    def _empty_result(self):
        return {
            'lines': [], 'circles': [], 'arrows': [],
            'dimension_lines': [], 'text_regions': [], 'line_types': {}
        }

    def _detect_lines(self, gray, img):
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                                  minLineLength=20, maxLineGap=10)

        results = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))

                results.append({
                    'type': 'line',
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'length': round(length, 1),
                    'angle': round(angle, 1),
                    'confidence': min(1.0, length / 100)
                })

        return sorted(results, key=lambda x: x['length'], reverse=True)[:50]

    def _detect_circles(self, gray):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                                    minDist=30, param1=50, param2=30,
                                    minRadius=5, maxRadius=200)

        results = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                results.append({
                    'type': 'circle',
                    'center': [int(x), int(y)],
                    'radius': int(r),
                    'bbox': [int(x-r), int(y-r), int(x+r), int(y+r)],
                    'confidence': min(1.0, r / 50)
                })

        return results

    def _detect_arrows(self, gray, img):
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 500:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

                if len(approx) >= 3 and len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h > 0 else 0

                    if 0.3 < aspect_ratio < 3.0:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            arrows.append({
                                'type': 'arrow',
                                'center': [cx, cy],
                                'bbox': [x, y, x+w, y+h],
                                'confidence': min(1.0, area / 200)
                            })

        return arrows[:20]

    def _detect_dimension_elements(self, gray, img):
        lines = self._detect_lines(gray, img)
        dimension_groups = []

        horizontal_short = [l for l in lines if l['length'] < 40 and abs(l['angle']) < 15]
        vertical_short = [l for l in lines if l['length'] < 40 and (abs(l['angle'] - 90) < 15 or abs(l['angle'] + 90) < 15)]

        long_horizontal = [l for l in lines if l['length'] > 60 and abs(l['angle']) < 10]
        long_vertical = [l for l in lines if l['length'] > 60 and (abs(l['angle'] - 90) < 10 or abs(l['angle'] + 90) < 10)]

        dimensions = []

        for long_line in long_horizontal:
            ly1 = long_line['bbox'][1]
            ly2 = long_line['bbox'][3]
            lx_mid = (long_line['bbox'][0] + long_line['bbox'][2]) / 2

            matching_extensions = [
                e for e in vertical_short
                if abs(e['bbox'][0] - lx_mid) < 25 and
                ((e['bbox'][1] < ly1 - 5 and e['bbox'][1] > ly1 - 30) or
                 (e['bbox'][1] > ly2 + 5 and e['bbox'][1] < ly2 + 30))
            ]

            if len(matching_extensions) >= 1:
                dimensions.append({
                    'type': 'horizontal_dimension',
                    'line': long_line,
                    'extensions': matching_extensions[:2],
                    'bbox': [long_line['bbox'][0], min(long_line['bbox'][1], long_line['bbox'][3]) - 15,
                           long_line['bbox'][2], max(long_line['bbox'][1], long_line['bbox'][3]) + 15],
                    'value_range': abs(long_line['bbox'][2] - long_line['bbox'][0])
                })

        for long_line in long_vertical:
            lx1 = long_line['bbox'][0]
            lx2 = long_line['bbox'][2]
            ly_mid = (long_line['bbox'][1] + long_line['bbox'][3]) / 2

            matching_extensions = [
                e for e in horizontal_short
                if abs(e['bbox'][1] - ly_mid) < 25 and
                ((e['bbox'][0] < lx1 - 5 and e['bbox'][0] > lx1 - 30) or
                 (e['bbox'][0] > lx2 + 5 and e['bbox'][0] < lx2 + 30))
            ]

            if len(matching_extensions) >= 1:
                dimensions.append({
                    'type': 'vertical_dimension',
                    'line': long_line,
                    'extensions': matching_extensions[:2],
                    'bbox': [min(long_line['bbox'][0], long_line['bbox'][2]) - 15, long_line['bbox'][1],
                           max(long_line['bbox'][0], long_line['bbox'][2]) + 15, long_line['bbox'][3]],
                    'value_range': abs(long_line['bbox'][3] - long_line['bbox'][1])
                })

        return dimensions

    def _detect_text_regions(self, gray):
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        text_regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect = float(w) / h if h > 0 else 0

            if 8 < w < 300 and 8 < h < 80 and 0.1 < aspect < 15 and area > 100:
                text_regions.append({
                    'type': 'text_region',
                    'bbox': [x, y, x+w, y+h],
                    'size': (w, h),
                    'area': area
                })

        return text_regions

    def _classify_line_types(self, img, gray):
        lines = self._detect_lines(gray, img)

        solid_lines = []
        dashed_lines = []
        center_lines = []

        for line in lines:
            x1, y1, x2, y2 = line['bbox']

            num_points = max(int(line['length']), 20)
            xs = np.linspace(x1, x2, num_points).astype(int)
            ys = np.linspace(y1, y2, num_points).astype(int)

            pixels = []
            for i in range(len(xs)):
                if 0 <= ys[i] < gray.shape[0] and 0 <= xs[i] < gray.shape[1]:
                    pixels.append(gray[ys[i], xs[i]])

            if len(pixels) < 5:
                continue

            pixels = np.array(pixels)

            transitions = np.sum(np.abs(np.diff(pixels)) > 50)
            transition_ratio = transitions / len(pixels)

            mean_val = np.mean(pixels)

            if transition_ratio < 0.05 and mean_val < 128:
                solid_lines.append(line)
            elif 0.1 < transition_ratio < 0.4 and mean_val < 128:
                dashed_lines.append(line)
            elif 0.05 < transition_ratio < 0.15 and mean_val < 128:
                center_lines.append(line)

        return {
            'solid_count': len(solid_lines),
            'dashed_count': len(dashed_lines),
            'center_line_count': len(center_lines),
            'total_lines': len(lines),
            'solid_lines': solid_lines[:10],
            'dashed_lines': dashed_lines[:10]
        }

    def generate_detection_summary(self, detection_result: Dict) -> str:
        parts = []
        parts.append(f"检测到 {len(detection_result['lines'])} 条直线")
        parts.append(f"检测到 {len(detection_result['circles'])} 个圆形")
        parts.append(f"检测到 {len(detection_result['arrows'])} 个箭头")
        parts.append(f"检测到 {len(detection_result['dimension_lines'])} 组尺寸标注")
        parts.append(f"检测到 {len(detection_result['text_regions'])} 个文字区域")

        lt = detection_result['line_types']
        parts.append(f"线型分布: 实线{lt.get('solid_count',0)}, 虚线{lt.get('dashed_count',0)}, 点画线{lt.get('center_line_count',0)}")

        return "; ".join(parts)

    def convert_to_yolo_format(self, detection_result: Dict) -> List[Dict]:
        results = []

        for circle in detection_result['circles']:
            results.append({
                'class': '圆',
                'bbox': circle['bbox'],
                'confidence': circle['confidence'],
                'source': 'geometric'
            })

        for dim in detection_result['dimension_lines']:
            results.append({
                'class': '尺寸标注',
                'bbox': dim['bbox'],
                'confidence': 0.85,
                'source': 'geometric',
                'detail': f"{dim['type']}, 跨度约{dim['value_range']}px"
            })

        for arrow in detection_result['arrows']:
            results.append({
                'class': '箭头',
                'bbox': arrow['bbox'],
                'confidence': arrow['confidence'],
                'source': 'geometric'
            })

        for line in detection_result['lines'][:10]:
            results.append({
                'class': '直线',
                'bbox': line['bbox'],
                'confidence': line['confidence'],
                'source': 'geometric'
            })

        return results
