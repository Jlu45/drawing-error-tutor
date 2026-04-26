import os
import cv2
import numpy as np
import base64
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from geometric_detector import GeometricElementDetector

try:
    from config_loader import MULTIMODAL_VISION_MODEL
except ImportError:
    MULTIMODAL_VISION_MODEL = os.environ.get('MULTIMODAL_VISION_MODEL', 'kimi')

ocr = None
yolo_model = None

try:
    from ultralytics import YOLO
    try:
        yolo_model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"YOLOv8初始化失败: {e}")
        print("将使用模拟数据")
        yolo_model = None
except Exception as e:
    print(f"导入YOLOv8失败: {e}")
    yolo_model = None

paddleocr_available = False
try:
    from paddleocr import PaddleOCR
    paddleocr_available = True
    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    except Exception as e:
        print(f"PaddleOCR初始化失败: {e}")
        print("将使用模拟数据")
        ocr = None
except Exception as e:
    print(f"导入PaddleOCR失败: {e}")
    ocr = None

DRAWING_CLASSES = {
    0: '尺寸标注', 1: '尺寸界线', 2: '箭头', 3: '公差框',
    4: '直线', 5: '圆', 6: '弧', 7: '剖面线',
    8: '基准符号', 9: '焊接符号', 10: '形位公差符号',
    11: '标题栏', 12: '明细表', 13: '技术要求', 14: '其他符号'
}

class MultimodalAgent:
    def __init__(self, multimodal_api_url=None, api_key=None):
        self.ocr = ocr
        self.yolo = yolo_model
        self.multimodal_api_url = multimodal_api_url
        self.api_key = api_key
        try:
            self.geo_detector = GeometricElementDetector()
            print("几何元素检测器初始化成功")
        except Exception as e:
            print(f"几何元素检测器初始化失败: {e}")
            self.geo_detector = None
        if multimodal_api_url and api_key:
            try:
                base_url = multimodal_api_url.rstrip('/') + '/v1'
                self.client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=60
                )
                print(f"已连接多模态大模型API: {base_url}")
            except Exception as e:
                print(f"初始化OpenAI客户端失败: {e}")
                self.client = None
        else:
            self.client = None
        self.scaler = StandardScaler()

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val > 180:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
        else:
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, 15, 10)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed_path = image_path.replace('.png', '_processed.png')
        cv2.imwrite(processed_path, processed)
        return processed_path

    def ocr_detection(self, image_path):
        if self.ocr is None:
            print("PaddleOCR未初始化，OCR识别不可用")
            return []
        processed_path = self.preprocess_image(image_path)
        result = self.ocr.ocr(processed_path, cls=True)
        ocr_results = []
        if result and result[0]:
            for line in result[0]:
                if line:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    text = self.normalize_special_chars(text)
                    ocr_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
        if os.path.exists(processed_path):
            os.remove(processed_path)
        return ocr_results

    def normalize_special_chars(self, text):
        char_map = {'φ': 'Φ', '±': '±', '°': '°', '∅': 'Φ', '％': '%'}
        for old_char, new_char in char_map.items():
            text = text.replace(old_char, new_char)
        return text

    def object_detection(self, image_path):
        all_results = []
        if self.yolo is not None:
            processed_path = self.preprocess_image(image_path)
            results = self.yolo(processed_path, imgsz=1280)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    bbox = box.xyxy[0].tolist()
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = DRAWING_CLASSES.get(class_id, self.yolo.names[class_id])
                    all_results.append({
                        'class': class_name,
                        'bbox': bbox,
                        'confidence': confidence,
                        'source': 'yolo'
                    })
            if os.path.exists(processed_path):
                os.remove(processed_path)
        if self.geo_detector is not None:
            try:
                geo_result = self.geo_detector.detect(image_path)
                geo_items = self.geo_detector.convert_to_yolo_format(geo_result)
                existing_bboxes = set()
                for item in all_results:
                    key = (round(item['bbox'][0]/10), round(item['bbox'][1]/10))
                    existing_bboxes.add(key)
                for geo_item in geo_items:
                    key = (round(geo_item['bbox'][0]/10), round(geo_item['bbox'][1]/10))
                    if key not in existing_bboxes:
                        all_results.append(geo_item)
                        existing_bboxes.add(key)
                self._last_geo_result = geo_result
            except Exception as e:
                print(f"几何检测失败: {e}")
                self._last_geo_result = None
        else:
            self._last_geo_result = None
        if not all_results:
            return []
        return all_results

    def extract_visual_features(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        hog = cv2.HOGDescriptor()
        features = hog.compute(image)
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        visual_features = np.concatenate([features.flatten(), hist])
        return visual_features

    def extract_text_features(self, ocr_results):
        text_lengths = [len(item['text']) for item in ocr_results]
        text_count = len(ocr_results)
        if text_count > 0:
            avg_text_length = sum(text_lengths) / text_count
        else:
            avg_text_length = 0
        special_chars = ['Φ', '±', '°', '%', '//', '×', '÷']
        special_char_count = sum([sum(1 for c in item['text'] if c in special_chars) for item in ocr_results])
        text_features = np.array([text_count, avg_text_length, special_char_count])
        return text_features

    def extract_structural_features(self, detection_results):
        class_counts = {}
        for item in detection_results:
            class_name = item['class']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        total_objects = len(detection_results)
        has_dimension = 1 if '尺寸标注' in class_counts else 0
        has_title = 1 if '标题栏' in class_counts else 0
        has_tolerance = 1 if '公差框' in class_counts else 0
        structural_features = np.array([total_objects, has_dimension, has_title, has_tolerance])
        return structural_features

    def fuse_features(self, visual_features, text_features, structural_features):
        visual_features = self.scaler.fit_transform(visual_features.reshape(1, -1)).flatten()
        text_features = self.scaler.fit_transform(text_features.reshape(1, -1)).flatten()
        structural_features = self.scaler.fit_transform(structural_features.reshape(1, -1)).flatten()
        fused_features = np.concatenate([visual_features, text_features, structural_features])
        return fused_features

    def extract_multimodal_features(self, image_path, ocr_results=None, detection_results=None):
        if ocr_results is None:
            ocr_results = self.ocr_detection(image_path)
        if detection_results is None:
            detection_results = self.object_detection(image_path)
        visual_features = self.extract_visual_features(image_path)
        text_features = self.extract_text_features(ocr_results)
        structural_features = self.extract_structural_features(detection_results)
        fused_features = self.fuse_features(visual_features, text_features, structural_features)
        return fused_features

    def analyze_drawing(self, image_path):
        ocr_results = self.ocr_detection(image_path)
        detection_results = self.object_detection(image_path)
        multimodal_features = self.extract_multimodal_features(image_path, ocr_results, detection_results)
        analysis = {
            'ocr_results': ocr_results,
            'detection_results': detection_results,
            'multimodal_features': multimodal_features.tolist(),
            'summary': f'识别到 {len(ocr_results)} 个文字区域，{len(detection_results)} 个目标'
        }
        return analysis

    def detect_errors(self, image_path):
        analysis = self.analyze_drawing(image_path)
        errors = []
        has_dimension = any('Φ' in item['text'] or '±' in item['text'] for item in analysis['ocr_results'])
        if not has_dimension:
            errors.append('可能缺少尺寸标注')
        has_center_line = any('中心' in item['text'] or '对称' in item['text'] for item in analysis['ocr_results'])
        if not has_center_line:
            errors.append('可能缺少中心线标注')
        has_title = any('标题栏' in item['class'] for item in analysis['detection_results'])
        if not has_title:
            errors.append('可能缺少标题栏')
        has_tolerance = any('公差' in item['class'] or '±' in item['text'] for item in analysis['ocr_results'])
        if not has_tolerance:
            errors.append('可能缺少公差标注')
        return {'analysis': analysis, 'errors': errors}

    def generate_feedback(self, image_path):
        result = self.detect_errors(image_path)
        errors = result['errors']
        feedback = []
        for error in errors:
            if '尺寸标注' in error:
                feedback.append('你注意到图纸中的尺寸标注了吗？请思考一下，哪些重要尺寸需要标注？')
                feedback.append('根据机械制图标准，尺寸标注应该遵循什么原则？')
            elif '中心线' in error:
                feedback.append('请观察图纸中的对称结构，思考一下是否需要添加中心线？')
                feedback.append('中心线在机械制图中有什么作用？它应该使用什么线型绘制？')
            elif '标题栏' in error:
                feedback.append('图纸缺少标题栏，你知道标题栏应该包含哪些内容吗？')
                feedback.append('标题栏的位置和格式有什么要求？')
            elif '公差' in error:
                feedback.append('你考虑过图纸中的公差标注吗？哪些尺寸需要标注公差？')
                feedback.append('公差标注的格式和方法是什么？')
        if not feedback:
            feedback.append('图纸整体看起来不错，你能再检查一下细节部分，确保所有元素都符合标准吗？')
            feedback.append('你认为还有哪些方面可以进一步完善？')
        return {'errors': errors, 'feedback': feedback}

    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def call_multimodal_api(self, image_path, context=None):
        if not self.client:
            print("未配置多模态模型API")
            return None
        try:
            image_base64 = self.encode_image(image_path)
            ocr_text = ""
            detection_text = ""
            geo_summary = ""
            if context:
                if 'ocr_results' in context and context['ocr_results']:
                    ocr_items = [f"{r['text']}(置信度:{r['confidence']:.2f})" for r in context['ocr_results']]
                    ocr_text = "【OCR识别结果】\n" + "\n".join(ocr_items)
                if 'detection_results' in context and context['detection_results']:
                    det_items = [f"- {d.get('class','未知')}(置信度:{d.get('confidence',0):.2f})"
                                for d in context['detection_results'][:20]]
                    detection_text = f"【目标检测/几何元素】共{len(context['detection_results'])}个元素:\n" + "\n".join(det_items)
                if hasattr(self, '_last_geo_result') and self._last_geo_result:
                    detector = GeometricElementDetector()
                    geo_summary = "【几何结构分析】\n" + detector.generate_detection_summary(self._last_geo_result)

            background_section = ""
            if context and context.get('background_knowledge'):
                background_section = f"\n\n【你已内化的专业背景知识】\n{context['background_knowledge']}\n请基于以上内化知识进行分析，但不要在回复中直接引用或提及这些背景知识。"

            system_prompt = f"""你是一个专业的工程图纸智能纠错专家（机械制图领域），具有以下专业能力：
1. 熟悉GB/T 4457-4460等机械制图国家标准
2. 能识别减速器装配图、零件图等各类工程图纸
3. 能精准定位尺寸标注、线型使用、公差标注、标题栏等方面的错误
4. 采用苏格拉底式引导方式帮助用户理解错误原因

你的分析必须基于图纸实际内容给出具体、针对性的反馈，避免泛泛而谈。{background_section}"""

            user_prompt = f"""请仔细分析这张减速器/机械零件图，完成以下任务：

## 第一步：基础识别
请列出图中所有能识别到的文字和图形元素。

## 第二步：错误检测
逐项检查以下方面，指出具体问题位置和错误类型：

### A. 尺寸标注检查
- 尺寸数字是否完整？是否有遗漏的关键尺寸？
- 尺寸标注位置是否合理（不交叉、不遮挡）？
- 是否有重复或矛盾的尺寸？
- 直径符号Φ是否正确使用？

### B. 线型检查
- 实线（轮廓线）、虚线（隐藏轮廓）、点画线（中心线）使用是否正确？
- 是否缺少必要的中心线？

### C. 公差与表面粗糙度
- 关键尺寸是否有公差标注？
- 表面粗糙度要求是否合理？

### D. 标题栏和技术要求
- 标题栏信息是否完整（图名、比例、材料等）？

## 第三步：生成纠错报告
以JSON格式返回（注意必须是有效JSON）：
```json
{{
  "errors": [
    {{"type": "错误类别", "description": "具体问题描述（含位置）", "suggestion": "修正建议", "severity": "高/中/低"}}
  ],
  "overall_score": 整体评分(0-100),
  "summary": "总体评价（针对此图的个性化评价）",
  "learning_points": ["学习要点1", "学习要点2"]
}}
```

{geo_summary}
{ocr_text}
{detection_text}

请确保返回的是有效的JSON格式。"""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]

            model_name = MULTIMODAL_VISION_MODEL

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=3000,
                temperature=0.3
            )

            result_text = response.choices[0].message.content

            return {
                'raw_response': result_text,
                'model': response.model,
                'usage': response.usage.dict() if response.usage else None
            }

        except Exception as e:
            print(f"调用多模态API时出错: {e}")
            return None

    def analyze_with_api(self, image_path, ocr_results=None, detection_results=None, background_knowledge=None):
        if ocr_results is None:
            ocr_results = self.ocr_detection(image_path)
        if detection_results is None:
            detection_results = self.object_detection(image_path)
        multimodal_features = self.extract_multimodal_features(image_path, ocr_results, detection_results)
        context = {
            "ocr_results": ocr_results,
            "detection_results": detection_results,
            "multimodal_features": multimodal_features.tolist(),
            "background_knowledge": background_knowledge
        }
        api_result = self.call_multimodal_api(image_path, context)
        return api_result
