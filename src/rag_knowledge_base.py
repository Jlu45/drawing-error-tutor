import os
import json
import cv2
import numpy as np
import shutil
from typing import List, Dict, Tuple
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:
    from config_loader import TEXT_KNOWLEDGE_DIR, IMAGE_KNOWLEDGE_DIR, GB_STANDARDS_DIR
except ImportError:
    TEXT_KNOWLEDGE_DIR = 'data/knowledge_base'
    IMAGE_KNOWLEDGE_DIR = 'data/standard_drawings'
    GB_STANDARDS_DIR = 'data/gb_standards'

class DualKnowledgeBase:
    def __init__(self, text_knowledge_dir=None, image_knowledge_dir=None, gb_standards_dir=None):
        self.text_knowledge_dir = text_knowledge_dir or TEXT_KNOWLEDGE_DIR
        os.makedirs(self.text_knowledge_dir, exist_ok=True)
        self.text_knowledge_items = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        self.text_index = None

        self.image_knowledge_dir = image_knowledge_dir or IMAGE_KNOWLEDGE_DIR
        os.makedirs(self.image_knowledge_dir, exist_ok=True)
        self.image_knowledge_items = []
        self.image_scaler = StandardScaler()
        self.image_index = None

        self.gb_standards_dir = gb_standards_dir or GB_STANDARDS_DIR
        self.gb_knowledge_items = []
        self._background_knowledge = []

        self._load_gb_standards()
        self._load_text_knowledge_as_background()
        self._load_image_knowledge()

        self._build_text_index()
        self._build_image_index()

    def _load_gb_standards(self):
        extracted_path = os.path.join(self.gb_standards_dir, 'gbt14665_extracted.json')
        if os.path.exists(extracted_path):
            try:
                with open(extracted_path, 'r', encoding='utf-8') as f:
                    chapters = json.load(f)
                for ch in chapters:
                    self.gb_knowledge_items.append({
                        'title': ch.get('title', ''),
                        'content': ch.get('content', ''),
                        'source': 'GB/T 14665-2012',
                        'is_gb_standard': True
                    })
                print(f"加载GB标准(已提取): {len(chapters)}个章节")
                return
            except Exception as e:
                print(f"加载已提取GB标准失败: {e}")
        try:
            import pdfplumber
            for filename in os.listdir(self.gb_standards_dir):
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(self.gb_standards_dir, filename)
                    try:
                        full_text = ''
                        with pdfplumber.open(pdf_path) as pdf:
                            for page in pdf.pages:
                                text = page.extract_text()
                                if text:
                                    full_text += text + '\n'
                        if full_text.strip():
                            self.gb_knowledge_items.append({
                                'title': os.path.splitext(filename)[0],
                                'content': full_text.strip(),
                                'source': 'GB/T 14665-2012',
                                'is_gb_standard': True
                            })
                            print(f"加载GB标准(PDF提取): {filename}")
                    except Exception as e:
                        print(f"加载GB标准PDF失败: {filename}, 错误: {e}")
        except ImportError:
            print("pdfplumber未安装，尝试加载预提取文件")
            for filename in os.listdir(self.gb_standards_dir):
                if filename.endswith('.json') and filename != 'gbt14665_extracted.json':
                    try:
                        with open(os.path.join(self.gb_standards_dir, filename), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                self.gb_knowledge_items.append({
                                    'title': item.get('title', ''),
                                    'content': item.get('content', ''),
                                    'source': 'GB/T 14665-2012',
                                    'is_gb_standard': True
                                })
                    except Exception as e:
                        print(f"加载GB标准JSON失败: {filename}, 错误: {e}")

    def _load_text_knowledge_as_background(self):
        for filename in os.listdir(self.text_knowledge_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.text_knowledge_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        self._background_knowledge.append(content)
                except Exception as e:
                    print(f"加载背景知识失败: {filename}, 错误: {e}")
        if self._background_knowledge:
            print(f"加载背景知识(内化用): {len(self._background_knowledge)}条")

    def get_background_knowledge_text(self, max_chars=3000):
        parts = []
        total = 0
        for item in self._background_knowledge:
            if total >= max_chars:
                break
            title = item.get('title', '')
            content = item.get('content', '')
            source = item.get('source', '')
            text = f"【{title}】({source})\n{content}\n"
            if total + len(text) > max_chars:
                text = text[:max_chars - total]
            parts.append(text)
            total += len(text)
        return '\n'.join(parts)

    def _load_image_knowledge(self):
        for filename in os.listdir(self.image_knowledge_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.image_knowledge_dir, filename)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        feature = self._extract_image_feature(image)
                        annotation_file = os.path.splitext(image_path)[0] + '.json'
                        annotation = {}
                        if os.path.exists(annotation_file):
                            with open(annotation_file, 'r', encoding='utf-8') as f:
                                annotation = json.load(f)
                        self.image_knowledge_items.append({
                            'path': image_path,
                            'filename': filename,
                            'feature': feature,
                            'annotation': annotation
                        })
                except Exception as e:
                    print(f"加载图像知识库文件失败: {filename}, 错误: {e}")

    def _extract_image_feature(self, image):
        image = cv2.resize(image, (256, 256))
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hog = cv2.HOGDescriptor()
        hog_feature = hog.compute(image)
        hog_feature = hog_feature.flatten()
        feature = np.concatenate([hist, hog_feature[:512]])
        return feature

    def _build_text_index(self):
        if not self.text_knowledge_items:
            return
        texts = [item['title'] + ' ' + item['content'] for item in self.text_knowledge_items]
        X = self.tfidf_vectorizer.fit_transform(texts)
        X_np = X.toarray().astype(np.float32)
        dimension = X_np.shape[1]
        self.text_index = faiss.IndexFlatL2(dimension)
        self.text_index.add(X_np)

    def _build_image_index(self):
        if not self.image_knowledge_items:
            return
        features = np.array([item['feature'] for item in self.image_knowledge_items]).astype(np.float32)
        features = self.image_scaler.fit_transform(features)
        dimension = features.shape[1]
        self.image_index = faiss.IndexFlatL2(dimension)
        self.image_index.add(features)

    def add_text_knowledge(self, title: str, content: str, source: str):
        knowledge_item = {
            'title': title,
            'content': content,
            'source': source,
            'id': len(self.text_knowledge_items) + 1
        }
        filename = f'knowledge_{knowledge_item["id"]}.json'
        file_path = os.path.join(self.text_knowledge_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_item, f, ensure_ascii=False, indent=2)
        self.text_knowledge_items.append(knowledge_item)
        self._build_text_index()
        print(f"添加文本知识成功: {title}")

    def add_image_knowledge(self, image_path: str, annotation: Dict):
        try:
            image = cv2.imread(image_path)
            if image is not None:
                feature = self._extract_image_feature(image)
                filename = os.path.basename(image_path)
                target_path = os.path.join(self.image_knowledge_dir, filename)
                if os.path.abspath(image_path) != os.path.abspath(target_path):
                    shutil.copy(image_path, target_path)
                else:
                    target_path = image_path
                annotation_file = os.path.splitext(target_path)[0] + '.json'
                with open(annotation_file, 'w', encoding='utf-8') as f:
                    json.dump(annotation, f, ensure_ascii=False, indent=2)
                self.image_knowledge_items.append({
                    'path': target_path,
                    'filename': filename,
                    'feature': feature,
                    'annotation': annotation
                })
                self._build_image_index()
                print(f"添加图像知识成功: {filename}")
        except Exception as e:
            print(f"添加图像知识失败: {e}")

    def search_text_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.text_index:
            results = []
            for item in self.text_knowledge_items:
                score = 0
                if query in item['title']:
                    score += 2
                if query in item['content']:
                    score += 1
                if score > 0:
                    results.append((item, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in results[:top_k]]
        query_vector = self.tfidf_vectorizer.transform([query]).toarray().astype(np.float32)
        distances, indices = self.text_index.search(query_vector, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.text_knowledge_items):
                item = self.text_knowledge_items[idx]
                results.append(item)
        return results

    def search_image_knowledge(self, query_image_path: str, top_k: int = 3) -> List[Dict]:
        try:
            query_image = cv2.imread(query_image_path)
            if query_image is None:
                return []
            query_feature = self._extract_image_feature(query_image)
            query_feature = query_feature.reshape(1, -1).astype(np.float32)
            query_feature = self.image_scaler.transform(query_feature)
            if self.image_index:
                distances, indices = self.image_index.search(query_feature, top_k)
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.image_knowledge_items):
                        item = self.image_knowledge_items[idx]
                        results.append(item)
                return results
            else:
                results = []
                for item in self.image_knowledge_items:
                    similarity = np.dot(query_feature.flatten(), item['feature']) / (
                        np.linalg.norm(query_feature) * np.linalg.norm(item['feature'])
                    )
                    results.append((item, similarity))
                results.sort(key=lambda x: x[1], reverse=True)
                return [item[0] for item in results[:top_k]]
        except Exception as e:
            print(f"搜索图像知识库失败: {e}")
            return []

    def search_gb_standards(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.gb_knowledge_items:
            return []
        results = []
        for item in self.gb_knowledge_items:
            score = 0
            title = item.get('title', '')
            content = item.get('content', '')
            if query in title:
                score += 5
            if query in content:
                count = content.count(query)
                score += min(count, 10)
            if score > 0:
                results.append((item, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in results[:top_k]]

    def get_all_gb_standards(self) -> List[Dict]:
        return self.gb_knowledge_items

    def get_text_knowledge_by_id(self, knowledge_id: int) -> Dict:
        for item in self.text_knowledge_items:
            if item['id'] == knowledge_id:
                return item
        return None

    def get_all_text_knowledge(self) -> List[Dict]:
        return self.text_knowledge_items

    def get_all_image_knowledge(self) -> List[Dict]:
        return self.image_knowledge_items

    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        return self.search_gb_standards(query, top_k)
