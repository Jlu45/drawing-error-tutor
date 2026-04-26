import pytest
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_knowledge_base import DualKnowledgeBase


class TestDualKnowledgeBase:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()
        self.text_dir = os.path.join(self.test_dir, 'knowledge_base')
        self.image_dir = os.path.join(self.test_dir, 'standard_drawings')
        self.gb_dir = os.path.join(self.test_dir, 'gb_standards')
        os.makedirs(self.text_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.gb_dir, exist_ok=True)

    def test_initialization_empty_dirs(self):
        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        assert kb is not None
        assert len(kb.gb_knowledge_items) == 0
        assert len(kb._background_knowledge) == 0

    def test_add_text_knowledge(self):
        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        kb.add_text_knowledge("Test Title", "Test Content", "Test Source")
        assert len(kb.text_knowledge_items) == 1
        assert kb.text_knowledge_items[0]['title'] == "Test Title"

    def test_search_gb_standards_empty(self):
        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        results = kb.search_gb_standards("test")
        assert results == []

    def test_search_gb_standards_with_data(self):
        test_data = [
            {"title": "尺寸注法", "content": "尺寸标注应完整清晰"},
            {"title": "图线", "content": "图线分为粗实线和细实线"},
        ]
        json_path = os.path.join(self.gb_dir, 'gbt14665_extracted.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)

        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        assert len(kb.gb_knowledge_items) == 2
        results = kb.search_gb_standards("尺寸")
        assert len(results) > 0

    def test_get_all_gb_standards(self):
        test_data = [{"title": "Test", "content": "Content"}]
        json_path = os.path.join(self.gb_dir, 'gbt14665_extracted.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False)

        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        all_gb = kb.get_all_gb_standards()
        assert len(all_gb) == 1

    def test_background_knowledge_text(self):
        kb = DualKnowledgeBase(
            text_knowledge_dir=self.text_dir,
            image_knowledge_dir=self.image_dir,
            gb_standards_dir=self.gb_dir
        )
        text = kb.get_background_knowledge_text(1000)
        assert isinstance(text, str)
