"""
Example: Knowledge Base Management

This example demonstrates how to manage the dual knowledge base,
including adding custom GB standards and background knowledge.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_knowledge_base import DualKnowledgeBase


def manage_knowledge_base():
    kb = DualKnowledgeBase()

    # Add custom background knowledge
    kb.add_text_knowledge(
        title="尺寸标注基本原则",
        content="尺寸标注应遵循以下原则：1) 尺寸必须完整，不遗漏也不重复；"
                "2) 尺寸标注应清晰，便于阅读；3) 尺寸应标注在反映形状特征最明显的视图上；"
                "4) 内形尺寸和外形尺寸应分别标注在视图的两侧。",
        source="机械制图教材"
    )

    # Add custom GB standard knowledge
    kb.add_text_knowledge(
        title="表面粗糙度标注方法",
        content="表面粗糙度代号应注在可见轮廓线、尺寸线、尺寸界线或它们的延长线上。"
                "符号的尖端必须从材料外指向表面。",
        source="GB/T 131"
    )

    # Search knowledge
    results = kb.search_gb_standards("尺寸标注")
    print(f"Found {len(results)} results for '尺寸标注':")
    for r in results:
        print(f"  - {r.get('title', 'Untitled')}")

    # Get background knowledge for LLM prompt
    bg_text = kb.get_background_knowledge_text(1000)
    print(f"\nBackground knowledge text ({len(bg_text)} chars):")
    print(bg_text[:200] + "..." if len(bg_text) > 200 else bg_text)


if __name__ == '__main__':
    manage_knowledge_base()
