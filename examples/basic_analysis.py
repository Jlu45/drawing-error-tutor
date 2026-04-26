"""
Example: Basic Drawing Analysis

This example demonstrates how to use the DrawingOrchestrator
to analyze an engineering drawing programmatically.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_agent_system import DrawingOrchestrator
from src.rag_knowledge_base import DualKnowledgeBase

API_URL = os.environ.get('MULTIMODAL_API_URL', '')
API_KEY = os.environ.get('MULTIMODAL_API_KEY', '')
LLM_MODEL = os.environ.get('LLM_MODEL', 'Qwen2.5-72B-Instruct')


def analyze_drawing(image_path: str):
    orchestrator = DrawingOrchestrator(
        api_url=API_URL,
        api_key=API_KEY,
        llm_model=LLM_MODEL
    )

    kb = DualKnowledgeBase()
    background_knowledge = kb.get_background_knowledge_text(2000)

    result = orchestrator.analyze(image_path, background_knowledge)

    print(f"Analysis complete for: {image_path}")
    print(f"Total errors found: {result['report']['total_errors']}")
    print(f"Overall score: {result['report']['overall_score']}/100")
    print(f"Summary: {result['report']['summary']}")
    print()

    for i, error in enumerate(result['errors'], 1):
        print(f"  Error {i}: [{error.get('severity', '?')}] {error.get('type', '')}")
        print(f"    Description: {error.get('description', '')}")
        print(f"    Suggestion: {error.get('suggestion', '')}")
        if error.get('gb_reference'):
            print(f"    GB Reference: {error['gb_reference']}")
        print()

    print("Socratic Guidance:")
    for fb in result['feedback'][:3]:
        print(f"  → {fb}")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python basic_analysis.py <image_path>")
        print("Example: python basic_analysis.py data/drawings/reducer.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    analyze_drawing(image_path)
