"""
Example: RL Feedback Integration

This example demonstrates how to integrate RL feedback
into the analysis workflow for self-evolution.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_agent_system import DrawingOrchestrator
from src.rl_memory_unit import RLMemoryUnit


def rl_feedback_workflow():
    rl = RLMemoryUnit(state_dim=10)

    print("Current RL Statistics:")
    stats = rl.get_stats()
    print(f"  Buffer size: {stats['buffer_size']}")
    print(f"  Training count: {stats['training_count']}")
    print(f"  Epsilon: {stats['epsilon']:.3f}")
    print(f"  Policy version: {stats['params_version']}")
    print()

    params = rl.get_policy_params()
    print("Current Policy Parameters:")
    print(f"  Severity weight (high): {params.severity_weight_high:.2f}")
    print(f"  Severity weight (medium): {params.severity_weight_medium:.2f}")
    print(f"  LLM fusion ratio: {params.llm_score_fusion_ratio:.2f}")
    print(f"  OCR enhance threshold: {params.ocr_enhance_threshold}")
    print()

    # Simulate feedback submission
    # In real usage, session_id comes from the analysis result
    print("Feedback types and their rewards:")
    feedback_types = {
        'confirmed': '+1.0 (Error confirmed by user)',
        'useful_guidance': '+0.5 (Learning guidance was helpful)',
        'partial_confirm': '+0.3 (Partially confirmed)',
        'ignored': '-0.5 (Error was irrelevant)',
        'dismissed_all': '-1.0 (All errors were wrong)',
    }
    for ft, desc in feedback_types.items():
        print(f"  {ft}: {desc}")


if __name__ == '__main__':
    rl_feedback_workflow()
