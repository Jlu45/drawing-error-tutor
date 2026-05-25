"""
Agent基类桥接模块
================
从 __init__.py 重新导出 BaseAgent, AgentResult, DrawingRegion，
使 `from agents.base import ...` 导入方式正常工作。
"""

from agents import BaseAgent, AgentResult, DrawingRegion

__all__ = ['BaseAgent', 'AgentResult', 'DrawingRegion']
