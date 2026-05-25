"""
Agent基类模块
=============
提供统一的Agent接口，复用原版BaseAgent的核心设计。
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger("AgentBase")


@dataclass
class AgentResult:
    """统一的Agent执行结果"""
    agent_name: str
    success: bool
    data: Dict
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class DrawingRegion:
    """图纸区域定义"""
    name: str
    x: int
    y: int
    w: int
    h: int
    ocr_text: List[Dict] = field(default_factory=list)
    geometry: Dict = field(default_factory=dict)
    description: str = ""


class BaseAgent(ABC):
    """
    Agent抽象基类

    提供统一的生命周期管理和容错机制：
    - 初始化（initialize）
    - 执行（analyze）+ 指数退避重试
    - 输入验证（validate_input）
    """

    def __init__(self, name: str, max_retries: int = 2, timeout: float = 60.0):
        self.name = name
        self.max_retries = max_retries
        self.timeout = timeout
        self._initialized = False

    @abstractmethod
    def _do_initialize(self) -> bool:
        """子类实现具体的初始化逻辑"""
        pass

    @abstractmethod
    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        """子类实现具体的分析逻辑"""
        pass

    def initialize(self) -> bool:
        try:
            self._initialized = self._do_initialize()
            status = "OK" if self._initialized else "FAIL"
            logger.info(f"[{self.name}] Initialize: {status}")
            return self._initialized
        except Exception as e:
            logger.error(f"[{self.name}] Initialize error: {e}")
            self._initialized = False
            return False

    def analyze(self, image_path: str, **kwargs) -> AgentResult:
        """执行分析，带重试机制"""
        if not self._initialized:
            return AgentResult(self.name, False, {}, ["Agent not initialized"], confidence=0.0)

        for attempt in range(self.max_retries + 1):
            start = time.time()
            try:
                result = self._do_analyze(image_path, **kwargs)
                result.execution_time_ms = (time.time() - start) * 1000
                if result.success:
                    return result
                if attempt < self.max_retries:
                    logger.warning(f"[{self.name}] Attempt {attempt+1} failed, retrying...")
                    time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                if attempt < self.max_retries:
                    logger.warning(f"[{self.name}] Exception on attempt {attempt+1}: {e}")
                    time.sleep(0.5 * (attempt + 1))
                else:
                    logger.error(f"[{self.name}] All attempts exhausted: {e}")
                    return AgentResult(self.name, False, {}, [str(e)],
                                       execution_time_ms=elapsed, confidence=0.0)

        return AgentResult(self.name, False, {}, ["All retries exhausted"], confidence=0.0)

    def validate_input(self, image_path: str) -> Optional[str]:
        """验证输入图像路径"""
        if not image_path:
            return "Image path is empty"
        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'):
            return f"Unsupported format: {ext}"
        return None
