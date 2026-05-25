"""
LLM分析Agent
============
基于大语言模型的深度分析与苏格拉底式导学。
支持经验上下文注入。
"""

import json
import logging
from typing import Dict, List, Optional, Any

from agents.base import BaseAgent, AgentResult

logger = logging.getLogger("LLMAgent")


class LLMAgent(BaseAgent):
    """LLM深度分析Agent"""

    def __init__(self, api_url: str, api_key: str, model: str = "Qwen2.5-72B-Instruct"):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model
        self.client = None
        super().__init__("LLM", max_retries=2, timeout=300.0)
        self.initialize()

    def _do_initialize(self) -> bool:
        try:
            from openai import OpenAI
            base_url = self.api_url.rstrip('/')
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=self.timeout)
            return True
        except Exception as e:
            logger.error(f"[LLM] Init failed: {e}")
            return False

    def _do_analyze(self, image_path: str, **kwargs) -> AgentResult:
        if self.client is None:
            return AgentResult("LLM", False, {}, ["LLM client not initialized"], confidence=0.0)

        ocr_result = kwargs.get('ocr_result')
        geometry_result = kwargs.get('geometry_result')
        structure_result = kwargs.get('structure_result')
        rule_result = kwargs.get('rule_result')
        background_knowledge = kwargs.get('background_knowledge', '')
        experience_context = kwargs.get('experience_context', '')

        context = self._build_context(ocr_result, geometry_result, structure_result, rule_result)

        system_prompt = """你是工程图纸智能纠错专家，精通GB/T机械制图标准。基于检测结果进行深度纠错分析，采用苏格拉底式引导。"""

        if background_knowledge:
            system_prompt += f"\n\n【内化知识】\n{background_knowledge}\n基于以上知识分析，不要直接引用。"

        if experience_context:
            system_prompt += f"\n\n{experience_context}"

        user_prompt = f"""基于检测结果分析工程图纸：

{context}

任务：1.综合分析图纸类型和内容 2.逐项检查尺寸标注/线型/公差/标题栏/符号 3.深度诊断规则检查问题 4.苏格拉底式提问引导

JSON格式返回：
```json
{{
  "drawing_type": "图纸类型",
  "content_summary": "内容概述",
  "errors": [
    {{"type": "类别", "description": "问题", "suggestion": "建议", "severity": "高/中/低", "gb_reference": "国标"}}
  ],
  "overall_score": 0-100,
  "summary": "总体评价",
  "learning_points": ["要点1", "要点2"]
}}
```"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )

            result_text = response.choices[0].message.content or ''
            if not result_text:
                logger.warning("[LLM] API返回空内容，尝试从其他字段提取")
                for choice in response.choices:
                    if choice.message.content:
                        result_text = choice.message.content
                        break
            if not result_text:
                logger.error("[LLM] API返回内容为空")
                return AgentResult("LLM", False, {}, ["LLM返回空内容"], confidence=0.0)
            
            usage_info = None
            if response.usage:
                try:
                    usage_info = response.usage.model_dump() if hasattr(response.usage, 'model_dump') else (response.usage.dict() if hasattr(response.usage, 'dict') else str(response.usage))
                except Exception:
                    usage_info = None

            logger.info(f"[LLM] API返回成功，内容长度: {len(result_text)} 字符")
            logger.debug(f"[LLM] 响应前200字符: {result_text[:200]}")
            return AgentResult("LLM", True, {
                'raw_response': result_text,
                'model': response.model,
                'usage': usage_info
            }, confidence=0.8)
        except Exception as e:
            logger.error(f"[LLM] API call failed: {e}")
            return AgentResult("LLM", False, {}, [str(e)], confidence=0.0)

    def _build_context(self, ocr_result, geometry_result, structure_result, rule_result):
        parts = []
        if ocr_result and ocr_result.success:
            texts = ocr_result.data.get('texts', [])
            if texts:
                text_summary = "\n".join([f"  - \"{t['text']}\" ({t['confidence']:.2f})"
                                          for t in texts[:20]])
                parts.append(f"【OCR】{len(texts)}个文字：\n{text_summary}")
            else:
                parts.append("【OCR】未识别到文字")

        if geometry_result and geometry_result.success:
            geo = geometry_result.data
            lines = geo.get('lines', [])
            circles = geo.get('circles', [])
            arrows = geo.get('arrows', [])
            lt = geo.get('line_types', {})
            dim_structs = geo.get('dimension_structures', [])
            parts.append(f"【几何】直线{len(lines)}条(水平{sum(1 for l in lines if l.get('is_horizontal'))},垂直{sum(1 for l in lines if l.get('is_vertical'))}) 圆{len(circles)}个 箭头{len(arrows)}个 尺寸线{len(dim_structs)}对 线型:实线{lt.get('solid_count',0)}/虚线{lt.get('dashed_count',0)}/点画线{lt.get('center_line_count',0)}")

        if structure_result and structure_result.success:
            s = structure_result.data
            parts.append(f"【结构】{s.get('image_size',{}).get('width',0)}x{s.get('image_size',{}).get('height',0)}px 标题栏:{'有' if s.get('title_block',{}).get('detected') else '无'} 图框:{'有' if s.get('has_border') else '无'} 视图{len(s.get('view_areas',[]))}个")

        if rule_result and rule_result.success:
            r = rule_result.data
            parts.append(f"【规则】错误{r.get('total_errors',0)}个(高{r.get('high_severity',0)}/中{r.get('medium_severity',0)}/低{r.get('low_severity',0)})")
            for err in r.get('errors', []):
                parts.append(f"  ⚠[{err.get('severity','?')}] {err.get('type','')}: {err.get('description','')}")

        return "\n".join(parts)
