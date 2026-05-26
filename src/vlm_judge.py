import os
import base64
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("VLMJudge")

try:
    from config_loader import (
        MULTIMODAL_API_URL as _API_URL,
        MULTIMODAL_API_KEY as _API_KEY,
        VLM_JUDGE_ENABLED as _ENABLED,
        VLM_JUDGE_MODEL as _MODEL
    )
    API_URL = _API_URL
    API_KEY = _API_KEY
    JUDGE_ENABLED = _ENABLED
    JUDGE_MODEL = _MODEL
except ImportError:
    API_URL = os.environ.get('MULTIMODAL_API_URL', '')
    API_KEY = os.environ.get('MULTIMODAL_API_KEY', '')
    JUDGE_ENABLED = os.environ.get('VLM_JUDGE_ENABLED', 'true').lower() == 'true'
    JUDGE_MODEL = os.environ.get('VLM_JUDGE_MODEL', 'qwen-vl-plus')


@dataclass
class JudgeVerdict:
    is_consistent: bool
    confidence: float
    verified_errors: List[Dict] = field(default_factory=list)
    missed_errors: List[Dict] = field(default_factory=list)
    false_positives: List[Dict] = field(default_factory=list)
    overall_assessment: str = ""
    details: str = ""


class VLMJudge:
    def __init__(self, api_url: str = "", api_key: str = "", model: str = "",
                 enabled: bool = True):
        self.api_url = api_url or API_URL
        self.api_key = api_key or API_KEY
        self.model = model or JUDGE_MODEL
        self.enabled = enabled and JUDGE_ENABLED
        self._client = None

        if self.enabled and self.api_url and self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            base_url = self.api_url.rstrip('/') + '/v1'
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=60.0
            )
            logger.info(f"[VLMJudge] Initialized with model={self.model}")
        except Exception as e:
            logger.error(f"[VLMJudge] Client init failed: {e}")
            self._client = None
            self.enabled = False

    def judge(self, image_path: str, analysis_result: Dict) -> JudgeVerdict:
        if not self.enabled or self._client is None:
            return self._fallback_judge(analysis_result)

        try:
            image_b64 = self._encode_image(image_path)
            if image_b64 is None:
                return JudgeVerdict(
                    is_consistent=False,
                    confidence=0.0,
                    overall_assessment="Failed to encode image for VLM judge"
                )

            prompt = self._build_judge_prompt(analysis_result)

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )

            result_text = response.choices[0].message.content
            return self._parse_judge_response(result_text, analysis_result)

        except Exception as e:
            logger.error(f"[VLMJudge] Judge failed: {e}")
            return self._fallback_judge(analysis_result)

    def _build_judge_prompt(self, analysis_result: Dict) -> str:
        errors = analysis_result.get('errors', [])
        report = analysis_result.get('report', {})

        error_list = []
        for i, e in enumerate(errors[:15], 1):
            error_list.append(
                f"{i}. [{e.get('severity', '?')}] {e.get('type', '')}: "
                f"{e.get('description', '')} (建议: {e.get('suggestion', '')})"
            )
        error_text = "\n".join(error_list) if error_list else "无检测到的错误"

        return f"""你是一个工程图纸纠错结果验证专家。请审视这张工程图纸，验证以下检测结果是否准确。

检测到的错误（共{report.get('total_errors', 0)}个，评分{report.get('overall_score', 0)}/100）：
{error_text}

请验证：
1. 检测到的错误是否真实存在（是否存在误报）
2. 是否有遗漏的重要错误
3. 错误严重程度评级是否合理

以JSON格式返回：
```json
{{
  "is_consistent": true/false,
  "confidence": 0.0-1.0,
  "verified_errors": [{{"index": 1, "confirmed": true, "reason": "确认原因"}}],
  "missed_errors": [{{"type": "错误类别", "description": "遗漏的错误描述", "severity": "高/中/低"}}],
  "false_positives": [{{"index": 1, "reason": "误报原因"}}],
  "overall_assessment": "总体评价"
}}
```"""

    def _parse_judge_response(self, response_text: str, analysis_result: Dict) -> JudgeVerdict:
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                data = json.loads(response_text[start_idx:end_idx])
                return JudgeVerdict(
                    is_consistent=data.get('is_consistent', True),
                    confidence=float(data.get('confidence', 0.5)),
                    verified_errors=data.get('verified_errors', []),
                    missed_errors=data.get('missed_errors', []),
                    false_positives=data.get('false_positives', []),
                    overall_assessment=data.get('overall_assessment', ''),
                    details=response_text
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[VLMJudge] Failed to parse response: {e}")

        return JudgeVerdict(
            is_consistent=True,
            confidence=0.3,
            overall_assessment="VLM response parsing failed, defaulting to consistent",
            details=response_text
        )

    def _fallback_judge(self, analysis_result: Dict) -> JudgeVerdict:
        errors = analysis_result.get('errors', [])
        high_count = sum(1 for e in errors if e.get('severity') == '高')
        total = len(errors)

        confidence = 0.5
        if total == 0:
            confidence = 0.3
        elif high_count > 3:
            confidence = 0.4

        return JudgeVerdict(
            is_consistent=True,
            confidence=confidence,
            verified_errors=[{"index": i + 1, "confirmed": True, "reason": "自动确认（VLM不可用）"}
                             for i in range(min(total, 15))],
            missed_errors=[],
            false_positives=[],
            overall_assessment=f"本地规则验证（VLM不可用），共{total}个错误，{high_count}个高严重度",
            details="Fallback: VLM judge not available"
        )

    def _encode_image(self, image_path: str) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"[VLMJudge] Image encoding failed: {e}")
            return None

    def is_available(self) -> bool:
        return self.enabled and self._client is not None
