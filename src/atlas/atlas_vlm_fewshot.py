import os
import base64
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .atlas_case_schema import AtlasCase

logger = logging.getLogger("AtlasVLMFewShot")

try:
    from config_loader import (
        MULTIMODAL_API_URL as _API_URL,
        MULTIMODAL_API_KEY as _API_KEY,
        ATLAS_ENABLE_VLM_FEWSHOT as _ENABLED,
        VLM_JUDGE_MODEL as _MODEL
    )
    API_URL = _API_URL
    API_KEY = _API_KEY
    VLM_FEWSHOT_ENABLED = _ENABLED
    VLM_MODEL = _MODEL
except ImportError:
    API_URL = os.environ.get('MULTIMODAL_API_URL', '')
    API_KEY = os.environ.get('MULTIMODAL_API_KEY', '')
    VLM_FEWSHOT_ENABLED = True
    VLM_MODEL = os.environ.get('VLM_JUDGE_MODEL', 'qwen-vl-plus')


@dataclass
class FewShotExample:
    image_path: str = ""
    image_b64: str = ""
    errors: List[Dict] = field(default_factory=list)
    corrections: List[Dict] = field(default_factory=list)
    category: str = ""
    description: str = ""


class AtlasVLMFewShot:
    def __init__(self, api_url: str = "", api_key: str = "", model: str = "",
                 enabled: bool = True):
        self.api_url = api_url or API_URL
        self.api_key = api_key or API_KEY
        self.model = model or VLM_MODEL
        self.enabled = enabled and VLM_FEWSHOT_ENABLED
        self._client = None
        self._example_cache: Dict[str, FewShotExample] = {}

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
            logger.info(f"[AtlasVLMFewShot] Initialized with model={self.model}")
        except Exception as e:
            logger.error(f"[AtlasVLMFewShot] Client init failed: {e}")
            self._client = None
            self.enabled = False

    def build_few_shot_prompt(self, cases: List[AtlasCase],
                               max_examples: int = 3) -> str:
        examples_text = []
        for i, case in enumerate(cases[:max_examples], 1):
            error_list = []
            for e in case.errors_found[:5]:
                error_list.append(f"  - [{e.get('severity', '?')}] {e.get('type', '')}: {e.get('description', '')}")

            correction_list = []
            for c in case.corrections[:3]:
                correction_list.append(f"  - {c.get('description', '')}")

            example = f"""示例{i}（类别: {case.category}, 错误类型: {case.error_type}）:
检测到的错误:
{chr(10).join(error_list) if error_list else '  无'}
修正建议:
{chr(10).join(correction_list) if correction_list else '  无'}
国标依据: {', '.join(case.gb_references) if case.gb_references else '无'}"""
            examples_text.append(example)

        prompt = f"""以下是{len(cases[:max_examples])}个历史相似案例的纠错示例，请参考这些案例进行当前图纸的分析：

{chr(10).join(examples_text)}

请参考以上案例的分析模式，对当前图纸进行类似的错误检测和修正建议。"""
        return prompt

    def analyze_with_few_shot(self, image_path: str, cases: List[AtlasCase],
                               analysis_context: str = "",
                               max_examples: int = 3) -> Optional[Dict]:
        if not self.enabled or self._client is None:
            return None

        try:
            image_b64 = self._encode_image(image_path)
            if image_b64 is None:
                return None

            few_shot_prompt = self.build_few_shot_prompt(cases, max_examples)

            full_prompt = few_shot_prompt
            if analysis_context:
                full_prompt += f"\n\n当前图纸的初步分析结果：\n{analysis_context}"

            full_prompt += "\n\n请以JSON格式返回你的分析结果。"

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                            {"type": "text", "text": full_prompt}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.2
            )

            result_text = response.choices[0].message.content
            return self._parse_response(result_text)

        except Exception as e:
            logger.error(f"[AtlasVLMFewShot] Analysis failed: {e}")
            return None

    def _encode_image(self, image_path: str) -> Optional[str]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"[AtlasVLMFewShot] Image encoding failed: {e}")
            return None

    def _parse_response(self, response_text: str) -> Optional[Dict]:
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                return json.loads(response_text[start_idx:end_idx])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[AtlasVLMFewShot] Failed to parse response: {e}")
        return None

    def is_available(self) -> bool:
        return self.enabled and self._client is not None
