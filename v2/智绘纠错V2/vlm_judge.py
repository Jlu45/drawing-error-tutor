"""
VLM评审Agent (VLM Judge)
=========================
借鉴 ArtiCAD 的 VLM-as-a-Judge 评估范式。

核心功能：
1. Chain-of-Thought评审流程：描述 → 比较 → 分析 → 打分
2. 四维评分：准确性 / 完整性 / 有用性 / 引导性
3. 多评委一致性验证（Krippendorff's α）
4. 评审结果作为RL精细奖励信号

与原版RL二元反馈（confirmed/ignored）的区别：
- 原版：标量信号，粒度粗
- V2：4维连续评分，可作为精细奖励信号，大幅提升RL收敛速度
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger("VLMJudge")


@dataclass
class JudgeScore:
    """四维评审评分"""
    accuracy: float = 0.0       # 准确性：检测到的错误是否真实存在 (0-5)
    completeness: float = 0.0   # 完整性：是否遗漏了重要错误 (0-5)
    helpfulness: float = 0.0    # 有用性：修正建议是否具体可操作 (0-5)
    guidance: float = 0.0       # 引导性：苏格拉底式提问是否有效 (0-5)

    @property
    def average(self) -> float:
        return (self.accuracy + self.completeness + self.helpfulness + self.guidance) / 4.0

    @property
    def weighted(self) -> float:
        """加权总分（准确性权重最高）"""
        return 0.35 * self.accuracy + 0.25 * self.completeness + \
               0.2 * self.helpfulness + 0.2 * self.guidance

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'completeness': self.completeness,
            'helpfulness': self.helpfulness,
            'guidance': self.guidance,
            'average': self.average,
            'weighted': self.weighted
        }


@dataclass
class JudgeResult:
    """评审结果"""
    judge_model: str = ""
    scores: JudgeScore = field(default_factory=JudgeScore)
    reasoning: str = ""           # Chain-of-thought推理过程
    verified_errors: List[Dict] = field(default_factory=list)   # 验证后的错误
    false_positives: List[Dict] = field(default_factory=list)   # 假阳性
    missed_errors: List[Dict] = field(default_factory=list)     # 漏检
    suggestions: List[str] = field(default_factory=list)        # 改进建议
    execution_time_ms: float = 0.0
    success: bool = False

    def to_rl_reward(self) -> float:
        w = self.scores.weighted
        if w >= 4.0:
            return 1.0
        elif w >= 3.0:
            return 0.5
        elif w >= 2.0:
            return 0.1
        elif w >= 1.0:
            return -0.3
        else:
            return -1.0

    def to_dict(self) -> Dict:
        return {
            'judge_model': self.judge_model,
            'scores': self.scores.to_dict(),
            'reasoning': self.reasoning,
            'verified_errors': self.verified_errors,
            'false_positives': self.false_positives,
            'missed_errors': self.missed_errors,
            'suggestions': self.suggestions,
            'execution_time_ms': self.execution_time_ms,
            'success': self.success,
            'rl_reward': self.to_rl_reward()
        }


class VLMJudge:
    """
    VLM评审Agent

    使用VLM（如GPT-4V、Claude-3、Qwen-VL等）对纠错报告进行评审。
    采用Chain-of-Thought评审流程确保可解释性。
    """

    # Chain-of-Thought评审提示词
    JUDGE_SYSTEM_PROMPT = """你是一个工程图纸纠错质量的评审专家。你的任务是评审一份纠错报告的质量。

评审流程（请严格按以下步骤进行）：

**第一步：描述（Describe）**
仔细观察图纸图像，描述你看到的图纸类型、内容和特征。

**第二步：比较（Compare）**
将你的观察与纠错报告中的错误列表逐一比较。

**第三步：分析（Analyze）**
对每个报告中的错误，判断：
- 该错误是否真实存在？（真阳性 vs 假阳性）
- 是否有报告遗漏的重要错误？（漏检）
- 修正建议是否具体可操作？
- 苏格拉底式引导是否有效？

**第四步：打分（Score）**
在四个维度上给出1-5分评分：
1. 准确性（Accuracy）：检测到的错误是否真实存在
2. 完整性（Completeness）：是否遗漏了重要错误
3. 有用性（Helpfulness）：修正建议是否具体可操作
4. 引导性（Guidance）：苏格拉底式提问是否有效启发思考

请以JSON格式返回评审结果。"""

    JUDGE_USER_TEMPLATE = """请评审以下纠错报告：

## 图纸信息
- 图纸类型：{drawing_type}
- 内容概述：{content_summary}

## 纠错报告
- 总错误数：{total_errors}
- 总体评分：{overall_score}/100
- 总体评价：{summary}

### 错误列表
{error_list}

### 学习引导
{feedback}

请按评审流程（描述→比较→分析→打分）进行评审，以JSON格式返回：
```json
{{
  "description": "你对图纸的观察描述",
  "comparison": "你与报告的比较分析",
  "analysis": "逐项分析结果",
  "scores": {{
    "accuracy": 1-5,
    "completeness": 1-5,
    "helpfulness": 1-5,
    "guidance": 1-5
  }},
  "verified_errors": ["验证为真实存在的错误索引"],
  "false_positives": ["误报的错误索引和原因"],
  "missed_errors": ["报告遗漏的错误描述"],
  "suggestions": ["改进建议"]
}}
```"""

    def __init__(self, api_url: str, api_key: str, model: str = "qwen-vl-max"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.client = None
        self._initialize()

    def _initialize(self):
        try:
            from openai import OpenAI
            base_url = self.api_url.rstrip('/')
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=60.0)
            logger.info(f"[VLM Judge] 初始化成功，模型: {self.model}")
        except Exception as e:
            logger.error(f"[VLM Judge] 初始化失败: {e}")

    def judge(self, image_path: str, analysis_result: Dict) -> JudgeResult:
        """
        执行评审

        Args:
            image_path: 图纸图像路径
            analysis_result: 纠错分析结果（来自Orchestrator）

        Returns:
            JudgeResult: 评审结果
        """
        start = time.time()
        result = JudgeResult(judge_model=self.model)

        if not self.client:
            result.reasoning = "VLM Judge未初始化，无法评审"
            return result

        try:
            # 构建评审请求
            report = analysis_result.get('report', {})
            errors = analysis_result.get('errors', [])
            feedback = analysis_result.get('feedback', [])

            error_list = ""
            for i, err in enumerate(errors):
                error_list += f"{i+1}. [{err.get('severity', '?')}] {err.get('type', '')}: {err.get('description', '')}\n"
                if err.get('suggestion'):
                    error_list += f"   建议: {err['suggestion']}\n"

            feedback_text = "\n".join(f"- {f}" for f in feedback[:5])

            user_prompt = self.JUDGE_USER_TEMPLATE.format(
                drawing_type=report.get('drawing_type', '未知'),
                content_summary=report.get('content_summary', '未知'),
                total_errors=report.get('total_errors', 0),
                overall_score=report.get('overall_score', 0),
                summary=report.get('summary', ''),
                error_list=error_list or "无",
                feedback=feedback_text or "无"
            )

            # 调用VLM API（传入图像+文本）
            import base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            messages = [
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.2
            )

            result_text = response.choices[0].message.content
            result.reasoning = result_text
            result.success = True

            # 解析JSON结果
            try:
                start_idx = result_text.find('{')
                end_idx = result_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    judge_data = json.loads(result_text[start_idx:end_idx])
                    scores = judge_data.get('scores', {})
                    result.scores = JudgeScore(
                        accuracy=float(scores.get('accuracy', 0)),
                        completeness=float(scores.get('completeness', 0)),
                        helpfulness=float(scores.get('helpfulness', 0)),
                        guidance=float(scores.get('guidance', 0))
                    )
                    result.verified_errors = judge_data.get('verified_errors', [])
                    result.false_positives = judge_data.get('false_positives', [])
                    result.missed_errors = judge_data.get('missed_errors', [])
                    result.suggestions = judge_data.get('suggestions', [])
            except json.JSONDecodeError as e:
                logger.warning(f"[VLM Judge] JSON解析失败: {e}")

        except Exception as e:
            logger.error(f"[VLM Judge] 评审失败: {e}")
            result.reasoning = f"评审过程出错: {str(e)}"
            result.success = False

        result.execution_time_ms = (time.time() - start) * 1000
        logger.info(f"[VLM Judge] 评审完成: {result.scores.weighted:.2f}/5.00 "
                    f"({result.execution_time_ms:.0f}ms)")
        return result


class MultiJudgeConsensus:
    """
    多评委一致性验证

    借鉴 ArtiCAD 使用多个VLM独立评分并通过Krippendorff's α验证一致性的方法。
    """

    def __init__(self, judges: List[VLMJudge]):
        self.judges = judges

    def evaluate(self, image_path: str, analysis_result: Dict) -> Dict:
        """
        多评委评审并计算一致性

        Returns:
            {
                'individual_results': List[JudgeResult],
                'consensus_score': JudgeScore,
                'agreement': float,  # Krippendorff's α近似值
                'is_reliable': bool
            }
        """
        results = []
        for judge in self.judges:
            result = judge.judge(image_path, analysis_result)
            results.append(result)

        # 计算平均分
        if results:
            avg_scores = JudgeScore(
                accuracy=sum(r.scores.accuracy for r in results) / len(results),
                completeness=sum(r.scores.completeness for r in results) / len(results),
                helpfulness=sum(r.scores.helpfulness for r in results) / len(results),
                guidance=sum(r.scores.guidance for r in results) / len(results)
            )
        else:
            avg_scores = JudgeScore()

        # 计算简化的一致性指标（标准差/均值）
        if len(results) >= 2:
            avg_weighted = avg_scores.weighted
            if avg_weighted > 0:
                variance = sum((r.scores.weighted - avg_weighted) ** 2 for r in results) / len(results)
                std = variance ** 0.5
                agreement = max(0, 1 - std / avg_weighted)  # 近似一致性
            else:
                agreement = 0
        else:
            agreement = 1.0  # 单评委视为完全一致

        return {
            'individual_results': [r.to_dict() for r in results],
            'consensus_score': avg_scores.to_dict(),
            'agreement': round(agreement, 3),
            'is_reliable': agreement >= 0.5
        }
