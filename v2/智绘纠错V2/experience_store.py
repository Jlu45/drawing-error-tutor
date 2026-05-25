"""
非对称经验库 (Asymmetric Experience Store)
============================================
借鉴 ArtiCAD 的 Self-Evolving Experience Store。

核心创新：
1. 双分区存储：Good Cases（正向案例）+ Issue Cases（负向案例）
2. 非对称检索策略：不同Agent按需检索不同分区
3. FAISS向量索引 + 结构化案例摘要
4. 无需微调即可持续改进检测准确率

非对称检索策略（借鉴ArtiCAD Table 2消融实验结论）：
- Planning Agent: Good + Issue（需要了解"什么好"和"什么不好"）
- Perception Agents (OCR/Geo/Struct): 仅 Good Cases（干净的检测模板）
- LLM Agent: Good Cases（少样本学习示例）
- VLM Judge: Issue Cases（校验基准，避免重复犯错）
"""

import os
import json
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger("ExperienceStore")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS未安装，将使用暴力搜索作为降级方案")


class CaseType(Enum):
    """案例类型"""
    GOOD = "good"          # 正向案例：正确检测 + 有效导学
    ISSUE = "issue"        # 负向案例：漏检 / 误检 / 无效导学


class AgentRole(Enum):
    """Agent角色，决定检索策略"""
    PLANNING = "planning"      # 规划Agent → Good + Issue
    PERCEPTION = "perception"  # 感知Agent → 仅 Good
    ANALYSIS = "analysis"      # 分析Agent (LLM) → Good
    JUDGE = "judge"            # 评审Agent (VLM Judge) → Issue


@dataclass
class ExperienceCase:
    """
    单条经验案例

    包含：
    - 案例元数据（类型、时间、来源）
    - 图纸特征向量（用于相似度检索）
    - 结构化摘要（用于上下文注入）
    - 反馈信号（用于RL奖励）
    """
    case_id: str = ""
    case_type: CaseType = CaseType.GOOD
    timestamp: float = field(default_factory=time.time)
    source_session: str = ""

    # 图纸特征
    drawing_type: str = ""           # 图纸类型
    feature_vector: Optional[np.ndarray] = None  # 特征向量

    # 结构化摘要
    summary: str = ""                # 案例摘要
    error_category: str = ""         # 错误类别
    detection_result: str = ""       # 检测结果描述
    correction_applied: str = ""     # 采取的修正措施
    user_feedback: str = ""          # 用户反馈

    # 评分
    accuracy_score: float = 0.0      # 检测准确性 0-1
    helpfulness_score: float = 0.0   # 导学有用性 0-1
    overall_score: float = 0.0       # 综合评分

    # 嵌入向量（用于FAISS检索）
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        d = {
            'case_id': self.case_id,
            'case_type': self.case_type.value,
            'timestamp': self.timestamp,
            'source_session': self.source_session,
            'drawing_type': self.drawing_type,
            'summary': self.summary,
            'error_category': self.error_category,
            'detection_result': self.detection_result,
            'correction_applied': self.correction_applied,
            'user_feedback': self.user_feedback,
            'accuracy_score': self.accuracy_score,
            'helpfulness_score': self.helpfulness_score,
            'overall_score': self.overall_score,
        }
        if self.feature_vector is not None:
            d['feature_vector'] = self.feature_vector.tolist()
        if self.embedding is not None:
            d['embedding'] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperienceCase':
        d = d.copy()
        if 'case_type' in d and isinstance(d['case_type'], str):
            d['case_type'] = CaseType(d['case_type'])
        if 'feature_vector' in d and isinstance(d['feature_vector'], list):
            d['feature_vector'] = np.array(d['feature_vector'], dtype=np.float32)
        if 'embedding' in d and isinstance(d['embedding'], list):
            d['embedding'] = np.array(d['embedding'], dtype=np.float32)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PartitionIndex:
    """
    分区向量索引

    每个分区（Good/Issue）维护独立的FAISS索引。
    支持添加、检索、持久化操作。
    """

    def __init__(self, dimension: int = 128, partition_name: str = ""):
        self.dimension = dimension
        self.partition_name = partition_name
        self.cases: List[ExperienceCase] = []
        self._embeddings: List[np.ndarray] = []
        self._index = None
        self._needs_rebuild = True

    def add(self, case: ExperienceCase):
        """添加一个案例"""
        self.cases.append(case)
        if case.embedding is not None:
            self._embeddings.append(case.embedding)
            self._needs_rebuild = True

    def _rebuild_index(self):
        """重建FAISS索引"""
        if not self._embeddings:
            self._index = None
            return

        embeddings_matrix = np.stack(self._embeddings).astype(np.float32)

        if HAS_FAISS:
            # 使用FAISS IndexFlatL2（精确搜索，适合中小规模数据）
            self._index = faiss.IndexFlatL2(self.dimension)
            # 归一化向量，使用内积搜索（余弦相似度）
            faiss.normalize_L2(embeddings_matrix)
            self._index.add(embeddings_matrix)
        else:
            # 降级方案：存储为numpy数组
            self._index = embeddings_matrix

        self._needs_rebuild = False

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[ExperienceCase, float]]:
        """
        检索最相似的案例

        Returns:
            List of (case, similarity_score) tuples
        """
        if self._needs_rebuild:
            self._rebuild_index()

        if not self.cases:
            return []

        query = query_embedding.astype(np.float32).reshape(1, -1)

        if HAS_FAISS and self._index is not None:
            faiss.normalize_L2(query)
            distances, indices = self._index.search(query, min(top_k, len(self.cases)))
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.cases):
                    # L2距离转相似度（越小越相似）
                    similarity = 1.0 / (1.0 + dist)
                    results.append((self.cases[idx], similarity))
            return results
        elif self._index is not None:
            # 暴力搜索降级方案
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            index_norm = self._index / (np.linalg.norm(self._index, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(index_norm, query_norm.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.cases[i], float(similarities[i])) for i in top_indices]
        else:
            # 无嵌入向量的降级：按评分排序
            sorted_cases = sorted(self.cases, key=lambda c: c.overall_score, reverse=True)
            return [(c, c.overall_score) for c in sorted_cases[:top_k]]

    @property
    def size(self) -> int:
        return len(self.cases)

    def get_top_cases(self, n: int = 5) -> List[ExperienceCase]:
        """按综合评分获取Top N案例"""
        return sorted(self.cases, key=lambda c: c.overall_score, reverse=True)[:n]


class AsymmetricExperienceStore:
    """
    非对称经验库

    双分区设计 + 非对称检索策略：

    检索策略矩阵：
    ┌──────────────┬────────────┬────────────┐
    │ Agent Role   │ Good Cases │ Issue Cases│
    ├──────────────┼────────────┼────────────┤
    │ Planning     │     ✓      │     ✓      │
    │ Perception   │     ✓      │     ✗      │
    │ Analysis     │     ✓      │     ✗      │
    │ Judge        │     ✗      │     ✓      │
    └──────────────┴────────────┴────────────┘
    """

    def __init__(self, embedding_dim: int = 128, persist_dir: str = ""):
        self.embedding_dim = embedding_dim
        self.persist_dir = persist_dir
        self.good_partition = PartitionIndex(embedding_dim, "good")
        self.issue_partition = PartitionIndex(embedding_dim, "issue")

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load()

    def add_case(self, case: ExperienceCase):
        """添加案例到对应分区"""
        if case.case_type == CaseType.GOOD:
            self.good_partition.add(case)
        else:
            self.issue_partition.add(case)

        logger.info(f"[经验库] 添加{case.case_type.value}案例: {case.case_id} "
                    f"(评分={case.overall_score:.2f})")

        if self.persist_dir:
            self._save_case(case)

    def retrieve(self, query_embedding: np.ndarray, role: AgentRole,
                 top_k: int = 5) -> List[Tuple[ExperienceCase, float]]:
        """
        非对称检索：根据Agent角色决定检索哪些分区

        Args:
            query_embedding: 查询向量
            role: Agent角色
            top_k: 每个分区返回的最大案例数

        Returns:
            按相似度排序的 (case, score) 列表
        """
        results = []

        if role == AgentRole.PLANNING:
            # Planning需要同时了解正向和负向案例
            good_results = self.good_partition.search(query_embedding, top_k)
            issue_results = self.issue_partition.search(query_embedding, top_k)
            results = good_results + issue_results
            # 交替排列，确保正负案例都有机会被看到
            results.sort(key=lambda x: x[1], reverse=True)

        elif role == AgentRole.PERCEPTION:
            # 感知Agent只需要正向案例作为检测模板
            results = self.good_partition.search(query_embedding, top_k)

        elif role == AgentRole.ANALYSIS:
            # 分析Agent只需要正向案例作为少样本示例
            results = self.good_partition.search(query_embedding, top_k)

        elif role == AgentRole.JUDGE:
            # 评审Agent只需要负向案例作为校验基准
            results = self.issue_partition.search(query_embedding, top_k)

        return results

    def retrieve_by_text(self, query_text: str, role: AgentRole,
                         top_k: int = 5) -> List[Tuple[ExperienceCase, float]]:
        """
        基于文本的检索（当没有嵌入向量时的降级方案）

        使用简单的TF-IDF相似度进行检索。
        """
        # 简单的词袋相似度
        query_words = set(query_text.lower().split())

        def text_similarity(case: ExperienceCase) -> float:
            case_text = f"{case.summary} {case.error_category} {case.detection_result}".lower()
            case_words = set(case_text.split())
            if not query_words or not case_words:
                return 0.0
            intersection = query_words & case_words
            union = query_words | case_words
            return len(intersection) / len(union)

        all_cases = []
        if role in (AgentRole.PLANNING, AgentRole.PERCEPTION, AgentRole.ANALYSIS):
            all_cases.extend(self.good_partition.cases)
        if role in (AgentRole.PLANNING, AgentRole.JUDGE):
            all_cases.extend(self.issue_partition.cases)

        scored = [(case, text_similarity(case)) for case in all_cases]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def create_case_from_feedback(self, session_id: str, drawing_type: str,
                                   error_category: str, detection_result: str,
                                   correction: str, feedback_type: str,
                                   accuracy: float = 0.0, helpfulness: float = 0.0,
                                   feature_vector: Optional[np.ndarray] = None) -> ExperienceCase:
        """
        从用户反馈创建经验案例

        Args:
            feedback_type: "confirmed" | "ignored" | "dismissed_all" | "partial_confirm" | "useful_guidance"
        """
        # 根据反馈类型确定案例分类
        if feedback_type in ('confirmed', 'useful_guidance'):
            case_type = CaseType.GOOD
        elif feedback_type in ('ignored', 'dismissed_all'):
            case_type = CaseType.ISSUE
        else:  # partial_confirm
            case_type = CaseType.GOOD if helpfulness > 0.5 else CaseType.ISSUE

        overall = 0.4 * accuracy + 0.6 * helpfulness

        case = ExperienceCase(
            case_id=f"{session_id[:8]}_{int(time.time())}",
            case_type=case_type,
            source_session=session_id,
            drawing_type=drawing_type,
            feature_vector=feature_vector,
            summary=f"[{drawing_type}] {error_category}: {detection_result[:100]}",
            error_category=error_category,
            detection_result=detection_result,
            correction_applied=correction,
            user_feedback=feedback_type,
            accuracy_score=accuracy,
            helpfulness_score=helpfulness,
            overall_score=overall
        )

        self.add_case(case)
        return case

    def get_context_for_agent(self, role: AgentRole, query: str = "",
                               top_k: int = 3) -> str:
        """
        为Agent生成经验上下文文本（注入到Prompt中）

        Args:
            role: Agent角色
            query: 查询文本
            top_k: 返回案例数

        Returns:
            格式化的经验上下文字符串
        """
        if query:
            results = self.retrieve_by_text(query, role, top_k)
        else:
            if role in (AgentRole.PLANNING, AgentRole.JUDGE):
                results = ([(c, c.overall_score) for c in self.good_partition.get_top_cases(top_k)] +
                           [(c, c.overall_score) for c in self.issue_partition.get_top_cases(top_k)])
            elif role in (AgentRole.PERCEPTION, AgentRole.ANALYSIS):
                results = [(c, c.overall_score) for c in self.good_partition.get_top_cases(top_k)]
            else:
                results = []

        if not results:
            return ""

        lines = ["【历史经验参考】"]
        for case, score in results:
            prefix = "✓" if case.case_type == CaseType.GOOD else "✗"
            lines.append(f"{prefix} [{case.error_category}] {case.summary}")
            if case.correction_applied:
                lines.append(f"  修正: {case.correction_applied[:100]}")
            lines.append(f"  评分: 准确性={case.accuracy_score:.1f}, 有用性={case.helpfulness_score:.1f}")

        return "\n".join(lines)

    @property
    def stats(self) -> Dict:
        return {
            'good_cases': self.good_partition.size,
            'issue_cases': self.issue_partition.size,
            'total_cases': self.good_partition.size + self.issue_partition.size,
            'embedding_dim': self.embedding_dim,
            'has_faiss': HAS_FAISS
        }

    def _save_case(self, case: ExperienceCase):
        """持久化单个案例"""
        try:
            filepath = os.path.join(self.persist_dir,
                                    f"{case.case_type.value}_{case.case_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(case.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[经验库] 保存案例失败: {e}")

    def _load(self):
        """从磁盘加载所有案例"""
        if not os.path.exists(self.persist_dir):
            return

        loaded = 0
        for filename in os.listdir(self.persist_dir):
            if not filename.endswith('.json'):
                continue
            try:
                filepath = os.path.join(self.persist_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    case = ExperienceCase.from_dict(json.load(f))
                self.add_case(case)
                loaded += 1
            except Exception as e:
                logger.warning(f"[经验库] 加载案例失败 {filename}: {e}")

        if loaded > 0:
            logger.info(f"[经验库] 从磁盘加载 {loaded} 条案例")
