import os
import json
import time
import hashlib
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("ExperienceStore")

try:
    from config_loader import (
        EXPERIENCE_STORE_DIR as _STORE_DIR,
        EXPERIENCE_EMBEDDING_DIM as _EMB_DIM,
        EXPERIENCE_TOP_K as _TOP_K
    )
    STORE_DIR = _STORE_DIR
    EMBEDDING_DIM = _EMB_DIM
    TOP_K = _TOP_K
except ImportError:
    STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'experience_store')
    EMBEDDING_DIM = 128
    TOP_K = 5

os.makedirs(STORE_DIR, exist_ok=True)


@dataclass
class ExperienceCase:
    case_id: str = ""
    image_hash: str = ""
    drawing_type: str = ""
    errors: List[Dict] = field(default_factory=list)
    error_summary: str = ""
    overall_score: float = 0.0
    feedback_types: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.embedding is not None:
            d['embedding'] = [float(x) for x in self.embedding]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ExperienceCase':
        if 'embedding' in d and d['embedding'] is not None:
            d['embedding'] = [float(x) for x in d['embedding']]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SimpleEncoder:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._hash_seed = 42

    def encode(self, analysis_result: Dict) -> np.ndarray:
        ocr_texts = analysis_result.get('ocr_results', [])
        geo = analysis_result.get('geo_result', {})
        report = analysis_result.get('report', {})

        features = []
        ocr_text_concat = " ".join([t.get('text', '') for t in ocr_texts[:20]])
        features.append(hashlib.md5(ocr_text_concat.encode()).hexdigest())

        features.append(str(len(ocr_texts)))
        features.append(str(len(geo.get('lines', [])) if geo else 0))
        features.append(str(len(geo.get('circles', [])) if geo else 0))
        features.append(str(report.get('total_errors', 0)))
        features.append(str(int(report.get('overall_score', 0))))

        error_types = set()
        for e in analysis_result.get('errors', []):
            error_types.add(e.get('type', ''))
        features.append(",".join(sorted(error_types)))

        feature_str = "|".join(features)
        rng = np.random.RandomState(self._hash_seed)
        embedding = np.zeros(self.dim, dtype=np.float32)
        for i, ch in enumerate(feature_str):
            idx = i % self.dim
            embedding[idx] += ord(ch) * 0.01

        for i in range(self.dim):
            embedding[i] += rng.randn() * 0.001

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding


class FaissIndex:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._index = None
        self._id_map: List[str] = []
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
            logger.info(f"[FaissIndex] Initialized FAISS index, dim={dim}")
        except ImportError:
            logger.warning("[FaissIndex] FAISS not available, using numpy fallback")
            self._index = None
        self._vectors: Optional[np.ndarray] = None

    def add(self, case_id: str, embedding: np.ndarray):
        vec = embedding.reshape(1, -1).astype(np.float32)
        if self._index is not None:
            import faiss
            self._index.add(vec)
        else:
            if self._vectors is None:
                self._vectors = vec
            else:
                self._vectors = np.vstack([self._vectors, vec])
        self._id_map.append(case_id)

    def search(self, query: np.ndarray, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        if len(self._id_map) == 0:
            return []

        query_vec = query.reshape(1, -1).astype(np.float32)

        if self._index is not None and self._index.ntotal > 0:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(query_vec, k)
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if 0 <= idx < len(self._id_map):
                    results.append((self._id_map[idx], float(scores[0][i])))
            return results
        elif self._vectors is not None and len(self._vectors) > 0:
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normalized = self._vectors / norms
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
            similarities = (normalized @ query_norm.T).flatten()
            k = min(top_k, len(similarities))
            top_indices = np.argsort(similarities)[::-1][:k]
            return [(self._id_map[idx], float(similarities[idx])) for idx in top_indices]
        return []

    def size(self) -> int:
        return len(self._id_map)

    def save(self, path: str):
        data = {
            'id_map': self._id_map,
            'dim': self.dim
        }
        if self._vectors is not None:
            np.save(path + '_vectors.npy', self._vectors)
        with open(path + '_meta.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        meta_path = path + '_meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._id_map = data.get('id_map', [])
            self.dim = data.get('dim', self.dim)

        vec_path = path + '_vectors.npy'
        if os.path.exists(vec_path):
            self._vectors = np.load(vec_path)
            if self._index is not None:
                import faiss
                self._index = faiss.IndexFlatIP(self.dim)
                self._index.add(self._vectors.astype(np.float32))


class ExperienceStore:
    def __init__(self, dim: int = EMBEDDING_DIM, top_k: int = TOP_K):
        self.dim = dim
        self.top_k = top_k
        self._encoder = SimpleEncoder(dim=dim)
        self._index = FaissIndex(dim=dim)
        self._cases: Dict[str, ExperienceCase] = {}
        self._lock = threading.Lock()
        self._persist_path = os.path.join(STORE_DIR, 'experience_store')
        self._load()

    def store(self, analysis_result: Dict, feedback_types: Optional[List[str]] = None) -> str:
        with self._lock:
            case_id = self._generate_case_id(analysis_result)
            embedding = self._encoder.encode(analysis_result)

            errors = analysis_result.get('errors', [])
            report = analysis_result.get('report', {})

            case = ExperienceCase(
                case_id=case_id,
                image_hash=self._hash_image_path(analysis_result),
                drawing_type=analysis_result.get('api_result', {}).get('model', 'unknown'),
                errors=errors[:20],
                error_summary=report.get('summary', ''),
                overall_score=report.get('overall_score', 0),
                feedback_types=feedback_types or [],
                embedding=embedding.tolist(),
                timestamp=time.time()
            )

            if case_id in self._cases:
                old_case = self._cases[case_id]
                if feedback_types:
                    case.feedback_types = list(set(old_case.feedback_types + feedback_types))
                case.timestamp = time.time()

            self._cases[case_id] = case
            self._index.add(case_id, embedding)

            logger.info(f"[ExperienceStore] Stored case {case_id[:8]}..., "
                        f"total={len(self._cases)}")
            return case_id

    def retrieve(self, analysis_result: Dict, top_k: Optional[int] = None) -> List[Tuple[ExperienceCase, float]]:
        k = top_k or self.top_k
        query_embedding = self._encoder.encode(analysis_result)

        with self._lock:
            results = self._index.search(query_embedding, k)

        cases = []
        for case_id, score in results:
            case = self._cases.get(case_id)
            if case is not None:
                cases.append((case, score))
        return cases

    def get_similar_errors(self, analysis_result: Dict, top_k: Optional[int] = None) -> List[Dict]:
        similar_cases = self.retrieve(analysis_result, top_k)
        all_errors = []
        seen_descriptions = set()

        for case, score in similar_cases:
            for error in case.errors:
                desc = error.get('description', '')
                if desc and desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    all_errors.append({
                        **error,
                        'similarity': round(score, 3),
                        'source_case': case.case_id[:8],
                        'historical_feedback': case.feedback_types
                    })
        return all_errors

    def get_stats(self) -> Dict:
        total_cases = len(self._cases)
        feedback_counts = {}
        for case in self._cases.values():
            for ft in case.feedback_types:
                feedback_counts[ft] = feedback_counts.get(ft, 0) + 1

        return {
            'total_cases': total_cases,
            'index_size': self._index.size(),
            'embedding_dim': self.dim,
            'top_k': self.top_k,
            'feedback_distribution': feedback_counts
        }

    def save(self):
        with self._lock:
            self._index.save(self._persist_path + '_index')
            cases_data = {cid: case.to_dict() for cid, case in self._cases.items()}
            cases_path = self._persist_path + '_cases.json'
            with open(cases_path, 'w', encoding='utf-8') as f:
                json.dump(cases_data, f, ensure_ascii=False, indent=2)
            logger.info(f"[ExperienceStore] Saved {len(self._cases)} cases")

    def _load(self):
        cases_path = self._persist_path + '_cases.json'
        if os.path.exists(cases_path):
            try:
                with open(cases_path, 'r', encoding='utf-8') as f:
                    cases_data = json.load(f)
                for cid, case_dict in cases_data.items():
                    case = ExperienceCase.from_dict(case_dict)
                    self._cases[cid] = case
                    if case.embedding is not None:
                        embedding = np.array(case.embedding, dtype=np.float32)
                        self._index.add(cid, embedding)
                logger.info(f"[ExperienceStore] Loaded {len(self._cases)} cases")
            except Exception as e:
                logger.warning(f"[ExperienceStore] Load failed: {e}")

    def _generate_case_id(self, analysis_result: Dict) -> str:
        errors = analysis_result.get('errors', [])
        error_descs = "|".join(sorted([e.get('description', '') for e in errors[:10]]))
        score = analysis_result.get('report', {}).get('overall_score', 0)
        key = f"{error_descs}|{score}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _hash_image_path(self, analysis_result: Dict) -> str:
        return ""
