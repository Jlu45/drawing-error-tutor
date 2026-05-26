import os
import json
import time
import random
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque

logger = logging.getLogger("RLMemory")

try:
    from config_loader import (
        RL_STATE_DIM as _STATE_DIM,
        RL_ACTION_DIM as _ACTION_DIM,
        RL_BUFFER_CAPACITY as _BUFFER_CAP,
        RL_LEARNING_RATE as _LR,
        RL_GAMMA as _GAMMA,
        RL_EPSILON_START as _EPS_START,
        RL_EPSILON_MIN as _EPS_MIN,
        RL_EPSILON_DECAY as _EPS_DECAY,
        RL_EXPERIENCE_DIR as _RL_DIR
    )
    STATE_DIM = _STATE_DIM
    ACTION_DIM = _ACTION_DIM
    BUFFER_CAPACITY = _BUFFER_CAP
    LEARNING_RATE = _LR
    GAMMA = _GAMMA
    EPSILON_START = _EPS_START
    EPSILON_MIN = _EPS_MIN
    EPSILON_DECAY = _EPS_DECAY
    EXPERIENCE_DIR = _RL_DIR
except ImportError:
    STATE_DIM = 10
    ACTION_DIM = 15
    BUFFER_CAPACITY = 500
    LEARNING_RATE = 0.01
    GAMMA = 0.95
    EPSILON_START = 0.3
    EPSILON_MIN = 0.05
    EPSILON_DECAY = 0.995
    EXPERIENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'rl_experience')

os.makedirs(EXPERIENCE_DIR, exist_ok=True)


@dataclass
class RLState:
    ocr_count_norm: float = 0.0
    ocr_confidence: float = 0.0
    line_count_norm: float = 0.0
    circle_count_norm: float = 0.0
    arrow_count_norm: float = 0.0
    has_title_block: float = 0.0
    total_errors_norm: float = 0.0
    high_errors_norm: float = 0.0
    quality_score: float = 0.0
    llm_fusion_ratio: float = 0.5

    def to_array(self) -> np.ndarray:
        return np.array([
            self.ocr_count_norm, self.ocr_confidence,
            self.line_count_norm, self.circle_count_norm,
            self.arrow_count_norm, self.has_title_block,
            self.total_errors_norm, self.high_errors_norm,
            self.quality_score, self.llm_fusion_ratio
        ], dtype=np.float32)

    @classmethod
    def from_analysis(cls, result: Dict, current_fusion_ratio: float = 0.5) -> 'RLState':
        ocr_texts = result.get('ocr_results', [])
        geo = result.get('geo_result', {})
        structure = result.get('structure_result', {})
        report = result.get('report', {})

        ocr_count = len(ocr_texts)
        high_conf = sum(1 for t in ocr_texts if t.get('confidence', 0) > 0.7)
        high_conf_ratio = high_conf / max(ocr_count, 1)

        return cls(
            ocr_count_norm=min(ocr_count / 50.0, 1.0),
            ocr_confidence=high_conf_ratio,
            line_count_norm=min(len(geo.get('lines', [])) / 100.0, 1.0) if geo else 0.0,
            circle_count_norm=min(len(geo.get('circles', [])) / 20.0, 1.0) if geo else 0.0,
            arrow_count_norm=min(len(geo.get('arrows', [])) / 20.0, 1.0) if geo else 0.0,
            has_title_block=1.0 if structure.get('title_block', {}).get('detected', False) else 0.0,
            total_errors_norm=min(report.get('total_errors', 0) / 20.0, 1.0),
            high_errors_norm=min(report.get('error_categories', {}).get('高', 0) / 10.0, 1.0),
            quality_score=result.get('metrics', {}).get('quality_score', 0.0),
            llm_fusion_ratio=current_fusion_ratio
        )


ACTION_NAMES = [
    "inc_sev_high", "dec_sev_high",
    "inc_sev_med", "dec_sev_med",
    "inc_sev_low", "dec_sev_low",
    "inc_penalty", "dec_penalty",
    "inc_llm_ratio", "dec_llm_ratio",
    "inc_ocr_thresh", "dec_ocr_thresh",
    "inc_rule_conf", "dec_rule_conf",
    "no_change"
]


@dataclass
class RLAction:
    action_id: int
    name: str
    param_index: int = -1
    delta: float = 0.0

    @classmethod
    def from_id(cls, action_id: int) -> 'RLAction':
        name = ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else "unknown"
        deltas = {
            0: (0, +0.3), 1: (0, -0.3),
            2: (1, +0.2), 3: (1, -0.2),
            4: (2, +0.1), 5: (2, -0.1),
            6: (3, +0.5), 7: (3, -0.5),
            8: (4, +0.05), 9: (4, -0.05),
            10: (5, +1.0), 11: (5, -1.0),
            12: (6, +0.05), 13: (6, -0.05),
            14: (-1, 0.0)
        }
        param_idx, delta = deltas.get(action_id, (-1, 0.0))
        return cls(action_id=action_id, name=name, param_index=param_idx, delta=delta)

    def is_noop(self) -> bool:
        return self.action_id == 14


class RewardCalculator:
    REWARD_MAP = {
        'confirmed': 1.0,
        'useful_guidance': 0.5,
        'partial_confirm': 0.3,
        'ignored': -0.5,
        'dismissed_all': -1.0
    }

    @classmethod
    def compute(cls, feedback_type: str, context: Optional[Dict] = None) -> float:
        base_reward = cls.REWARD_MAP.get(feedback_type, 0.0)
        if context and 'confidence' in context:
            confidence = context['confidence']
            base_reward *= (0.5 + 0.5 * confidence)
        return base_reward


@dataclass
class RLTransition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""

    def to_dict(self) -> Dict:
        return {
            'state': self.state, 'action': self.action, 'reward': self.reward,
            'next_state': self.next_state, 'done': self.done,
            'timestamp': self.timestamp, 'session_id': self.session_id
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'RLTransition':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MiniDQN:
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden_dim: int = 64, lr: float = LEARNING_RATE,
                 gamma: float = GAMMA, epsilon: float = EPSILON_START,
                 epsilon_min: float = EPSILON_MIN, epsilon_decay: float = EPSILON_DECAY):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()

        self.update_count = 0
        self.target_update_freq = 10

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, state: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
        h = self._relu(state @ W1 + b1)
        return h @ W2 + b2

    def predict(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        q_values = self._forward(state, self.W1, self.b1, self.W2, self.b2)
        return int(np.argmax(q_values))

    def predict_greedy(self, state: np.ndarray) -> int:
        q_values = self._forward(state, self.W1, self.b1, self.W2, self.b2)
        return int(np.argmax(q_values))

    def train_step(self, batch: List[RLTransition]):
        if len(batch) < 4:
            return

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch], dtype=np.float32)

        current_q = self._forward(states, self.W1, self.b1, self.W2, self.b2)
        next_q_target = self._forward(next_states, self.target_W1, self.target_b1,
                                       self.target_W2, self.target_b2)

        targets = current_q.copy()
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_target[i])

        h = self._relu(states @ self.W1 + self.b1)
        grad_output = (current_q - targets) / len(batch)

        grad_W2 = h.T @ grad_output
        grad_b2 = grad_output.sum(axis=0)

        grad_h = grad_output @ self.W2.T
        grad_h[h <= 0] = 0

        grad_W1 = states.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)

        self.W1 -= self.lr * np.clip(grad_W1, -1, 1)
        self.b1 -= self.lr * np.clip(grad_b1, -1, 1)
        self.W2 -= self.lr * np.clip(grad_W2, -1, 1)
        self.b2 -= self.lr * np.clip(grad_b2, -1, 1)

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_W1 = self.W1.copy()
            self.target_b1 = self.b1.copy()
            self.target_W2 = self.W2.copy()
            self.target_b2 = self.b2.copy()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        return self._forward(state, self.W1, self.b1, self.W2, self.b2)

    def save(self, path: str):
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 target_W1=self.target_W1, target_b1=self.target_b1,
                 target_W2=self.target_W2, target_b2=self.target_b2,
                 epsilon=self.epsilon, update_count=self.update_count)

    def load(self, path: str):
        if os.path.exists(path):
            data = np.load(path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.target_W1 = data['target_W1']
            self.target_b1 = data['target_b1']
            self.target_W2 = data['target_W2']
            self.target_b2 = data['target_b2']
            self.epsilon = float(data['epsilon'])
            self.update_count = int(data['update_count'])


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self.buffer: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, transition: RLTransition):
        with self._lock:
            self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[RLTransition]:
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)

    def save(self, path: str):
        with self._lock:
            data = [e.to_dict() for e in self.buffer]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            with self._lock:
                self.buffer = deque([RLTransition.from_dict(d) for d in data],
                                    maxlen=self.buffer.maxlen)


@dataclass
class PolicyParams:
    severity_weight_high: float = 3.0
    severity_weight_medium: float = 2.0
    severity_weight_low: float = 1.0
    score_penalty_per_weight: float = 5.0
    llm_score_fusion_ratio: float = 0.5
    ocr_enhance_threshold: int = 5
    rule_confidence_threshold: float = 0.3
    version: int = 0

    def to_vector(self) -> List[float]:
        return [
            self.severity_weight_high, self.severity_weight_medium, self.severity_weight_low,
            self.score_penalty_per_weight, self.llm_score_fusion_ratio,
            float(self.ocr_enhance_threshold), self.rule_confidence_threshold
        ]

    @classmethod
    def from_vector(cls, v: List[float], version: int = 0) -> 'PolicyParams':
        return cls(
            severity_weight_high=v[0], severity_weight_medium=v[1], severity_weight_low=v[2],
            score_penalty_per_weight=v[3], llm_score_fusion_ratio=v[4],
            ocr_enhance_threshold=max(1, int(v[5])),
            rule_confidence_threshold=v[6], version=version
        )

    def clamp(self) -> 'PolicyParams':
        self.severity_weight_high = np.clip(self.severity_weight_high, 1.0, 6.0)
        self.severity_weight_medium = np.clip(self.severity_weight_medium, 0.5, 4.0)
        self.severity_weight_low = np.clip(self.severity_weight_low, 0.1, 2.0)
        self.score_penalty_per_weight = np.clip(self.score_penalty_per_weight, 1.0, 10.0)
        self.llm_score_fusion_ratio = np.clip(self.llm_score_fusion_ratio, 0.1, 0.9)
        self.ocr_enhance_threshold = max(1, min(15, int(self.ocr_enhance_threshold)))
        self.rule_confidence_threshold = np.clip(self.rule_confidence_threshold, 0.05, 0.8)
        return self


class RLMemory:
    def __init__(self, state_dim: int = STATE_DIM):
        self.state_dim = state_dim
        self.policy_params = PolicyParams()
        self.dqn = MiniDQN(state_dim=state_dim)
        self.replay_buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)
        self._lock = threading.Lock()
        self._session_states: Dict[str, Dict] = {}
        self._training_count = 0
        self._min_experiences = 8
        self._train_interval = 3
        self._persist_path = os.path.join(EXPERIENCE_DIR, 'rl_state_v2')
        self._load_state()

    def extract_state(self, analysis_result: Dict) -> np.ndarray:
        rl_state = RLState.from_analysis(
            analysis_result,
            current_fusion_ratio=self.policy_params.llm_score_fusion_ratio
        )
        return rl_state.to_array()

    def select_action(self, state: np.ndarray) -> int:
        return self.dqn.predict(state)

    def apply_action(self, action_id: int) -> PolicyParams:
        action = RLAction.from_id(action_id)
        if action.is_noop():
            return self.policy_params

        with self._lock:
            vec = self.policy_params.to_vector()
            if 0 <= action.param_index < len(vec):
                vec[action.param_index] += action.delta
                self.policy_params = PolicyParams.from_vector(
                    vec, self.policy_params.version + 1
                )
                self.policy_params.clamp()
            return self.policy_params

    def get_policy_params(self) -> PolicyParams:
        with self._lock:
            return PolicyParams(**asdict(self.policy_params))

    def register_session(self, session_id: str, state: np.ndarray,
                         action_id: int, analysis_result: Dict):
        self._session_states[session_id] = {
            'state': state.tolist(),
            'action': action_id,
            'timestamp': time.time(),
            'error_ids': [e.get('description', '') for e in analysis_result.get('errors', [])],
            'quality_score': analysis_result.get('metrics', {}).get('quality_score', 0.0)
        }

    def submit_feedback(self, session_id: str, error_description: str,
                        feedback_type: str, next_state: Optional[np.ndarray] = None):
        session = self._session_states.get(session_id)
        if session is None:
            logger.warning(f"[RLMemory] Unknown session: {session_id}")
            return

        reward = RewardCalculator.compute(feedback_type, context=session)

        state = np.array(session['state'], dtype=np.float32)
        action_id = session['action']

        if next_state is None:
            next_state = state.copy()

        transition = RLTransition(
            state=state.tolist(),
            action=action_id,
            reward=reward,
            next_state=next_state.tolist(),
            done=(feedback_type in ('confirmed', 'dismissed_all')),
            session_id=session_id
        )

        self.replay_buffer.push(transition)
        self._training_count += 1

        if (self._training_count % self._train_interval == 0
                and len(self.replay_buffer) >= self._min_experiences):
            self._train()

        logger.info(f"[RLMemory] Feedback: {feedback_type}, reward={reward:.2f}, "
                     f"buffer={len(self.replay_buffer)}")

        if self._training_count % 5 == 0:
            self._save_state()

    def _train(self):
        batch = self.replay_buffer.sample(batch_size=min(16, len(self.replay_buffer)))
        if len(batch) < 4:
            return

        self.dqn.train_step(batch)

        best_action = self.dqn.predict_greedy(
            np.array(batch[-1].state, dtype=np.float32)
        )
        self.apply_action(best_action)

        logger.info(f"[RLMemory] Training step, params_version={self.policy_params.version}, "
                     f"action={ACTION_NAMES[best_action]}, epsilon={self.dqn.epsilon:.3f}")

    def _save_state(self):
        try:
            self.dqn.save(self._persist_path + '_dqn.npz')
            self.replay_buffer.save(self._persist_path + '_buffer.json')
            params_data = asdict(self.policy_params)
            with open(self._persist_path + '_params.json', 'w', encoding='utf-8') as f:
                json.dump(params_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[RLMemory] Save failed: {e}")

    def _load_state(self):
        try:
            self.dqn.load(self._persist_path + '_dqn.npz')
            self.replay_buffer.load(self._persist_path + '_buffer.json')
            params_path = self._persist_path + '_params.json'
            if os.path.exists(params_path):
                with open(params_path, 'r', encoding='utf-8') as f:
                    params_data = json.load(f)
                self.policy_params = PolicyParams(**params_data)
            logger.info(f"[RLMemory] Loaded, version={self.policy_params.version}, "
                         f"buffer={len(self.replay_buffer)}")
        except Exception as e:
            logger.warning(f"[RLMemory] Load failed (using defaults): {e}")

    def get_stats(self) -> Dict:
        return {
            'buffer_size': len(self.replay_buffer),
            'training_count': self._training_count,
            'epsilon': round(self.dqn.epsilon, 3),
            'params_version': self.policy_params.version,
            'policy_params': asdict(self.policy_params),
            'active_sessions': len(self._session_states)
        }
