"""
RL记忆单元 (复用原版设计)
==========================
MiniDQN + 经验回放池 + 策略参数自适应。
从原版 rl_memory_unit.py 复用核心逻辑。
"""

import os
import json
import time
import random
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("RLMemory")

EXPERIENCE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'data', 'rl_experience')
os.makedirs(EXPERIENCE_DIR, exist_ok=True)


@dataclass
class Experience:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""


@dataclass
class PolicyParameters:
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
    def from_vector(cls, v: List[float], version: int = 0) -> 'PolicyParameters':
        return cls(
            severity_weight_high=v[0], severity_weight_medium=v[1], severity_weight_low=v[2],
            score_penalty_per_weight=v[3], llm_score_fusion_ratio=v[4],
            ocr_enhance_threshold=max(1, int(v[5])),
            rule_confidence_threshold=v[6], version=version
        )

    def clamp(self) -> 'PolicyParameters':
        self.severity_weight_high = float(np.clip(self.severity_weight_high, 1.0, 6.0))
        self.severity_weight_medium = float(np.clip(self.severity_weight_medium, 0.5, 4.0))
        self.severity_weight_low = float(np.clip(self.severity_weight_low, 0.1, 2.0))
        self.score_penalty_per_weight = float(np.clip(self.score_penalty_per_weight, 1.0, 10.0))
        self.llm_score_fusion_ratio = float(np.clip(self.llm_score_fusion_ratio, 0.1, 0.9))
        self.ocr_enhance_threshold = max(1, min(15, int(self.ocr_enhance_threshold)))
        self.rule_confidence_threshold = float(np.clip(self.rule_confidence_threshold, 0.05, 0.8))
        return self


class MiniDQN:
    """轻量级DQN（2层全连接网络，纯numpy实现）"""

    def __init__(self, state_dim: int = 10, action_dim: int = 15, hidden_dim: int = 64,
                 lr: float = 0.01, gamma: float = 0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma

        # 初始化权重（Xavier初始化）
        scale1 = np.sqrt(2.0 / (state_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + action_dim))
        self.w1 = np.random.randn(state_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, action_dim) * scale2
        self.b2 = np.zeros(action_dim)

        # 目标网络
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

        self.train_step = 0
        self.target_update_freq = 10

    def forward(self, x: np.ndarray, use_target: bool = False) -> np.ndarray:
        if use_target:
            h = np.maximum(0, x @ self.target_w1 + self.target_b1)
            return h @ self.target_w2 + self.target_b2
        h = np.maximum(0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def predict(self, state: List[float]) -> np.ndarray:
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        return self.forward(x).flatten()

    def predict_greedy(self, state: List[float]) -> int:
        q = self.predict(state)
        return int(np.argmax(q))

    def train(self, batch: List[Experience]):
        if not batch:
            return
        states = np.array([e.state for e in batch], dtype=np.float32)
        actions = np.array([e.action for e in batch], dtype=np.int32)
        rewards = np.array([e.reward for e in batch], dtype=np.float32)
        next_states = np.array([e.next_state for e in batch], dtype=np.float32)
        dones = np.array([e.done for e in batch], dtype=np.float32)

        # 当前Q值
        q_values = self.forward(states)
        q_selected = q_values[np.arange(len(batch)), actions]

        # 目标Q值
        next_q = self.forward(next_states, use_target=True)
        target_q = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)

        # MSE损失梯度
        td_error = q_selected - target_q
        loss = np.mean(td_error ** 2)

        # 反向传播（简化版）
        dq = np.zeros_like(q_values)
        dq[np.arange(len(batch)), actions] = 2 * td_error / len(batch)

        dh = np.maximum(0, states @ self.w1 + self.b1)
        dh_grad = (dq @ self.w2.T) * (dh > 0)

        self.w2 -= self.lr * (dh.T @ dq)
        self.b2 -= self.lr * dq.sum(axis=0)
        self.w1 -= self.lr * (states.T @ dh_grad)
        self.b1 -= self.lr * dh_grad.sum(axis=0)

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_w1 = self.w1.copy()
            self.target_b1 = self.b1.copy()
            self.target_w2 = self.w2.copy()
            self.target_b2 = self.b2.copy()

        return loss

    def save(self, path: str):
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                 tw1=self.target_w1, tb1=self.target_b1, tw2=self.target_w2, tb2=self.target_b2)

    def load(self, path: str):
        data = np.load(path)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']
        self.target_w1 = data['tw1']
        self.target_b1 = data['tb1']
        self.target_w2 = data['tw2']
        self.target_b2 = data['tb2']


class RLMemoryUnit:
    """RL记忆单元：DQN + 经验回放 + 策略参数"""

    def __init__(self, state_dim: int = 10, action_dim: int = 15,
                 buffer_capacity: int = 500, lr: float = 0.01,
                 gamma: float = 0.95, epsilon_start: float = 0.3,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.995):
        self.dqn = MiniDQN(state_dim, action_dim, lr=lr, gamma=gamma)
        self.buffer = deque(maxlen=buffer_capacity)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = 16

        self.policy_params = PolicyParameters()
        self._sessions: Dict[str, Dict] = {}

        self._load()

    def extract_state(self, result: Dict) -> List[float]:
        """从分析结果提取10维状态向量"""
        errors = result.get('errors', [])
        report = result.get('report', {})
        ocr = result.get('ocr_results', [])
        geo = result.get('geo_result', {})

        state = [
            len(ocr) / 50.0,                                          # OCR数量（归一化）
            sum(1 for t in ocr if t.get('confidence', 0) > 0.7) / max(len(ocr), 1),  # OCR置信度
            len(geo.get('lines', [])) / 50.0,                         # 直线数
            len(geo.get('circles', [])) / 10.0,                       # 圆数
            len(geo.get('arrows', [])) / 20.0,                        # 箭头数
            1.0 if report.get('total_errors', 0) > 0 else 0.0,        # 是否有错误
            sum(1 for e in errors if e.get('severity') == '高') / 10.0,  # 高严重度错误
            report.get('overall_score', 50) / 100.0,                  # 质量评分
            self.policy_params.llm_score_fusion_ratio,                 # LLM融合比例
            self.policy_params.rule_confidence_threshold               # 规则置信度阈值
        ]
        return [float(np.clip(s, 0, 1)) for s in state]

    def select_action(self, state: List[float]) -> int:
        """ε-贪心策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.dqn.action_dim - 1)
        return self.dqn.predict_greedy(state)

    def apply_action(self, action: int):
        """将动作映射到策略参数调整"""
        params = self.policy_params.to_vector()
        n_params = len(params)
        param_idx = action // 3  # 哪个参数
        direction = (action % 3) - 1  # -1/0/+1

        if param_idx < n_params and direction != 0:
            delta = direction * params[param_idx] * 0.1
            params[param_idx] += delta
            self.policy_params = PolicyParameters.from_vector(params)
            self.policy_params.clamp()
            self.policy_params.version += 1

    def register_session(self, session_id: str, state: List[float],
                          action: int, result: Dict):
        """注册分析会话"""
        self._sessions[session_id] = {
            'state': state, 'action': action, 'result': result,
            'timestamp': time.time(), 'reward': None
        }

    def give_feedback(self, session_id: str, reward: float):
        """给予反馈奖励"""
        if session_id not in self._sessions:
            return
        session = self._sessions[session_id]
        session['reward'] = reward

        # 计算next_state（使用当前状态作为近似）
        next_state = session['state']

        # 添加到经验池
        exp = Experience(
            state=session['state'],
            action=session['action'],
            reward=reward,
            next_state=next_state,
            done=True,
            session_id=session_id
        )
        self.buffer.append(exp)

        # 训练
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(list(self.buffer), self.batch_size)
            loss = self.dqn.train(batch)
            logger.debug(f"[RL] 训练 loss={loss:.4f}")

        # 衰减ε
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._save()

    def get_policy_params(self) -> PolicyParameters:
        return self.policy_params

    def get_stats(self) -> Dict:
        return {
            'buffer_size': len(self.buffer),
            'epsilon': round(self.epsilon, 4),
            'policy_version': self.policy_params.version,
            'total_sessions': len(self._sessions),
            'feedback_received': sum(1 for s in self._sessions.values() if s['reward'] is not None),
            'train_steps': self.dqn.train_step
        }

    def _save(self):
        try:
            self.dqn.save(os.path.join(EXPERIENCE_DIR, 'dqn_weights.npz'))
            with open(os.path.join(EXPERIENCE_DIR, 'policy_params.json'), 'w') as f:
                json.dump(self.policy_params.__dict__, f, indent=2)
            with open(os.path.join(EXPERIENCE_DIR, 'buffer.json'), 'w') as f:
                json.dump([{'state': e.state, 'action': e.action, 'reward': e.reward,
                            'next_state': e.next_state, 'done': e.done,
                            'session_id': e.session_id}
                           for e in list(self.buffer)[-100:]], f)  # 只保存最近100条
        except Exception as e:
            logger.error(f"[RL] 保存失败: {e}")

    def _load(self):
        try:
            dqn_path = os.path.join(EXPERIENCE_DIR, 'dqn_weights.npz')
            if os.path.exists(dqn_path):
                self.dqn.load(dqn_path)
            params_path = os.path.join(EXPERIENCE_DIR, 'policy_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    data = json.load(f)
                self.policy_params = PolicyParameters(**{k: v for k, v in data.items()
                                                          if k in PolicyParameters.__dataclass_fields__})
        except Exception as e:
            logger.warning(f"[RL] 加载失败: {e}")
