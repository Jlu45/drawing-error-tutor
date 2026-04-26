import pytest
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_memory_unit import (
    MiniDQN, ExperienceReplayBuffer, RLMemoryUnit,
    PolicyParameters, Experience, ACTION_NAMES, NUM_ACTIONS
)


class TestPolicyParameters:
    def test_default_values(self):
        params = PolicyParameters()
        assert params.severity_weight_high == 3.0
        assert params.severity_weight_medium == 2.0
        assert params.severity_weight_low == 1.0
        assert params.llm_score_fusion_ratio == 0.5
        assert params.ocr_enhance_threshold == 5

    def test_to_vector_and_back(self):
        params = PolicyParameters()
        vec = params.to_vector()
        assert len(vec) == 7
        restored = PolicyParameters.from_vector(vec, version=1)
        assert abs(restored.severity_weight_high - params.severity_weight_high) < 1e-6
        assert restored.version == 1

    def test_clamp(self):
        params = PolicyParameters(
            severity_weight_high=10.0,
            severity_weight_low=-1.0,
            llm_score_fusion_ratio=1.5,
            ocr_enhance_threshold=0
        )
        params.clamp()
        assert params.severity_weight_high <= 6.0
        assert params.severity_weight_low >= 0.1
        assert params.llm_score_fusion_ratio <= 0.9
        assert params.ocr_enhance_threshold >= 1


class TestMiniDQN:
    def test_initialization(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS)
        assert dqn.state_dim == 10
        assert dqn.action_dim == NUM_ACTIONS
        assert dqn.W1.shape == (10, 64)
        assert dqn.W2.shape == (64, NUM_ACTIONS)

    def test_predict_returns_valid_action(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS, epsilon=0.0)
        state = np.random.randn(10).astype(np.float32)
        action = dqn.predict(state)
        assert 0 <= action < NUM_ACTIONS

    def test_predict_greedy(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS)
        state = np.random.randn(10).astype(np.float32)
        action = dqn.predict_greedy(state)
        assert 0 <= action < NUM_ACTIONS

    def test_get_q_values(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS)
        state = np.random.randn(10).astype(np.float32)
        q_values = dqn.get_q_values(state)
        assert q_values.shape == (NUM_ACTIONS,)

    def test_train_step(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS, epsilon=0.0)
        experiences = []
        for _ in range(8):
            experiences.append(Experience(
                state=np.random.randn(10).tolist(),
                action=np.random.randint(0, NUM_ACTIONS),
                reward=np.random.randn(),
                next_state=np.random.randn(10).tolist(),
                done=False
            ))
        initial_w1 = dqn.W1.copy()
        dqn.train_step(experiences)
        assert not np.array_equal(dqn.W1, initial_w1)

    def test_save_and_load(self):
        dqn = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS)
        state = np.random.randn(10).astype(np.float32)
        q_before = dqn.get_q_values(state)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            path = f.name
        dqn.save(path)
        dqn2 = MiniDQN(state_dim=10, action_dim=NUM_ACTIONS)
        dqn2.load(path)
        q_after = dqn2.get_q_values(state)
        assert np.allclose(q_before, q_after, atol=1e-5)
        os.unlink(path)


class TestExperienceReplayBuffer:
    def test_push_and_sample(self):
        buffer = ExperienceReplayBuffer(capacity=100)
        for i in range(10):
            buffer.push(Experience(
                state=[0.0] * 10, action=0, reward=1.0,
                next_state=[0.0] * 10, done=False
            ))
        assert len(buffer) == 10
        sample = buffer.sample(5)
        assert len(sample) == 5

    def test_capacity(self):
        buffer = ExperienceReplayBuffer(capacity=5)
        for i in range(10):
            buffer.push(Experience(
                state=[float(i)] * 10, action=0, reward=float(i),
                next_state=[0.0] * 10, done=False
            ))
        assert len(buffer) == 5

    def test_save_and_load(self):
        buffer = ExperienceReplayBuffer(capacity=100)
        for i in range(5):
            buffer.push(Experience(
                state=[float(i)] * 10, action=i % NUM_ACTIONS, reward=float(i),
                next_state=[0.0] * 10, done=False
            ))
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        buffer.save(path)
        buffer2 = ExperienceReplayBuffer(capacity=100)
        buffer2.load(path)
        assert len(buffer2) == 5
        os.unlink(path)


class TestRLMemoryUnit:
    def test_initialization(self):
        unit = RLMemoryUnit(state_dim=10)
        assert unit.policy_params is not None
        assert unit.dqn is not None
        assert unit.replay_buffer is not None

    def test_extract_state(self):
        unit = RLMemoryUnit(state_dim=10)
        analysis_result = {
            'ocr_results': [{'text': 'test', 'confidence': 0.9}] * 10,
            'geo_result': {
                'lines': list(range(30)),
                'circles': list(range(5)),
                'arrows': list(range(3)),
            },
            'structure_result': {'title_block': {'detected': True}},
            'report': {'total_errors': 3, 'error_categories': {}},
            'metrics': {'quality_score': 0.7}
        }
        state = unit.extract_state(analysis_result)
        assert state.shape == (10,)
        assert np.all(state >= 0) and np.all(state <= 1)

    def test_select_action(self):
        unit = RLMemoryUnit(state_dim=10)
        state = np.random.randn(10).astype(np.float32)
        action = unit.select_action(state)
        assert 0 <= action < NUM_ACTIONS

    def test_get_stats(self):
        unit = RLMemoryUnit(state_dim=10)
        stats = unit.get_stats()
        assert 'buffer_size' in stats
        assert 'epsilon' in stats
        assert 'policy_params' in stats
