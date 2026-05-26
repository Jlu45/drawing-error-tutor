import os
import sys

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
if SKILL_DIR not in sys.path:
    sys.path.insert(0, SKILL_DIR)

SRC_DIR = os.path.join(SKILL_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from config import (
        MULTIMODAL_API_URL, MULTIMODAL_API_KEY, LLM_MODEL,
        MULTIMODAL_VISION_MODEL, UPLOAD_FOLDER, ALLOWED_EXTENSIONS,
        TEXT_KNOWLEDGE_DIR, IMAGE_KNOWLEDGE_DIR, GB_STANDARDS_DIR,
        RL_EXPERIENCE_DIR, RL_STATE_DIM, RL_BUFFER_CAPACITY,
        RL_LEARNING_RATE, RL_GAMMA, RL_EPSILON_START, RL_EPSILON_MIN,
        RL_EPSILON_DECAY, FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
        VLM_MODEL, VLM_JUDGE_ENABLED, EXPERIENCE_STORE_DIR,
        ENABLE_ATLAS_PACK, ATLAS_RULE_MODE, ATLAS_CASES_PATH,
        ATLAS_RULES_PATH, ATLAS_EVAL_PATH, ATLAS_MAX_CONTEXT_CASES,
        ATLAS_SHOW_REFERENCE_IN_UI, ATLAS_ENABLE_VLM_FEWSHOT,
        ROLLBACK_MAX_RETRIES, LOG_LEVEL
    )
except ImportError:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "config_example",
        os.path.join(SKILL_DIR, "config.example.py")
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _names = [n for n in dir(_mod) if not n.startswith('_')]
    globals().update({n: getattr(_mod, n) for n in _names})

    MULTIMODAL_API_URL = os.environ.get('MULTIMODAL_API_URL', MULTIMODAL_API_URL)
    MULTIMODAL_API_KEY = os.environ.get('MULTIMODAL_API_KEY', MULTIMODAL_API_KEY)
    LLM_MODEL = os.environ.get('LLM_MODEL', LLM_MODEL)
    MULTIMODAL_VISION_MODEL = os.environ.get('MULTIMODAL_VISION_MODEL', MULTIMODAL_VISION_MODEL)
    VLM_MODEL = os.environ.get('VLM_MODEL', VLM_MODEL)
    VLM_JUDGE_ENABLED = os.environ.get('VLM_JUDGE_ENABLED', str(VLM_JUDGE_ENABLED)).lower() == 'true'
    ENABLE_ATLAS_PACK = os.environ.get('ENABLE_ATLAS_PACK', str(ENABLE_ATLAS_PACK)).lower() == 'true'
    ATLAS_RULE_MODE = os.environ.get('ATLAS_RULE_MODE', ATLAS_RULE_MODE)
    LOG_LEVEL = os.environ.get('LOG_LEVEL', LOG_LEVEL)
