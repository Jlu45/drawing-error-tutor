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
        RL_EPSILON_DECAY, FLASK_HOST, FLASK_PORT, FLASK_DEBUG
    )
except ImportError:
    from config_example import *

    MULTIMODAL_API_URL = os.environ.get('MULTIMODAL_API_URL', MULTIMODAL_API_URL)
    MULTIMODAL_API_KEY = os.environ.get('MULTIMODAL_API_KEY', MULTIMODAL_API_KEY)
    LLM_MODEL = os.environ.get('LLM_MODEL', LLM_MODEL)
    MULTIMODAL_VISION_MODEL = os.environ.get('MULTIMODAL_VISION_MODEL', MULTIMODAL_VISION_MODEL)
