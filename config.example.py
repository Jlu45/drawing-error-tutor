import os

MULTIMODAL_API_URL = os.environ.get('MULTIMODAL_API_URL', '')
MULTIMODAL_API_KEY = os.environ.get('MULTIMODAL_API_KEY', '')
LLM_MODEL = os.environ.get('LLM_MODEL', 'Qwen2.5-72B-Instruct')
MULTIMODAL_VISION_MODEL = os.environ.get('MULTIMODAL_VISION_MODEL', 'kimi')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

TEXT_KNOWLEDGE_DIR = 'data/knowledge_base'
IMAGE_KNOWLEDGE_DIR = 'data/standard_drawings'
GB_STANDARDS_DIR = 'data/gb_standards'
RL_EXPERIENCE_DIR = 'data/rl_experience'

RL_STATE_DIM = 10
RL_BUFFER_CAPACITY = 500
RL_LEARNING_RATE = 0.01
RL_GAMMA = 0.95
RL_EPSILON_START = 0.3
RL_EPSILON_MIN = 0.05
RL_EPSILON_DECAY = 0.995

FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

VLM_MODEL = os.environ.get('VLM_MODEL', 'qwen-vl-max')
VLM_JUDGE_ENABLED = os.environ.get('VLM_JUDGE_ENABLED', 'true').lower() == 'true'
EXPERIENCE_STORE_DIR = 'data/experience_store'
ENABLE_ATLAS_PACK = os.environ.get('ENABLE_ATLAS_PACK', 'true').lower() == 'true'
ATLAS_RULE_MODE = os.environ.get('ATLAS_RULE_MODE', 'safe')
ATLAS_CASES_PATH = 'data/atlas/atlas_cases.jsonl'
ATLAS_RULES_PATH = 'data/atlas/atlas_rules.yaml'
ATLAS_EVAL_PATH = 'data/atlas/atlas_eval_cases.jsonl'
ATLAS_MAX_CONTEXT_CASES = 3
ATLAS_SHOW_REFERENCE_IN_UI = True
ATLAS_ENABLE_VLM_FEWSHOT = True
ROLLBACK_MAX_RETRIES = {'ocr': 2, 'geometry': 2, 'structure': 1, 'rule_check': 1, 'llm': 1}
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
