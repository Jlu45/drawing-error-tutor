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
