# 🎯 Engineering Drawing Intelligent Error Correction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

An intelligent error correction and tutoring platform for mechanical engineering drawings based on GB/T national standards. Powered by a multi-agent collaborative architecture with 5 specialized agents, dual knowledge base, Socratic tutoring methodology, and RL-based self-evolution.

## ✨ Features

- **Multi-Agent Collaborative Analysis**: 5 specialized agents (OCR, Geometry, Structure, RuleCheck, LLM) orchestrated through a 4-phase pipeline
- **GB/T Standard Compliance**: Validates drawings against GB/T 4457-4460 national standards across 6 error categories
- **Socratic Tutoring**: Generates启发式 guidance to help students understand error causes
- **RL Self-Evolution**: MiniDQN-based reinforcement learning adapts analysis policies based on user feedback
- **Dual Knowledge Base**: GB standards KB + background knowledge KB + image KB with FAISS vector search
- **Graceful Degradation**: Falls back to local rule engine when LLM API is unavailable
- **CAD-Style UI**: Professional engineering drawing interface with light/dark/eye-care themes

## 🏗️ Architecture

```
DrawingOrchestrator
├── Phase 1 (Parallel): OCRAgent / GeometryAgent / StructureAgent
├── Phase 2 (Conditional): OCR Enhancement (RL-adaptive threshold)
├── Phase 3: RuleCheckAgent (GB standard rule validation)
└── Phase 4: LLMAgent (Deep analysis with Qwen2.5-72B-Instruct)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/drawing-error-correction.git
cd drawing-error-correction

# Run the setup script (installs dependencies, creates config, verifies environment)
python setup.py
```

### Configuration

Copy `config.example.py` to `config.py` and fill in your API credentials:

```bash
cp config.example.py config.py
```

Edit `config.py`:

```python
# REQUIRED: Your LLM API endpoint URL
MULTIMODAL_API_URL = 'https://your-api-endpoint.example.com'

# REQUIRED: Your API key (NEVER commit this file)
MULTIMODAL_API_KEY = 'your-api-key-here'

# Optional: Model configuration
LLM_MODEL = 'Qwen2.5-72B-Instruct'
MULTIMODAL_VISION_MODEL = 'your-vision-model-name'
```

Or use environment variables:

```bash
export MULTIMODAL_API_URL='https://your-api-endpoint.example.com'
export MULTIMODAL_API_KEY='your-api-key-here'
```

### Running

```bash
# Start the application
python app.py

# Or use the quick-start scripts
./start.sh      # Linux/Mac
start.bat       # Windows
```

Open http://localhost:5000 in your browser.

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t drawing-error-correction .
docker run -p 5000:5000 \
  -e MULTIMODAL_API_URL='https://your-api-endpoint.example.com' \
  -e MULTIMODAL_API_KEY='your-api-key' \
  drawing-error-correction
```

## 📖 API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload page |
| `/upload` | POST | Upload and analyze drawing |
| `/uploads/<filename>` | GET | Serve uploaded file |
| `/api/gb_standards?q=<query>` | GET | Search GB standards |
| `/api/rl_feedback` | POST | Submit RL feedback |
| `/api/rl_stats` | GET | Get RL memory stats |

### RL Feedback API

```bash
curl -X POST http://localhost:5000/api/rl_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "drawing.png_1700000000",
    "error_description": "Missing diameter symbol",
    "feedback_type": "confirmed"
  }'
```

Valid `feedback_type` values: `confirmed`, `ignored`, `dismissed_all`, `partial_confirm`, `useful_guidance`

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/test_geometric_detector.py -v
```

## 📁 Project Structure

```
drawing-error-correction/
├── app.py                    # Flask web application entry point
├── config.example.py         # Configuration template
├── config_loader.py          # Configuration loader
├── setup.py                  # One-click setup script
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata and tool config
├── src/                      # Core source code
│   ├── multi_agent_system.py # Multi-agent orchestrator + 5 agents
│   ├── geometric_detector.py # OpenCV geometric element detector
│   ├── rag_knowledge_base.py # Dual knowledge base
│   ├── rl_memory_unit.py     # RL memory unit with MiniDQN
│   ├── multimodal_agent.py   # Multimodal analysis agent
│   ├── error_injection.py    # Error injection for test data
│   ├── process_gb_pdf.py     # GB standard PDF processor
│   ├── process_standard_drawings.py
│   └── collect_drawings.py   # Test drawing generator
├── templates/                # HTML templates
│   ├── index.html            # Upload page (CAD-style UI)
│   └── result.html           # Analysis result page
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Usage examples
├── data/                     # Data directory (see data/DATA_README.md)
└── uploads/                  # User uploaded files
```

## 🔒 Security

- **NEVER** commit `config.py` to any public repository — it contains API keys
- **NEVER** hardcode API keys in source code — always use `config.py` or environment variables
- The `config.py` file is listed in `.gitignore` to prevent accidental commits
- If API keys are accidentally exposed, rotate them immediately
- The system falls back to local rule engine when API is unavailable
- RL experience data is stored locally and does not transmit personal information

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to contribute to this project.

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- GB/T 4457-4460 Mechanical Drawing National Standards
- [RapidOCR](https://github.com/RapidAI/RapidOCR) for text recognition
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [OpenAI-compatible API](https://github.com/openai/openai-python) for LLM integration

## 📧 Contact

For questions and support, please open a [GitHub Issue](https://github.com/your-username/drawing-error-correction/issues).
