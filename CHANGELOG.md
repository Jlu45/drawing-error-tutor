# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-24

### Added

- Multi-agent collaborative analysis system with 5 specialized agents
  - OCRAgent: Text recognition using RapidOCR
  - GeometryAgent: Geometric element detection using OpenCV
  - StructureAgent: Drawing structure analysis
  - RuleCheckAgent: GB standard rule validation (6 categories)
  - LLMAgent: Deep analysis with Qwen2.5-72B-Instruct
- DrawingOrchestrator with 4-phase analysis pipeline
- Dual Knowledge Base (GB standards + background knowledge + image KB)
- RL Memory Unit with MiniDQN for self-evolution
  - 10-dimensional state space
  - 15 discrete actions (7 policy parameters ± or no change)
  - Experience replay buffer with target network
- Socratic tutoring methodology for error guidance
- CAD-style web interface with light/dark/eye-care themes
- REST API endpoints for analysis, GB search, and RL feedback
- Graceful degradation to local rule engine when API unavailable
- Modular configuration system (config.py / environment variables)
- Docker support with Dockerfile and docker-compose.yml
- Comprehensive test suite (geometric detector, rule check, RL memory, knowledge base, security)
- Documentation (API docs, development guide, deployment guide)
- Usage examples (basic analysis, knowledge management, RL feedback)
- GitHub Actions CI/CD workflow
- Issue and PR templates

### Security

- Removed all hardcoded API keys and internal URLs from source code
- Configuration separation with config.example.py template
- .gitignore protection for config.py and .env files
- Privacy verification in test suite
