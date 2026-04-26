---
name: "drawing-error-correction"
description: "Intelligent engineering drawing error correction and tutoring platform. Invoke when user uploads engineering drawings for analysis, asks about GB/T standard compliance, or needs drawing error detection and correction guidance."
---

# Engineering Drawing Intelligent Error Correction Skill

## Overview

This skill provides intelligent error correction and tutoring for mechanical engineering drawings based on GB/T national standards. It employs a multi-agent collaborative architecture with 5 specialized agents orchestrated through a 4-phase pipeline, combined with a dual knowledge base, Socratic tutoring methodology, and RL-based self-evolution.

**The complete deployable skill package is located at `drawing-error-correction-skill/` in the project root.**

## When to Invoke

- User uploads an engineering drawing image for analysis
- User asks about GB/T standard compliance of a drawing
- User needs dimension annotation, line type, tolerance, or title block error detection
- User requests Socratic-style tutoring feedback on drawing errors
- User wants to search GB standards knowledge base
- User submits RL feedback for system improvement

## Skill Package Structure

The self-contained deployable package at `drawing-error-correction-skill/` contains:

```
drawing-error-correction-skill/
├── app.py                    # Flask web application entry point
├── config.example.py         # Configuration template (copy to config.py)
├── config_loader.py          # Configuration loader (config.py > env vars > defaults)
├── setup.py                  # One-click setup script
├── start.bat                 # Windows quick start
├── start.sh                  # Linux/Mac quick start
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules (protects config.py)
├── src/                      # Core source code
│   ├── multi_agent_system.py # Multi-agent orchestrator + 5 agents
│   ├── geometric_detector.py # OpenCV geometric element detector
│   ├── rag_knowledge_base.py # Dual knowledge base (GB standards + background)
│   ├── rl_memory_unit.py     # RL memory unit with MiniDQN
│   ├── multimodal_agent.py   # Multimodal analysis agent
│   ├── error_injection.py    # Error injection for test data
│   ├── process_gb_pdf.py     # GB standard PDF processor
│   ├── process_standard_drawings.py # Standard drawing processor
│   └── collect_drawings.py   # Test drawing generator
├── templates/                # HTML templates
│   ├── index.html            # Upload page (CAD-style UI)
│   └── result.html           # Analysis result page
├── data/                     # Data directory (see DATA_README.md)
│   ├── drawings/             # Drawing images
│   ├── standard_drawings/    # Standard reference drawings
│   ├── error_drawings/       # Error-annotated drawings
│   ├── error_labels/         # Error annotation text files
│   ├── gb_standards/         # GB standard PDF/JSON files
│   ├── knowledge_base/       # Background knowledge JSON files
│   ├── rl_experience/        # RL experience data (auto-generated)
│   └── DATA_README.md        # Data directory guide
└── uploads/                  # User uploaded files (auto-created)
```

## Architecture

### Multi-Agent System (5 Agents + 1 Orchestrator)

```
DrawingOrchestrator
├── Phase 1 (Parallel): OCRAgent / GeometryAgent / StructureAgent
├── Phase 2 (Conditional): OCR Enhancement (RL-adaptive threshold)
├── Phase 3: RuleCheckAgent (GB standard rule validation)
└── Phase 4: LLMAgent (Deep analysis with Qwen2.5-72B-Instruct)
```

**OCRAgent**: Text recognition using RapidOCR, supports full-image and region-enhanced OCR for title blocks.

**GeometryAgent**: Geometric element detection using OpenCV (Hough transform, contour analysis), detects lines, circles, arrows, dimension structures, and classifies line types (solid/dashed/center-line).

**StructureAgent**: Drawing structure analysis, detects 6 functional regions, title block, view areas, and border.

**RuleCheckAgent**: 6 categories of GB standard rule validation: dimension annotation, line type, tolerance, title block, symbols, geometric completeness.

**LLMAgent**: Deep analysis using Qwen2.5-72B-Instruct, receives structured context from all agents, generates GB references and Socratic learning guidance. Falls back to local rule engine when API unavailable.

### Dual Knowledge Base

- **GB Standards KB**: Extracted from GB/T 14665-2012, serves as the sole display source for frontend
- **Background Knowledge KB**: 37 professional knowledge items injected into LLM system prompt (internalized, not displayed)
- **Image Knowledge KB**: Standard drawing references with HOG features for similarity search

### RL Memory Unit

- **MiniDQN**: 2-layer neural network (10→64→15), experience replay, target network
- **State Space**: 10-dimensional continuous vector (OCR count, confidence, geometry stats, error counts, quality score)
- **Action Space**: 15 discrete actions (adjust 7 policy parameters ± or no change)
- **Reward Function**: confirmed +1.0, useful_guidance +0.5, partial_confirm +0.3, ignored -0.5, dismissed_all -1.0

## Quick Deploy (One Command)

```bash
# Windows
start.bat

# Linux/Mac
chmod +x start.sh && ./start.sh
```

Or manually:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API access
cp config.example.py config.py
# Edit config.py with your API credentials

# 3. Run the application
python app.py

# 4. Open browser to http://localhost:5000
```

## Configuration Guide

### Required Configuration

Copy `config.example.py` to `config.py` and fill in your API credentials:

```python
# REQUIRED: Your LLM API endpoint URL
MULTIMODAL_API_URL = 'https://your-api-endpoint.example.com'

# REQUIRED: Your API key (NEVER commit this file)
MULTIMODAL_API_KEY = 'your-api-key-here'

# Optional: Model name (default: Qwen2.5-72B-Instruct)
LLM_MODEL = 'Qwen2.5-72B-Instruct'

# Optional: Vision model name for multimodal analysis
MULTIMODAL_VISION_MODEL = 'your-vision-model-name'
```

### Environment Variables (Alternative)

```bash
export MULTIMODAL_API_URL='https://your-api-endpoint.example.com'
export MULTIMODAL_API_KEY='your-api-key-here'
export LLM_MODEL='Qwen2.5-72B-Instruct'
export MULTIMODAL_VISION_MODEL='your-vision-model-name'
```

### Adding Example Drawings

Place your standard engineering drawings in `data/standard_drawings/`. See `data/DATA_README.md` for detailed instructions.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload page |
| `/upload` | POST | Upload and analyze drawing |
| `/uploads/<filename>` | GET | Serve uploaded file |
| `/api/gb_standards?q=<query>` | GET | Search GB standards |
| `/api/rl_feedback` | POST | Submit RL feedback |
| `/api/rl_stats` | GET | Get RL memory stats |

## Error Categories

| Category | GB Reference | Typical Issues |
|----------|-------------|----------------|
| Dimension Annotation | GB/T 4458.4 | Missing dimensions, missing Φ symbol, cramped spacing |
| Line Type | GB/T 4457.4 | Missing center lines, incorrect solid line ratio |
| Tolerance | GB/T 1800.1 | Missing tolerance annotations |
| Title Block | GB/T 10609.1 | Incomplete title block information |
| Symbols | GB/T 131 | Missing surface roughness (Ra) annotations |
| Geometric Completeness | — | Insufficient geometric elements, missing arrows |

## Security Notes

- **NEVER** commit `config.py` to any public repository — it contains API keys
- **NEVER** hardcode API keys in source code — always use `config.py` or environment variables
- The `config.py` file is listed in `.gitignore` to prevent accidental commits
- If API keys are accidentally exposed, rotate them immediately
- The system falls back to local rule engine when API is unavailable, ensuring core functionality without external dependencies
- RL experience data is stored locally and does not transmit any personal information
- Do not commit personal/sensitive drawing files to public repositories

## Dependencies

```
flask>=2.3.0
opencv-python>=4.8.0
numpy>=1.24.0
rapidocr-onnxruntime>=1.3.0
openai>=1.0.0
faiss-cpu>=1.7.4
scikit-learn>=1.3.0
pdfplumber>=0.9.0
Pillow>=10.0.0
PyMuPDF>=1.23.0
```
