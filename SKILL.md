---
name: "drawing-error-correction"
version: "2.0.0"
description: "Intelligent engineering drawing error correction and tutoring platform. Invoke when user uploads engineering drawings for analysis, asks about GB/T standard compliance, or needs drawing error detection and correction guidance. Supports V1 (4-phase) and V2 (6-phase) pipeline modes."
---

# Engineering Drawing Intelligent Error Correction Skill

## Overview

This skill provides intelligent error correction and tutoring for mechanical engineering drawings based on GB/T national standards. It employs a multi-agent collaborative architecture with 5 specialized agents orchestrated through a **6-phase pipeline (V2)** or **4-phase pipeline (V1)**, combined with a dual knowledge base, Socratic tutoring methodology, RL-based self-evolution, and four core V2 innovation modules: Pre-check Protocol, Cross-Stage Rollback, Asymmetric Experience Store, and VLM Judge.

**Version**: 2.0.0 (backward compatible with V1)

**The complete deployable skill package is located at `drawing-error-correction-skill/` in the project root.**

## When to Invoke

- User uploads an engineering drawing image for analysis
- User asks about GB/T standard compliance of a drawing
- User needs dimension annotation, line type, tolerance, or title block error detection
- User requests Socratic-style tutoring feedback on drawing errors
- User wants to search GB standards knowledge base
- User submits RL feedback for system improvement
- User queries Atlas drawing atlas cases or rules
- User requests VLM-based quality evaluation of analysis results

## Skill Package Structure

The self-contained deployable package at `drawing-error-correction-skill/` contains:

```
drawing-error-correction-skill/
├── app.py                    # Flask web application entry point (V1 + V2 routes)
├── config.example.py         # Configuration template (copy to config.py)
├── config_loader.py          # Configuration loader (config.py > env vars > defaults)
├── setup.py                  # One-click setup script
├── start.bat                 # Windows quick start
├── start.sh                  # Linux/Mac quick start
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules (protects config.py)
├── src/                      # Core source code
│   ├── multi_agent_system.py # Multi-agent orchestrator + 5 agents (V1 pipeline)
│   ├── connector_contract.py # [V2] Pre-check Protocol - detection contracts
│   ├── cross_stage_rollback.py # [V2] Cross-Stage Rollback - targeted error recovery
│   ├── experience_store.py   # [V2] Asymmetric Experience Store - dual-partition FAISS retrieval
│   ├── vlm_judge.py          # [V2] VLM Judge - 4-dimension quality evaluation
│   ├── pipeline_v2.py        # [V2] 6-phase pipeline orchestrator
│   ├── geometric_detector.py # OpenCV geometric element detector
│   ├── rag_knowledge_base.py # Dual knowledge base (GB standards + background)
│   ├── rl_memory_unit.py     # RL memory unit with MiniDQN (V2: 4-dim reward)
│   ├── multimodal_agent.py   # Multimodal analysis agent
│   ├── error_injection.py    # Error injection for test data
│   ├── process_gb_pdf.py     # GB standard PDF processor
│   ├── process_standard_drawings.py # Standard drawing processor
│   └── collect_drawings.py   # Test drawing generator
├── atlas/                    # [V2] Atlas Drawing Atlas package
│   ├── __init__.py           # Package init
│   ├── cases.py              # 34 curated drawing error cases
│   ├── rules.py              # 24 GB/T standard rules
│   ├── categories.py         # 8 error category definitions
│   └── feedback.py           # Atlas feedback collector
├── rl/                       # [V2] Enhanced RL Memory package
│   ├── __init__.py           # Package init
│   ├── memory.py             # Enhanced RL memory with 4-dim reward
│   └── reward.py             # 4-dim continuous reward signal processing
├── templates/                # HTML templates
│   ├── index.html            # Upload page (CAD-style UI, V1)
│   ├── index_v2.html         # [V2] Upload page (enhanced UI with Atlas)
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

### V2 6-Phase Pipeline (Default)

```
PipelineV2
├── Phase 1 (Planning):       Pre-check Protocol → detection contract definition
├── Phase 2 (Perception):     OCRAgent / GeometryAgent / StructureAgent (parallel)
├── Phase 3 (RuleCheck):      RuleCheckAgent (GB standard rule validation)
├── Phase 4 (LLM):            LLMAgent (Deep analysis with Qwen2.5-72B-Instruct)
├── Phase 5 (Fusion):         Cross-Stage Rollback + result fusion
└── Phase 6 (Judge):          VLM Judge (4-dimension quality evaluation)
```

**Planning Phase**: Defines detection contracts via `connector_contract.py` before execution. Each contract specifies expected inputs, outputs, and validation criteria for downstream stages, enabling early error detection and targeted recovery.

**Perception Phase**: Parallel execution of OCR, Geometry, and Structure agents (same as V1 Phase 1).

**RuleCheck Phase**: GB standard rule validation (same as V1 Phase 3).

**LLM Phase**: Deep analysis with structured context from all agents (same as V1 Phase 4).

**Fusion Phase**: Cross-stage rollback mechanism via `cross_stage_rollback.py`. If a downstream stage detects issues, it can trigger targeted rollback to a specific upstream stage instead of full pipeline retry. Results from all stages are fused into a unified output.

**Judge Phase**: VLM Judge via `vlm_judge.py` evaluates the final output across 4 dimensions:
- **Accuracy** (准确性): Are the detected errors correct?
- **Completeness** (完整性): Are all significant errors identified?
- **Helpfulness** (有用性): Is the guidance practical and actionable?
- **Guidance** (引导性): Does it follow Socratic tutoring methodology?

### V1 4-Phase Pipeline (Legacy, Still Supported)

```
DrawingOrchestrator
├── Phase 1 (Parallel): OCRAgent / GeometryAgent / StructureAgent
├── Phase 2 (Conditional): OCR Enhancement (RL-adaptive threshold)
├── Phase 3: RuleCheckAgent (GB standard rule validation)
└── Phase 4: LLMAgent (Deep analysis with Qwen2.5-72B-Instruct)
```

### V2 Core Innovation Modules

#### 1. Pre-check Protocol (`connector_contract.py`)

Defines detection contracts before pipeline execution. Each contract specifies:
- Expected input schema and ranges for each stage
- Output validation criteria and type constraints
- Inter-stage data flow contracts (e.g., OCR results must have confidence > threshold)

If a contract is violated, the system can immediately identify which stage failed and why, enabling targeted recovery rather than blind retry.

#### 2. Cross-Stage Rollback (`cross_stage_rollback.py`)

Provides targeted error recovery instead of full pipeline retry:
- Maintains a stage dependency graph to identify which upstream stages affect a failed downstream result
- Supports partial rollback: only re-executes affected stages while preserving valid intermediate results
- Configurable rollback depth (shallow vs. deep recovery strategies)
- Tracks rollback history to prevent infinite retry loops

#### 3. Asymmetric Experience Store (`experience_store.py`)

Dual-partition case retrieval system powered by FAISS:
- **Positive Partition**: Stores successful correction cases (high-quality analysis results confirmed by user feedback)
- **Negative Partition**: Stores failed correction cases (dismissed or partially confirmed results)
- Asymmetric weighting: positive cases receive higher retrieval priority; negative cases serve as anti-patterns
- FAISS-based similarity search for fast case retrieval
- Automatic partition management with configurable capacity limits

#### 4. VLM Judge (`vlm_judge.py`)

4-dimension quality evaluation using a Vision-Language Model:
- Evaluates analysis results on Accuracy, Completeness, Helpfulness, and Guidance
- Produces continuous scores (0.0–1.0) per dimension
- Feeds 4-dim reward signals back to the RL Memory Unit for policy optimization
- Can be enabled/disabled via `VLM_JUDGE_ENABLED` config option
- Falls back to rule-based scoring when VLM is unavailable

### Multi-Agent System (5 Agents + 1 Orchestrator)

**OCRAgent**: Text recognition using RapidOCR, supports full-image and region-enhanced OCR for title blocks.

**GeometryAgent**: Geometric element detection using OpenCV (Hough transform, contour analysis), detects lines, circles, arrows, dimension structures, and classifies line types (solid/dashed/center-line).

**StructureAgent**: Drawing structure analysis, detects 6 functional regions, title block, view areas, and border.

**RuleCheckAgent**: 8 categories of GB standard rule validation: dimension annotation, line type, tolerance, title block, symbols, geometric completeness, welding symbols, and drawing sheet specification.

**LLMAgent**: Deep analysis using Qwen2.5-72B-Instruct, receives structured context from all agents, generates GB references and Socratic learning guidance. Falls back to local rule engine when API unavailable.

### Dual Knowledge Base

- **GB Standards KB**: Extracted from GB/T 14665-2012, serves as the sole display source for frontend
- **Background Knowledge KB**: 37 professional knowledge items injected into LLM system prompt (internalized, not displayed)
- **Image Knowledge KB**: Standard drawing references with HOG features for similarity search

### Atlas Drawing Atlas (`atlas/` package)

V2 introduces the Atlas Drawing Atlas — a curated knowledge base of drawing error cases:

- **34 Cases**: Real-world engineering drawing error cases with annotations and corrections
- **24 Rules**: GB/T standard rules organized by category for structured validation
- **8 Error Categories**: Comprehensive coverage of common drawing errors
- **Feedback Loop**: Users can submit feedback on atlas cases for continuous improvement
- **Stats API**: Provides usage statistics and case coverage metrics

### RL Memory Unit (`rl/` package)

V2 enhances the RL Memory Unit with 4-dimension continuous reward signals from VLM Judge:

- **MiniDQN**: 2-layer neural network (10→64→15), experience replay, target network
- **State Space**: 10-dimensional continuous vector (OCR count, confidence, geometry stats, error counts, quality score)
- **Action Space**: 15 discrete actions (adjust 7 policy parameters ± or no change)
- **V1 Reward**: Scalar reward (confirmed +1.0, useful_guidance +0.5, partial_confirm +0.3, ignored -0.5, dismissed_all -1.0)
- **V2 Reward**: 4-dim continuous reward vector from VLM Judge (accuracy, completeness, helpfulness, guidance), each in [0.0, 1.0], mapped to policy gradient updates

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

# 4. Open browser to http://localhost:5000 (V1) or http://localhost:5000/v2 (V2)
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

### V2 Configuration Options

```python
# V2 Pipeline Configuration
VLM_MODEL = 'Qwen2.5-VL-72B-Instruct'        # VLM model for Judge phase
VLM_JUDGE_ENABLED = True                       # Enable/disable VLM Judge (default: True)
ENABLE_ATLAS_PACK = True                       # Enable Atlas Drawing Atlas (default: True)
ATLAS_RULE_MODE = 'strict'                     # 'strict' or 'lenient' rule matching
EXPERIENCE_STORE_CAPACITY = 1000               # Max cases per partition in Experience Store
ROLLBACK_MAX_DEPTH = 3                         # Max rollback depth for Cross-Stage Rollback
```

### Environment Variables (Alternative)

```bash
export MULTIMODAL_API_URL='https://your-api-endpoint.example.com'
export MULTIMODAL_API_KEY='your-api-key-here'
export LLM_MODEL='Qwen2.5-72B-Instruct'
export MULTIMODAL_VISION_MODEL='your-vision-model-name'
export VLM_MODEL='Qwen2.5-VL-72B-Instruct'
export VLM_JUDGE_ENABLED='True'
export ENABLE_ATLAS_PACK='True'
export ATLAS_RULE_MODE='strict'
```

### Adding Example Drawings

Place your standard engineering drawings in `data/standard_drawings/`. See `data/DATA_README.md` for detailed instructions.

## API Endpoints

### V1 Endpoints (Legacy, Still Supported)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Upload page (V1 UI) |
| `/upload` | POST | Upload and analyze drawing (V1 4-phase pipeline) |
| `/uploads/<filename>` | GET | Serve uploaded file |
| `/api/gb_standards?q=<query>` | GET | Search GB standards |
| `/api/rl_feedback` | POST | Submit RL feedback (V1 scalar reward) |
| `/api/rl_stats` | GET | Get RL memory stats |

### V2 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2` | GET | Upload page (V2 enhanced UI with Atlas) |
| `/v2/upload` | POST | Upload and analyze drawing (V2 6-phase pipeline) |
| `/api/v2/rl_feedback` | POST | Submit RL feedback (V2 4-dim reward from VLM Judge) |
| `/api/v2/rl_stats` | GET | Get RL memory stats (V2 enhanced) |
| `/api/v2/experience_stats` | GET | Get Asymmetric Experience Store stats |
| `/api/atlas/case/<case_id>` | GET | Retrieve specific Atlas case |
| `/api/atlas/rules` | GET | List all Atlas rules |
| `/api/atlas/feedback` | POST | Submit feedback on Atlas case |
| `/api/atlas/stats` | GET | Get Atlas usage statistics |

## Error Categories

| Category | GB Reference | Typical Issues |
|----------|-------------|----------------|
| Dimension Annotation | GB/T 4458.4 | Missing dimensions, missing Φ symbol, cramped spacing |
| Line Type | GB/T 4457.4 | Missing center lines, incorrect solid line ratio |
| Tolerance | GB/T 1800.1 | Missing tolerance annotations |
| Title Block | GB/T 10609.1 | Incomplete title block information |
| Symbols | GB/T 131 | Missing surface roughness (Ra) annotations |
| Geometric Completeness | — | Insufficient geometric elements, missing arrows |
| Welding Symbols | GB/T 324 | Missing or incorrect welding symbols, improper annotation placement |
| Drawing Sheet Specification | GB/T 14689 | Incorrect sheet size, improper border margins, wrong title block placement |
| Surface Roughness | GB/T 131 | Missing Ra values, incorrect roughness symbol orientation, missing machining marks |
| View Annotation | GB/T 17451 | Missing view labels, incorrect section view markings, improper projection direction |

> **Note**: The last 4 categories (Welding Symbols, Drawing Sheet Specification, Surface Roughness, View Annotation) are new in V2 and are supported in the Atlas Drawing Atlas.

## V2 Upgrade Guide

### Upgrading from V1 to V2

V2 is **backward compatible** with V1. All V1 routes (`/`, `/upload`) continue to function unchanged. V2 features are opt-in via new routes and configuration.

#### Step 1: Install New Dependency

```bash
pip install pyyaml
```

Or reinstall all dependencies:

```bash
pip install -r requirements.txt
```

#### Step 2: Update Configuration

Add V2 configuration options to your `config.py` (or set environment variables):

```python
VLM_MODEL = 'Qwen2.5-VL-72B-Instruct'
VLM_JUDGE_ENABLED = True
ENABLE_ATLAS_PACK = True
ATLAS_RULE_MODE = 'strict'
```

If `VLM_JUDGE_ENABLED` is `False`, the V2 pipeline skips the Judge phase and falls back to rule-based scoring, maintaining full functionality without a VLM endpoint.

#### Step 3: Access V2 Features

- Use `/v2` instead of `/` for the enhanced upload UI
- Use `/v2/upload` instead of `/upload` for the 6-phase pipeline
- V1 endpoints remain fully functional at their original paths

#### Key Differences Between V1 and V2

| Feature | V1 | V2 |
|---------|----|----|
| Pipeline Phases | 4 (Perception → OCR Enhancement → RuleCheck → LLM) | 6 (Planning → Perception → RuleCheck → LLM → Fusion → Judge) |
| Error Recovery | Full pipeline retry | Cross-Stage Rollback (targeted) |
| Case Retrieval | None | Asymmetric Experience Store (FAISS) |
| Quality Evaluation | None | VLM Judge (4-dim) |
| Drawing Atlas | None | Atlas (34 cases, 24 rules, 8 categories) |
| RL Reward | Scalar (single value) | 4-dim continuous vector |
| Error Categories | 6 | 10 (4 new GB/T categories) |
| API Routes | `/`, `/upload` | `/v2`, `/v2/upload` + Atlas API |

#### Migration Notes

- **No database migration needed**: V2 uses the same data directory structure; new data is stored in additional subdirectories
- **RL experience data**: V1 experience data is compatible; V2 adds 4-dim reward fields (old entries default to scalar reward)
- **Config backward compatibility**: All V1 config options work in V2; V2 options have sensible defaults
- **API backward compatibility**: All V1 endpoints continue to work; V2 endpoints are additive

## Security Notes

- **NEVER** commit `config.py` to any public repository — it contains API keys
- **NEVER** hardcode API keys in source code — always use `config.py` or environment variables
- The `config.py` file is listed in `.gitignore` to prevent accidental commits
- If API keys are accidentally exposed, rotate them immediately
- The system falls back to local rule engine when API is unavailable, ensuring core functionality without external dependencies
- VLM Judge falls back to rule-based scoring when VLM endpoint is unavailable
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
pyyaml>=6.0
```
