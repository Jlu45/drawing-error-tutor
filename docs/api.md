# API Documentation

## REST API Endpoints

### Upload and Analyze

**POST** `/upload`

Upload an engineering drawing image for analysis.

**Request**: `multipart/form-data`
- `file`: Image file (PNG, JPG, JPEG, GIF, BMP)

**Response**: HTML result page with analysis report

---

### GB Standards Search

**GET** `/api/gb_standards?q=<query>`

Search GB/T national standards knowledge base.

**Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `q` | string | No | Search query. Returns all standards if empty. |

**Response**: JSON array of matching standards

```json
[
  {
    "title": "尺寸注法",
    "content": "尺寸标注应完整、清晰...",
    "source": "GB/T 14665-2012",
    "is_gb_standard": true
  }
]
```

---

### RL Feedback

**POST** `/api/rl_feedback`

Submit user feedback for RL-based self-evolution.

**Request**: `application/json`
```json
{
  "session_id": "drawing.png_1700000000",
  "error_description": "Missing diameter symbol Φ",
  "feedback_type": "confirmed"
}
```

**Valid feedback types**:
| Type | Reward | Description |
|------|--------|-------------|
| `confirmed` | +1.0 | Error confirmed by user |
| `useful_guidance` | +0.5 | Learning guidance was helpful |
| `partial_confirm` | +0.3 | Partially confirmed |
| `ignored` | -0.5 | Error was irrelevant |
| `dismissed_all` | -1.0 | All errors were wrong |

**Response**:
```json
{
  "success": true,
  "message": "Feedback \"confirmed\" recorded for session drawing.png_1700000000",
  "rl_stats": {
    "buffer_size": 42,
    "training_count": 15,
    "epsilon": 0.245,
    "params_version": 8,
    "policy_params": { ... }
  }
}
```

---

### RL Statistics

**GET** `/api/rl_stats`

Get current RL memory unit statistics.

**Response**:
```json
{
  "buffer_size": 42,
  "training_count": 15,
  "epsilon": 0.245,
  "params_version": 8,
  "policy_params": {
    "severity_weight_high": 3.2,
    "severity_weight_medium": 1.8,
    "severity_weight_low": 1.1,
    "score_penalty_per_weight": 5.5,
    "llm_score_fusion_ratio": 0.55,
    "ocr_enhance_threshold": 5,
    "rule_confidence_threshold": 0.35,
    "version": 8
  },
  "active_sessions": 1
}
```

## Internal APIs (src/)

### DrawingOrchestrator

```python
from src.multi_agent_system import DrawingOrchestrator

orchestrator = DrawingOrchestrator(
    api_url="https://your-api-endpoint.com",
    api_key="your-api-key",
    llm_model="Qwen2.5-72B-Instruct"
)

result = orchestrator.analyze(
    image_path="path/to/drawing.png",
    background_knowledge="..."  # Optional background knowledge text
)
```

**Result structure**:
```python
{
    'ocr_results': [...],           # OCR text items
    'detection_results': [...],     # Detected geometric elements
    'errors': [...],                # All detected errors
    'feedback': [...],              # Socratic learning guidance
    'api_result': {...},            # LLM API response (or None)
    'geo_result': {...},            # Geometry detection results
    'structure_result': {...},      # Structure analysis results
    'report': {
        'total_errors': 5,
        'error_categories': {'尺寸标注': 2, '线型': 1, ...},
        'overall_score': 72,
        'summary': "..."
    },
    'metrics': {
        'total_time_ms': 3500.0,
        'agent_timings': {'ocr': 800, 'geometry': 1200, ...},
        'quality_score': 0.75,
        'rl_session_id': 'drawing.png_1700000000',
        'rl_stats': {...}
    }
}
```

### DualKnowledgeBase

```python
from src.rag_knowledge_base import DualKnowledgeBase

kb = DualKnowledgeBase()

# Search GB standards
results = kb.search_gb_standards("尺寸标注", top_k=5)

# Get background knowledge for LLM prompt
text = kb.get_background_knowledge_text(max_chars=2000)

# Add custom knowledge
kb.add_text_knowledge("Title", "Content", "Source")
kb.add_image_knowledge("path/to/image.png", {"description": "..."})
```

### RLMemoryUnit

```python
from src.rl_memory_unit import RLMemoryUnit

rl = RLMemoryUnit(state_dim=10)

# Get current policy parameters
params = rl.get_policy_params()

# Get statistics
stats = rl.get_stats()
```

## Error Categories Reference

| Category | GB Reference | Typical Issues |
|----------|-------------|----------------|
| 尺寸标注 | GB/T 4458.4 | Missing dimensions, missing Φ symbol, cramped spacing |
| 线型 | GB/T 4457.4 | Missing center lines, incorrect solid line ratio |
| 公差 | GB/T 1800.1 | Missing tolerance annotations |
| 标题栏 | GB/T 10609.1 | Incomplete title block information |
| 符号 | GB/T 131 | Missing surface roughness (Ra) annotations |
| 几何完整性 | — | Insufficient geometric elements, missing arrows |
