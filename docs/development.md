# Development Guide

## Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/your-username/drawing-error-correction.git
cd drawing-error-correction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest pytest-cov flake8 mypy

# Configure API access
cp config.example.py config.py
# Edit config.py with your API credentials
```

## Project Architecture

### Multi-Agent Pipeline

The analysis pipeline consists of 4 phases:

1. **Phase 1 (Parallel)**: OCR + Geometry + Structure agents run concurrently
2. **Phase 2 (Conditional)**: Enhanced OCR for title block if initial results insufficient
3. **Phase 3**: Rule-based error checking against GB standards
4. **Phase 4**: LLM deep analysis (with fallback to local rule engine)

### Key Design Patterns

- **Agent Pattern**: All agents inherit from `BaseAgent` with standardized `analyze()` interface
- **Result Fusion**: Rule-based and LLM-detected errors are merged with deduplication
- **Graceful Degradation**: System continues to function when external APIs are unavailable
- **RL Self-Evolution**: User feedback drives policy parameter adaptation

### Adding New Error Detection Rules

1. Add a new method in `RuleCheckAgent` class (e.g., `_check_new_rule`)
2. Call it from `_do_analyze()` method
3. Add corresponding GB reference in `LLMAgent._generate_local_analysis()`
4. Add test cases in `tests/test_rule_check.py`

### Adding New Agents

1. Create a class inheriting from `BaseAgent`
2. Implement `_do_initialize()` and `_do_analyze()` methods
3. Register in `DrawingOrchestrator._register_agents()`
4. Integrate into the pipeline in `DrawingOrchestrator.analyze()`
5. Add tests in `tests/`

### Code Style

- Follow PEP 8 with 120-character line limit
- Use type hints for function signatures
- Write docstrings for public methods
- Run `flake8 src/ tests/ --max-line-length=120` before committing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test
pytest tests/test_geometric_detector.py -v

# Skip slow tests
pytest -m "not slow"
```

### Debugging Tips

1. Enable debug logging:
   ```python
   import logging
   logging.getLogger("DrawingAgent").setLevel(logging.DEBUG)
   ```

2. Test individual agents:
   ```python
   from src.multi_agent_system import OCRAgent
   agent = OCRAgent()
   result = agent.analyze("path/to/drawing.png")
   print(result.data)
   ```

3. Inspect RL state:
   ```python
   from src.rl_memory_unit import RLMemoryUnit
   rl = RLMemoryUnit()
   print(rl.get_stats())
   ```
