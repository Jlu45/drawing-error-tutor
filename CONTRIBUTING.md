# Contributing to Engineering Drawing Intelligent Error Correction

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/drawing-error-correction.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest`
6. Run linter: `flake8 src/ tests/`
7. Commit and push: `git commit -m "Add: your feature" && git push origin feature/your-feature-name`
8. Open a Pull Request

## 🐛 Reporting Issues

When reporting bugs, please include:

- **Python version** (`python --version`)
- **OS and version**
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Error logs** (if applicable)
- **Screenshots** (for UI issues)

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md) when creating an issue.

## ✨ Requesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md) and describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternative approaches you've considered

## 📝 Code Style

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Maximum line length: 120 characters
- Use type hints for function signatures
- Run `flake8 src/ tests/` before committing

### HTML/CSS/JS

- 2-space indentation
- Use semantic HTML elements
- Follow existing naming conventions in templates

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new dimension rule checker
fix: correct circle detection threshold
docs: update API documentation
test: add unit tests for geometric detector
refactor: simplify OCR preprocessing pipeline
chore: update dependencies
```

## 🧪 Testing

- Write tests for all new functionality
- Maintain test coverage above 70%
- Run the full test suite before submitting PRs:

```bash
pytest -v
pytest --cov=src --cov-report=term-missing
```

## 🔒 Security

- **NEVER** commit API keys, passwords, or other secrets
- **NEVER** commit `config.py` — only modify `config.example.py`
- Report security vulnerabilities privately via email, not in public issues

## 📋 Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated for new functionality
- [ ] All tests pass (`pytest`)
- [ ] Linter passes (`flake8 src/ tests/`)
- [ ] No hardcoded secrets or API keys
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (if applicable)

## 🏗️ Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov flake8 mypy

# Run in debug mode
export FLASK_DEBUG=true
python app.py

# Run tests
pytest -v

# Run linter
flake8 src/ tests/ --max-line-length=120
```

## 📦 Adding New Error Detection Rules

To add a new GB standard rule:

1. Add the rule method in `src/multi_agent_system.py` → `RuleCheckAgent`
2. Add corresponding GB reference in `_generate_local_analysis()`
3. Add test cases in `tests/test_rule_check.py`
4. Update documentation in `docs/`

## 🤖 Adding New Agents

To add a new analysis agent:

1. Create a class inheriting from `BaseAgent` in `src/multi_agent_system.py`
2. Implement `_do_initialize()` and `_do_analyze()` methods
3. Register the agent in `DrawingOrchestrator._register_agents()`
4. Add the agent to the analysis pipeline in `DrawingOrchestrator.analyze()`
5. Write tests in `tests/`

Thank you for contributing! 🎉
