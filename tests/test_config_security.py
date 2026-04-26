import pytest
import os
import sys
import hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_KNOWN_SECRET_HASHES = [
    "a3f5e8c1b2d4f6a8e0c2b4d6f8a0e2c4b6d8f0a2e4c6b8d0f2a4e6c8b0d2f4",
    "e5c7a9b1d3f5e7a9b1d3f5e7a9b1d3f5e7a9b1d3f5e7a9b1d3f5e7a9b1d3",
]

def _check_content_for_known_secrets(content: str) -> bool:
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('//'):
            continue
        if len(stripped) > 20 and not any(c in stripped for c in '{}[]()=+-*/<>'):
            h = hashlib.sha256(stripped.encode()).hexdigest()[:58]
            if h in _KNOWN_SECRET_HASHES:
                return True
    return False


class TestConfigLoader:
    def test_config_example_exists(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        config_example = os.path.join(base_dir, 'config.example.py')
        assert os.path.exists(config_example), "config.example.py must exist"

    def test_config_example_has_placeholders(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        config_example = os.path.join(base_dir, 'config.example.py')
        with open(config_example, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "your-api-key-here" in content or "MULTIMODAL_API_KEY" in content
        assert "your-api-endpoint" in content or "MULTIMODAL_API_URL" in content

    def test_config_example_no_real_keys(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        config_example = os.path.join(base_dir, 'config.example.py')
        with open(config_example, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "os.environ.get" in content or "''" in content
        lines_with_keys = [l for l in content.split('\n') if 'API_KEY' in l and '=' in l]
        for line in lines_with_keys:
            if "os.environ.get" not in line and "''" not in line and '""' not in line:
                value_part = line.split('=', 1)[1].strip().strip("'\"")
                if len(value_part) > 10 and value_part not in ('your-api-key-here',):
                    pytest.fail(f"Possible hardcoded key in config.example.py: {line.strip()}")

    def test_gitignore_protects_config(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        gitignore_path = os.path.join(base_dir, '.gitignore')
        assert os.path.exists(gitignore_path), ".gitignore must exist"
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "config.py" in content
        assert ".env" in content

    def test_no_secrets_in_source(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        suspicious_patterns = [
            'api_key = "',
            "api_key = '",
            'API_KEY = "',
            "API_KEY = '",
        ]
        for root, dirs, files in os.walk(os.path.join(base_dir, 'src')):
            for fname in files:
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    for pattern in suspicious_patterns:
                        if pattern in content:
                            for line in content.split('\n'):
                                if pattern in line and 'os.environ' not in line and 'config' not in line.lower():
                                    value = line.split('=', 1)[1].strip().strip("'\"")
                                    if len(value) > 10 and value not in ('your-api-key-here',):
                                        pytest.fail(f"Possible hardcoded secret in {fpath}: {line.strip()}")

    def test_no_internal_urls_in_source(self):
        base_dir = os.path.join(os.path.dirname(__file__), '..')
        internal_indicators = ['.h3i.', '.buaa.edu.cn', '.internal.']
        for root, dirs, files in os.walk(os.path.join(base_dir, 'src')):
            for fname in files:
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    for indicator in internal_indicators:
                        if indicator in content:
                            pytest.fail(f"Internal URL pattern '{indicator}' found in {fpath}")
