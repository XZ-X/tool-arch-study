import os
from pathlib import Path

import yaml


def resolve_api_key(value: str, path: str | Path = "api_keys.yaml") -> str:
    """Resolve a literal key, local key reference, or environment placeholder."""
    value = value or os.environ.get("OPENAI_API_KEY", "")
    key_path = Path(path)
    if key_path.exists():
        with key_path.open() as stream:
            keys = yaml.safe_load(stream) or {}
        value = keys.get(value, value)
    value = os.path.expandvars(str(value))
    if value == "openai-compatible":
        value = os.environ.get("OPENAI_API_KEY", "")
    if not value or "${" in value or value == "API_KEY_HERE":
        raise ValueError("Set OPENAI_API_KEY or provide an API key in the local configuration.")
    return value


def redact_secret(text: str, secret: str) -> str:
    """Remove the configured credential from diagnostic text."""
    return text.replace(secret, "<redacted>") if secret else text
