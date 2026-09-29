from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def resolve_api_key(api_keys_path: Path, ref: str) -> str:
    """Look up an API key reference in api_keys.yaml.

    Returns the resolved secret, or *ref* itself if the file doesn't exist
    or the key isn't found (treating *ref* as a literal value).
    """
    if api_keys_path.exists():
        with open(api_keys_path, "r") as f:
            all_keys = yaml.safe_load(f) or {}
        return os.path.expandvars(str(all_keys.get(ref, ref)))
    if ref == "openai-compatible":
        return os.environ.get("OPENAI_API_KEY", "")
    return os.path.expandvars(ref)


def load_model(models_path: str | Path, model_name: str) -> dict[str, Any]:
    """Load a model definition by short name from a models.yaml file.

    Returns a dict with ``"model"``, ``"base_url"``, and ``"api_key"``
    (resolved via sibling ``api_keys.yaml``).
    """
    models_path = Path(models_path)
    with open(models_path, "r") as f:
        data = yaml.safe_load(f)

    models = data.get("models", {})
    if model_name not in models:
        raise KeyError(
            f"Model '{model_name}' not found in {models_path}. "
            f"Available: {list(models.keys())}"
        )

    entry = models[model_name]
    cfg: dict[str, Any] = {
        "model": entry["model"],
        "base_url": entry["base_url"],
    }

    api_key_ref = entry.get("api_key")
    if api_key_ref:
        api_keys_path = models_path.parent / "api_keys.yaml"
        cfg["api_key"] = resolve_api_key(api_keys_path, api_key_ref)

    return cfg


def load_agent_config(config_path: str | Path, agent_name: str) -> dict[str, Any]:
    """Load an agent config by name from an agent_configs.yaml file.

    Resolves the model reference via a sibling ``models.yaml`` and
    ``api_keys.yaml``.  Returns a dict ready to unpack into
    ``CodingAgent(**config)``.
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    agents = data.get("agents", {})
    if agent_name not in agents:
        raise KeyError(
            f"Agent config '{agent_name}' not found in {config_path}. "
            f"Available: {list(agents.keys())}"
        )

    entry = agents[agent_name]

    # Resolve model reference
    model_ref = entry.get("model")
    if not model_ref:
        raise ValueError(
            f"Agent config '{agent_name}' is missing required 'model' field."
        )
    # Look for models.yaml: first in parent dir, then in config dir
    models_path = config_path.parent.parent / "models.yaml"
    if not models_path.exists():
        models_path = config_path.parent / "models.yaml"
    cfg = load_model(models_path, model_ref)

    # Merge agent behavior params
    _AGENT_PARAMS = {
        "max_steps": int,
        "max_tokens": int,
        "temperature": float,
        "tool_interface": str,
    }
    for key, cast in _AGENT_PARAMS.items():
        if key in entry:
            cfg[key] = cast(entry[key])

    return cfg


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a single JSON config file (trae-agent style).

    Returns a dict ready to unpack into ``CodingAgent(**config)``.
    Expects JSON format with:
    {
      "default_provider": "provider_name",
      "max_steps": 200,
      "model_providers": {
        "provider_name": {
          "model": "model-id",
          "base_url": "http://...",
          "api_key": "key_or_ref",
          "max_tokens": 4096,
          "temperature": 0,
          "tool_interface": "tool_call" or "python"
        }
      }
    }
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        data = json.load(f)

    default_provider = data.get("default_provider")
    if not default_provider:
        raise ValueError(f"Config {config_path} missing 'default_provider' field")

    model_providers = data.get("model_providers", {})
    if default_provider not in model_providers:
        raise KeyError(
            f"Provider '{default_provider}' not found in model_providers. "
            f"Available: {list(model_providers.keys())}"
        )

    provider_config = model_providers[default_provider]

    # Build config for CodingAgent
    cfg: dict[str, Any] = {
        "model": provider_config["model"],
        "base_url": provider_config["base_url"],
    }

    # Resolve API key (either literal or reference)
    api_key_ref = provider_config.get("api_key")
    if api_key_ref:
        api_keys_path = config_path.parent.parent / "api_keys.yaml"
        cfg["api_key"] = resolve_api_key(api_keys_path, api_key_ref)

    # Add optional params
    if "max_tokens" in provider_config:
        cfg["max_tokens"] = int(provider_config["max_tokens"])
    if "temperature" in provider_config:
        cfg["temperature"] = float(provider_config["temperature"])
    if "tool_interface" in provider_config:
        cfg["tool_interface"] = str(provider_config["tool_interface"])
    if "max_steps" in data:
        cfg["max_steps"] = int(data["max_steps"])

    return cfg
