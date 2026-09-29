# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

# TODO: remove these annotations by defining fine-grained types
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, override


# data class for model parameters
@dataclass
class ModelParameters:
    """Model parameters for a model provider."""

    model: str
    api_key: str = field(repr=False)
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    parallel_tool_calls: bool
    max_retries: int
    scope: str | None = None  # Azure specific field
    base_url: str | None = None
    api_version: str | None = None
    candidate_count: int | None = None  # Gemini specific field
    stop_sequences: list[str] | None = None
    frequency_penalty: float | None = None


@dataclass
class AgentAblationConfig:
    use_chat_completion: bool = False
    tool_bash_only: bool = False
    tool_find_code_context: bool = False
    use_atomic_tool_common: bool = False
    cog_use_hypotheses_tracking: bool = False
    cog_use_scratchpad: bool = False

@dataclass
class Config:
    """Configuration manager for Trae Agent."""
    default_provider: str
    max_steps: int
    model_providers: dict[str, ModelParameters]
    agent_ablation_config: AgentAblationConfig
    sweb_live: bool = False

    def _load_agent_ablation_config(self, config):
        """Load the study tool configuration."""
        self.agent_ablation_config = AgentAblationConfig(
            use_chat_completion=config.get("use_chat_completion", False),
            tool_bash_only=config.get("tool_bash_only", False),
            tool_find_code_context=config.get("tool_find_code_context", False),
            use_atomic_tool_common=config.get("use_atomic_tool_common", False),
            cog_use_hypotheses_tracking=config.get("cog_use_hypotheses_tracking", False),
            cog_use_scratchpad=config.get("cog_use_scratchpad", False),
        )

    def __init__(self, config_or_config_file: str | dict = "trae_config.json"):
        # Accept either file path or direct config dict
        if isinstance(config_or_config_file, dict):
            self._config = config_or_config_file
        else:
            config_path = Path(config_or_config_file)
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        self._config = json.load(f)
                except Exception as e:
                    print(
                        f"Warning: Could not load config file {config_or_config_file}: {e}"
                    )
                    self._config = {}
            else:
                self._config = {}

        self.default_provider = self._config.get("default_provider", "openai-compatible")
        self.max_steps = self._config.get("max_steps", 20)
        self.model_providers = {}

        if len(self._config.get("model_providers", [])) == 0:
            self.model_providers = {
                "openai-compatible": ModelParameters(
                    model="Sonnet-4.5",
                    api_key="",
                    max_tokens=4096,
                    temperature=0.5,
                    top_p=1,
                    top_k=0,
                    parallel_tool_calls=False,
                    max_retries=10,
                ),
            }
        else:
            for provider in self._config.get("model_providers", {}):
                provider_config: dict[str, Any] = self._config.get(
                    "model_providers", {}
                ).get(provider, {})

                candidate_count = provider_config.get("candidate_count")
                self.model_providers[provider] = ModelParameters(
                    model=str(provider_config.get("model", "")),
                    api_key=str(provider_config.get("api_key", "")),
                    max_tokens=int(provider_config.get("max_tokens", 1000)),
                    temperature=float(provider_config.get("temperature", 0.5)),
                    top_p=float(provider_config.get("top_p", 1)),
                    top_k=int(provider_config.get("top_k", 0)),
                    max_retries=int(provider_config.get("max_retries", 10)),
                    parallel_tool_calls=bool(
                        provider_config.get("parallel_tool_calls", False)
                    ),
                    base_url=str(provider_config.get("base_url"))
                    if "base_url" in provider_config
                    else None,
                    api_version=str(provider_config.get("api_version"))
                    if "api_version" in provider_config
                    else None,
                    candidate_count=int(candidate_count)
                    if candidate_count is not None
                    else None,
                    stop_sequences=provider_config.get("stop_sequences")
                    if "stop_sequences" in provider_config
                    else None,
                    frequency_penalty=float(provider_config.get("frequency_penalty", 0.0))
                    if "frequency_penalty" in provider_config
                    else None,
                )

        if 'ablation' in self._config:
            self._load_agent_ablation_config(config=self._config['ablation'])
        else:
            self.agent_ablation_config = AgentAblationConfig()

        return

    @override
    def __str__(self) -> str:
        return f"Config(default_provider={self.default_provider}, max_steps={self.max_steps}, model_providers={self.model_providers})"


def load_config(config_file: str = "trae_config.json") -> Config:
    """Load configuration from file."""
    return Config(config_file)


def resolve_config_value(
    cli_value: int | str | float | None,
    config_value: int | str | float | None,
    env_var: str | None = None,
) -> int | str | float | None:
    """Resolve configuration value with priority: CLI > ENV > Config > Default."""
    if cli_value is not None:
        return cli_value

    if env_var and os.getenv(env_var):
        return os.getenv(env_var)

    if config_value is not None:
        return config_value

    return None
