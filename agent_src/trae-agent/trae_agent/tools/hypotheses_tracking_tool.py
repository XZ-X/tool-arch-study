# Copyright (c) 2023 Anthropic
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 13 June 2025
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/anthropics/anthropic-quickstarts/blob/main/LICENSE
#
# This modified file is released under the same license.

import json
from dataclasses import dataclass
from typing import override

from .base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    hypothesis: str
    status: str  # "active" | "confirmed" | "invalidated" | "parked"
    confidence: float | None = None  # 0.0 - 1.0 (optional)
    rationale: str | None = None
    evidence_for: list[str] | None = None
    evidence_against: list[str] | None = None
    next_tests: list[str] | None = None


@dataclass
class HypothesisUpdateData:
    note: str | None = None
    hypotheses: list[HypothesisRecord] | None = None
    add: list[HypothesisRecord] | None = None
    update: list[HypothesisRecord] | None = None
    remove_ids: list[str] | None = None
    change_summary: list[str] | None = None
    next_action: str | None = None


class HypothesesTrackingTool(Tool):
    """A tool for hypothesis scaffolding that helps structure and maintain hypotheses.

    This tool keeps hypotheses explicit, structured, and easy to revise as new information arrives.
    """

    @override
    def get_name(self) -> str:
        return "hypotheses_tracking"

    @override
    def get_description(self) -> str:
        return """A structured tool for managing hypotheses during problem-solving.

Use this tool to keep an explicit, up-to-date set of hypotheses about the problem.
After each non-trivial observation, update the hypothesis set: add new hypotheses,
revise existing ones, adjust confidence, and mark hypotheses as confirmed or invalidated.

When to use this tool:
- Debugging or investigating root causes
- Exploratory coding / ambiguous requirements
- Planning experiments, tests, and validation steps
- Any task where you want explicit belief tracking

Hypothesis structure:
- hypothesis_id: stable identifier (keep stable across revisions)
- hypothesis: short, testable claim
- status: one of "active", "parked", "invalidated", "confirmed"
- confidence: optional number in [0.0, 1.0]
- rationale: optional short explanation
- evidence_for / evidence_against: optional concrete bullets (tests, logs, constraints, user statements)
- next_tests: optional actionable, discriminative tests (separate hypotheses)

Recommended lifecycle:
- active: plausible and being considered
- parked: plausible but not currently pursued
- invalidated: contradicted by evidence
- confirmed: strongly supported / effectively proven

Confidence guidance (optional):
- 0.2: weakly plausible
- 0.5: credible
- 0.8: strong
- 0.95+: near-certain

Update modes:
- Provide the full current hypothesis set in "hypotheses", OR
- Provide a delta update using "add", "update", and "remove_ids"

Guidelines:
1. Prefer frequent small updates.
2. Invalidate quickly; confirm only with strong evidence.
3. Keep evidence concrete; keep next_tests discriminative.
4. Preserve hypothesis_id stability when revising.
5. Use delta updates for small changes; send full state when many changes occur."""

    @override
    def get_parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="note",
                type="string",
                description="Optional context note about why you're updating hypotheses.",
            ),
            ToolParameter(
                name="hypotheses",
                type="array",
                description=(
                    "Full current hypothesis set. Each item is an object with fields: "
                    "hypothesis_id (string), hypothesis (string), status (string), "
                    "confidence (number, optional), rationale (string, optional), "
                    "evidence_for (array of strings, optional), evidence_against (array of strings, optional), "
                    "next_tests (array of strings, optional)."
                ),
            ),
            ToolParameter(
                name="add",
                type="array",
                description="Hypotheses to add (same object schema as in hypotheses).",
            ),
            ToolParameter(
                name="update",
                type="array",
                description="Hypotheses to update/replace by hypothesis_id (same object schema as in hypotheses).",
            ),
            ToolParameter(
                name="remove_ids",
                type="array",
                description="Hypothesis IDs to remove from active tracking (array of strings).",
            ),
            ToolParameter(
                name="change_summary",
                type="string",
                description="Concise summary of what changed.",
            ),
            ToolParameter(
                name="next_action",
                type="string",
                description="Optional: what you plan to do next, based on the current hypothesis set.",
            ),
        ]


    def __init__(self, model_provider: str | None = None, is_live: bool = False) -> None:
        super().__init__(model_provider)
        self.history: list[HypothesisUpdateData] = []
        self.current: dict[str, HypothesisRecord] = {}
        self.is_live = is_live

    @override
    def get_model_provider(self) -> str | None:
        return self._model_provider

    def _validate_record(self, obj: object, field_name: str) -> HypothesisRecord:
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid {field_name} item: must be an object")

        def _req_str(key: str) -> str:
            if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
                raise ValueError(f"Invalid {field_name} item: {key} must be a non-empty string")
            return str(obj[key]).strip()

        hypothesis_id = _req_str("hypothesis_id")
        hypothesis = _req_str("hypothesis")

        status = obj.get("status")
        if not isinstance(status, str) or status not in {"active", "confirmed", "invalidated", "parked"}:
            raise ValueError(
                f"Invalid {field_name} item: status must be one of "
                '"active", "confirmed", "invalidated", "parked"'
            )

        confidence = obj.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Invalid {field_name} item: confidence must be a number")
            confidence = float(confidence)
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(f"Invalid {field_name} item: confidence must be between 0.0 and 1.0")

        def _opt_str(key: str) -> str | None:
            v = obj.get(key)
            if v is None:
                return None
            if not isinstance(v, str):
                raise ValueError(f"Invalid {field_name} item: {key} must be a string")
            return str(v)

        def _opt_str_list(key: str) -> list[str] | None:
            v = obj.get(key)
            if v is None:
                return None
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                raise ValueError(f"Invalid {field_name} item: {key} must be an array of strings")
            return [str(x) for x in v]

        return HypothesisRecord(
            hypothesis_id=hypothesis_id,
            hypothesis=hypothesis,
            status=str(status),
            confidence=confidence,
            rationale=_opt_str("rationale"),
            evidence_for=_opt_str_list("evidence_for"),
            evidence_against=_opt_str_list("evidence_against"),
            next_tests=_opt_str_list("next_tests"),
        )

    def _validate_update_data(self, arguments: ToolCallArguments) -> HypothesisUpdateData:
        note = arguments.get("note")
        if note is not None and not isinstance(note, str):
            raise ValueError("Invalid note: must be a string")

        change_summary = arguments.get("change_summary")
        if change_summary is not None and not isinstance(change_summary, str):
            raise ValueError("Invalid change_summary: must be a string")

        next_action = arguments.get("next_action")
        if next_action is not None and not isinstance(next_action, str):
            raise ValueError("Invalid next_action: must be a string")

        def _opt_records(key: str) -> list[HypothesisRecord] | None:
            v = arguments.get(key)
            if v is None:
                return None
            if not isinstance(v, list):
                raise ValueError(f"Invalid {key}: must be an array of objects")
            return [self._validate_record(item, key) for item in v]

        hypotheses = _opt_records("hypotheses")
        add = _opt_records("add")
        update = _opt_records("update")

        remove_ids = arguments.get("remove_ids")
        if remove_ids is not None:
            if not isinstance(remove_ids, list) or any(not isinstance(x, str) for x in remove_ids):
                raise ValueError("Invalid remove_ids: must be an array of strings")
            remove_ids = [str(x) for x in remove_ids]

        # Require at least one meaningful field
        if all(
            v is None
            for v in [hypotheses, add, update, remove_ids, note, change_summary, next_action]
        ):
            raise ValueError("No update provided: include hypotheses and/or add/update/remove_ids and/or note")

        return HypothesisUpdateData(
            note=note,
            hypotheses=hypotheses,
            add=add,
            update=update,
            remove_ids=remove_ids,
            change_summary=change_summary,
            next_action=next_action,
        )

    def _apply(self, data: HypothesisUpdateData) -> None:
        # If full state is provided, replace current state
        if data.hypotheses is not None:
            self.current = {h.hypothesis_id: h for h in data.hypotheses}

        # Apply deltas on top
        if data.add:
            for h in data.add:
                self.current[h.hypothesis_id] = h

        if data.update:
            for h in data.update:
                self.current[h.hypothesis_id] = h

        if data.remove_ids:
            for hid in data.remove_ids:
                self.current.pop(hid, None)

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            validated = self._validate_update_data(arguments)
            self._apply(validated)
            self.history.append(validated)

            active = sum(1 for h in self.current.values() if h.status == "active")
            parked = sum(1 for h in self.current.values() if h.status == "parked")
            invalidated = sum(1 for h in self.current.values() if h.status == "invalidated")
            confirmed = sum(1 for h in self.current.values() if h.status == "confirmed")

            response_data = {
                "hypotheses_total": len(self.current),
                "active": active,
                "parked": parked,
                "invalidated": invalidated,
                "confirmed": confirmed,
                "history_length": len(self.history),
            }

            return ToolExecResult(
                output=(
                    "Hypothesis scaffolding update completed.\n\nStatus:\n"
                    f"{json.dumps(response_data, indent=2)}"
                )
            )

        except Exception as e:
            error_data = {"error": str(e), "status": "failed"}
            return ToolExecResult(
                error=(
                    f"Hypothesis scaffolding failed: {str(e)}\n\nDetails:\n"
                    f"{json.dumps(error_data, indent=2)}"
                ),
                error_code=-1,
            )
