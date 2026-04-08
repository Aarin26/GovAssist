"""GovFormEnv — the core environment: reset / step / state."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from govform_env.models import Action, FieldStatus, FormField, Observation
from govform_env.reward import compute_reward

# Base path for form JSON schemas
_FORMS_DIR = Path(__file__).resolve().parent / "forms"


class GovFormEnv:
    """Stateful environment for one government form-filling episode."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._schema: Dict[str, Any] = {}
        self._fields: List[FormField] = []
        self._step_number: int = 0
        self._done: bool = False
        self._last_agent_action: Optional[str] = None
        self._last_system_message: str = ""
        self._cumulative_reward: float = 0.0

    # ── public API ────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Load the form schema and return an observation with all fields EMPTY."""
        schema_path = _FORMS_DIR / f"{self.task_id}.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"No form schema for task_id={self.task_id!r}")
        with open(schema_path, encoding="utf-8") as f:
            self._schema = json.load(f)

        self._fields = [
            FormField(
                name=fd["name"],
                label=fd["label"],
                required=fd.get("required", True),
            )
            for fd in self._schema["fields"]
        ]
        self._step_number = 0
        self._done = False
        self._last_agent_action = None
        self._last_system_message = "Form loaded. All fields are empty."
        self._cumulative_reward = 0.0
        return self._observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Process one agent action and return (obs, reward, done, info)."""
        if self._done:
            return self._observation(), 0.0, True, {"message": "Episode already finished."}

        prev_obs = self._observation()
        self._step_number += 1
        self._last_agent_action = f"{action.field_name}={action.value}"

        # 1. Find the target field
        target = self._field_by_name(action.field_name)
        if target is None:
            self._last_system_message = (
                f"Field '{action.field_name}' does not exist in this form."
            )
            new_obs = self._observation()
            reward = compute_reward(prev_obs, action, new_obs, self._done)
            self._cumulative_reward += reward
            return new_obs, reward, self._done, {"error": "unknown_field"}

        # 2. Set the value
        target.value = action.value

        # 3. Field-level validation
        self._validate_field(target)

        # 4. Cross-field validation (may override status)
        self._validate_cross_field_rules()

        # 5. Update system message
        if target.status == FieldStatus.VALID:
            self._last_system_message = f"Field '{target.label}' accepted."
        else:
            self._last_system_message = (
                f"Field '{target.label}' invalid: {target.error_message}"
            )

        # 6. Check done
        self._done = self._check_done()

        new_obs = self._observation()
        reward = compute_reward(prev_obs, action, new_obs, self._done)
        self._cumulative_reward += reward

        info: Dict[str, Any] = {}
        if self._done:
            info["message"] = "All required fields are valid. Episode complete!"
            info["cumulative_reward"] = self._cumulative_reward

        return new_obs, reward, self._done, info

    def state(self) -> dict:
        """Return a serialisable snapshot of the full current state."""
        return {
            "task_id": self.task_id,
            "form_id": self._schema.get("form_id", self.task_id),
            "title": self._schema.get("title", ""),
            "fields": [f.model_dump() for f in self._fields],
            "step_number": self._step_number,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
        }

    # ── private helpers ───────────────────────────────────────────────────

    def _observation(self) -> Observation:
        required_fields = [f for f in self._fields if f.required]
        return Observation(
            form_id=self._schema.get("form_id", self.task_id),
            task_id=self.task_id,
            fields=[f.model_copy() for f in self._fields],
            last_agent_action=self._last_agent_action,
            last_system_message=self._last_system_message,
            filled_count=sum(1 for f in self._fields if f.value is not None),
            valid_count=sum(1 for f in required_fields if f.status == FieldStatus.VALID),
            total_required=len(required_fields),
            step_number=self._step_number,
        )

    def _field_by_name(self, name: str) -> Optional[FormField]:
        for f in self._fields:
            if f.name == name:
                return f
        return None

    def _field_schema(self, name: str) -> Optional[dict]:
        for fd in self._schema.get("fields", []):
            if fd["name"] == name:
                return fd
        return None

    def _validate_field(self, field: FormField) -> None:
        """Run the individual validation rule from the schema."""
        fd = self._field_schema(field.name)
        if fd is None:
            return

        validation = fd.get("validation")
        if validation is None:
            # No validation rule → any non-empty value is valid
            if field.value:
                field.status = FieldStatus.VALID
                field.error_message = None
            else:
                field.status = FieldStatus.EMPTY
            return

        vtype = validation["type"]
        value = field.value or ""

        if vtype == "regex":
            pattern = validation["pattern"]
            if re.match(pattern, value):
                field.status = FieldStatus.VALID
                field.error_message = None
            else:
                field.status = FieldStatus.INVALID
                field.error_message = validation.get("error", "Invalid value.")

        elif vtype == "enum":
            allowed = validation.get("values", [])
            if value in allowed:
                field.status = FieldStatus.VALID
                field.error_message = None
            else:
                field.status = FieldStatus.INVALID
                field.error_message = validation.get(
                    "error", f"Must be one of: {', '.join(allowed)}"
                )

    def _validate_cross_field_rules(self) -> None:
        """Apply cross-field validation rules from the schema."""
        rules = self._schema.get("cross_field_rules", [])
        for rule in rules:
            self._apply_cross_field_rule(rule)

    def _apply_cross_field_rule(self, rule: dict) -> None:
        """Evaluate a single cross-field rule and set errors on offending fields."""
        rule_type = rule.get("rule", "")

        # ── Income-certificate style rules (condition_field → target_field) ──
        if "condition_field" in rule:
            cond_field = self._field_by_name(rule["condition_field"])
            target_field = self._field_by_name(rule["target_field"])
            if cond_field is None or target_field is None:
                return
            # Only check if both fields have values
            if cond_field.value is None or (target_field.value is None and rule_type != "not_empty"):
                return

            if rule_type == "less_than":
                if cond_field.value != rule["condition_value"]:
                    return  # Condition not met, rule doesn't apply
                try:
                    if int(target_field.value) >= rule["rule_value"]:
                        target_field.status = FieldStatus.INVALID
                        target_field.error_message = rule.get("error", "Cross-field rule violated.")
                except (ValueError, TypeError):
                    pass

            elif rule_type == "not_empty":
                if cond_field.value != rule["condition_value"]:
                    return
                if not target_field.value:
                    target_field.status = FieldStatus.INVALID
                    target_field.error_message = rule.get("error", "This field is required.")

        # ── Passport-style rules (field1, field2) ────────────────────────
        elif "field1" in rule and "field2" in rule:
            f1 = self._field_by_name(rule["field1"])
            f2 = self._field_by_name(rule["field2"])
            if f1 is None or f2 is None:
                return
            if f1.value is None or f2.value is None:
                return

            if rule_type == "min_age":
                dob = self._parse_date(f1.value)
                app_date = self._parse_date(f2.value)
                if dob and app_date:
                    age = (app_date - dob).days / 365.25
                    if age < rule["rule_value"]:
                        f1.status = FieldStatus.INVALID
                        f1.error_message = rule.get("error", "Age requirement not met.")

            elif rule_type == "not_equal":
                if f1.value.strip().lower() == f2.value.strip().lower():
                    f1.status = FieldStatus.INVALID
                    f1.error_message = rule.get("error", "Fields must not be equal.")

            elif rule_type == "max_years_before":
                d1 = self._parse_date(f1.value)
                d2 = self._parse_date(f2.value)
                if d1 and d2:
                    limit = d2 - timedelta(days=int(rule["rule_value"]) * 365)
                    if d1 < limit:
                        f1.status = FieldStatus.INVALID
                        f1.error_message = rule.get("error", "Date out of allowed range.")

    @staticmethod
    def _parse_date(s: str) -> Optional[date]:
        try:
            return date.fromisoformat(s)
        except (ValueError, TypeError):
            return None

    def _check_done(self) -> bool:
        """Episode is done when all required fields are VALID."""
        for f in self._fields:
            if f.required and f.status != FieldStatus.VALID:
                return False
        return True
