"""Pydantic models for the GovForm OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FieldStatus(str, Enum):
    """Status of a single form field."""
    EMPTY = "empty"
    FILLED = "filled"
    INVALID = "invalid"
    VALID = "valid"


class FormField(BaseModel):
    """One field in a government form."""
    name: str
    label: str
    value: Optional[str] = None
    status: FieldStatus = FieldStatus.EMPTY
    error_message: Optional[str] = None
    required: bool = True


class Observation(BaseModel):
    """What the agent sees after every step."""
    form_id: str
    task_id: str
    fields: List[FormField]
    last_agent_action: Optional[str] = None
    last_system_message: str = ""
    filled_count: int = 0
    valid_count: int = 0
    total_required: int = 0
    step_number: int = 0


class Action(BaseModel):
    """What the agent sends to the environment."""
    field_name: str
    value: str
    reasoning: Optional[str] = None


class Reward(BaseModel):
    """Reward signal returned after each step."""
    value: float
    breakdown: Dict[str, float]
    done: bool
    info: Dict[str, Any] = {}
