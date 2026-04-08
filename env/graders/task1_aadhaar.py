"""Task 1 grader — Aadhaar Address Update (easy).

Score = valid_fields / total_required_fields
Each correct field contributes 1/6 of the score.
"""

from __future__ import annotations

from typing import Any, Dict

from env.graders.base import BaseGrader
from env.models import FieldStatus


class AadhaarGrader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        fields = final_state.get("fields", [])
        required = [f for f in fields if f.get("required", True)]
        if not required:
            return 0.0
        valid = sum(
            1 for f in required if f.get("status") == FieldStatus.VALID.value
        )
        return valid / len(required)
