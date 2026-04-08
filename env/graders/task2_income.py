"""Task 2 grader — Income Certificate (medium).

Score = (field_score × 0.6) + (cross_field_consistency × 0.4)
cross_field_consistency checks 3 inter-field rules deterministically.
Score 0.0 if *any* cross-field rule is violated, even if all fields individually valid.
"""

from __future__ import annotations

from typing import Any, Dict

from env.graders.base import BaseGrader
from env.models import FieldStatus


class IncomeGrader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        fields = final_state.get("fields", [])
        field_map: Dict[str, Dict[str, Any]] = {f["name"]: f for f in fields}

        # --- field_score --------------------------------------------------
        required = [f for f in fields if f.get("required", True)]
        if not required:
            return 0.0
        valid_count = sum(
            1 for f in required if f.get("status") == FieldStatus.VALID.value
        )
        field_score = valid_count / len(required)

        # --- cross-field consistency (all-or-nothing) ----------------------
        cross_ok = True

        # Rule 1: BPL → income < 3,00,000
        cert = field_map.get("certificate_type", {})
        income = field_map.get("annual_income", {})
        if (
            cert.get("value") == "BPL"
            and income.get("value") is not None
        ):
            try:
                if int(income["value"]) >= 300_000:
                    cross_ok = False
            except (ValueError, TypeError):
                cross_ok = False

        # Rule 2: EWS → income < 8,00,000
        if (
            cert.get("value") == "EWS"
            and income.get("value") is not None
        ):
            try:
                if int(income["value"]) >= 800_000:
                    cross_ok = False
            except (ValueError, TypeError):
                cross_ok = False

        # Rule 3: Salaried → employer_name must be filled
        emp_type = field_map.get("employment_type", {})
        employer = field_map.get("employer_name", {})
        if emp_type.get("value") == "Salaried":
            if not employer.get("value"):
                cross_ok = False

        cross_field_score = 1.0 if cross_ok else 0.0

        return (field_score * 0.6) + (cross_field_score * 0.4)
