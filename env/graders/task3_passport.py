"""Task 3 grader — Passport Renewal (hard).

Score = (field_score × 0.5) + (conflict_resolution × 0.3) + (completeness_bonus × 0.2)

conflict_resolution: agent must detect and correct 2 pre-seeded conflicting fields.
completeness_bonus: 0.2 only if ALL 14 fields valid AND no conflicts remain.
Any unresolved conflict → conflict_resolution score = 0.0.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict

from env.graders.base import BaseGrader
from env.models import FieldStatus


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError):
        return None


class PassportGrader(BaseGrader):
    def grade(self, final_state: Dict[str, Any]) -> float:
        fields = final_state.get("fields", [])
        field_map: Dict[str, Dict[str, Any]] = {f["name"]: f for f in fields}

        # --- field_score (all 14 required) --------------------------------
        required = [f for f in fields if f.get("required", True)]
        if not required:
            return 0.0
        valid_count = sum(
            1 for f in required if f.get("status") == FieldStatus.VALID.value
        )
        field_score = valid_count / len(required)

        # --- conflict_resolution (3 cross-field rules) --------------------
        conflicts_resolved = 0
        total_conflict_rules = 3

        # Rule 1: Applicant must be ≥ 18 years old
        dob = _parse_date(field_map.get("date_of_birth", {}).get("value"))
        app_date = _parse_date(field_map.get("application_date", {}).get("value"))
        if dob and app_date:
            age = (app_date - dob).days / 365.25
            if age >= 18:
                conflicts_resolved += 1
        # If either date is missing/unparseable, conflict is unresolved

        # Rule 2: emergency_contact_name ≠ applicant_name
        applicant = (field_map.get("applicant_name", {}).get("value") or "").strip().lower()
        emergency = (field_map.get("emergency_contact_name", {}).get("value") or "").strip().lower()
        if applicant and emergency and applicant != emergency:
            conflicts_resolved += 1

        # Rule 3: existing_passport_expiry not more than 3 years before application_date
        expiry = _parse_date(field_map.get("existing_passport_expiry", {}).get("value"))
        if expiry and app_date:
            three_years_ago = app_date - timedelta(days=3 * 365)
            if expiry >= three_years_ago:
                conflicts_resolved += 1

        conflict_score = conflicts_resolved / total_conflict_rules

        # --- completeness_bonus -------------------------------------------
        all_valid = valid_count == len(required)
        no_conflicts = conflicts_resolved == total_conflict_rules
        completeness_bonus = 1.0 if (all_valid and no_conflicts) else 0.0

        return (field_score * 0.5) + (conflict_score * 0.3) + (completeness_bonus * 0.2)
