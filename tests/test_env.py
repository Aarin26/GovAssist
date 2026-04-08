"""Smoke tests for the GovForm OpenEnv environment.

Tests reset/step/state for each task, plus basic reward and grader behaviour.
"""

from __future__ import annotations

import pytest

from govform_env.environment import GovFormEnv
from govform_env.graders.task1_aadhaar import AadhaarGrader
from govform_env.graders.task2_income import IncomeGrader
from govform_env.graders.task3_passport import PassportGrader
from govform_env.models import Action, FieldStatus


# ── Helpers ───────────────────────────────────────────────────────────────

TASK_IDS = ["aadhaar_update", "income_certificate", "passport_renewal"]


# ── Reset tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_reset_returns_observation(task_id: str):
    env = GovFormEnv(task_id)
    obs = env.reset()
    assert obs.task_id == task_id
    assert obs.step_number == 0
    assert obs.valid_count == 0
    assert len(obs.fields) > 0
    assert all(f.status == FieldStatus.EMPTY for f in obs.fields)


# ── Step tests ───────────────────────────────────────────────────────────


def test_step_valid_field_aadhaar():
    env = GovFormEnv("aadhaar_update")
    env.reset()
    action = Action(field_name="full_name", value="Ravi Kumar")
    obs, reward, done, info = env.step(action)
    field = next(f for f in obs.fields if f.name == "full_name")
    assert field.status == FieldStatus.VALID
    assert reward > 0


def test_step_invalid_field_aadhaar():
    env = GovFormEnv("aadhaar_update")
    env.reset()
    action = Action(field_name="aadhaar_number", value="123")  # too short
    obs, reward, done, info = env.step(action)
    field = next(f for f in obs.fields if f.name == "aadhaar_number")
    assert field.status == FieldStatus.INVALID
    assert field.error_message is not None


def test_step_unknown_field():
    env = GovFormEnv("aadhaar_update")
    env.reset()
    action = Action(field_name="nonexistent_field", value="foo")
    obs, reward, done, info = env.step(action)
    assert reward < 0  # penalty
    assert "error" in info


# ── State tests ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_state_returns_dict(task_id: str):
    env = GovFormEnv(task_id)
    env.reset()
    state = env.state()
    assert isinstance(state, dict)
    assert "fields" in state
    assert state["task_id"] == task_id


# ── Done / full episode tests ────────────────────────────────────────────


def test_aadhaar_full_episode():
    """Fill all 6 fields with valid values → done should be True."""
    env = GovFormEnv("aadhaar_update")
    env.reset()

    actions = [
        Action(field_name="full_name", value="Aarav Sharma"),
        Action(field_name="aadhaar_number", value="123456789012"),
        Action(field_name="new_address_line1", value="42 MG Road Bengaluru"),
        Action(field_name="new_pincode", value="560001"),
        Action(field_name="state", value="Karnataka"),
        Action(field_name="mobile_number", value="9876543210"),
    ]
    for a in actions:
        obs, reward, done, info = env.step(a)

    assert done is True
    assert obs.valid_count == obs.total_required


def test_income_cross_field_bpl_violation():
    """BPL with income ≥ 3L should fail cross-field rule."""
    env = GovFormEnv("income_certificate")
    env.reset()

    env.step(Action(field_name="certificate_type", value="BPL"))
    obs, _, _, _ = env.step(Action(field_name="annual_income", value="500000"))
    field = next(f for f in obs.fields if f.name == "annual_income")
    assert field.status == FieldStatus.INVALID


def test_passport_emergency_not_self():
    """Emergency contact same as applicant should be invalid."""
    env = GovFormEnv("passport_renewal")
    env.reset()

    env.step(Action(field_name="applicant_name", value="Priya Singh"))
    obs, _, _, _ = env.step(Action(field_name="emergency_contact_name", value="Priya Singh"))
    field = next(f for f in obs.fields if f.name == "emergency_contact_name")
    assert field.status == FieldStatus.INVALID


# ── Grader tests ─────────────────────────────────────────────────────────


def test_aadhaar_grader_perfect():
    env = GovFormEnv("aadhaar_update")
    env.reset()
    actions = [
        Action(field_name="full_name", value="Aarav Sharma"),
        Action(field_name="aadhaar_number", value="123456789012"),
        Action(field_name="new_address_line1", value="42 MG Road Bengaluru"),
        Action(field_name="new_pincode", value="560001"),
        Action(field_name="state", value="Karnataka"),
        Action(field_name="mobile_number", value="9876543210"),
    ]
    for a in actions:
        env.step(a)
    score = AadhaarGrader().grade(env.state())
    assert score == pytest.approx(1.0)


def test_aadhaar_grader_partial():
    env = GovFormEnv("aadhaar_update")
    env.reset()
    # Fill only 3 of 6 fields
    env.step(Action(field_name="full_name", value="Aarav Sharma"))
    env.step(Action(field_name="aadhaar_number", value="123456789012"))
    env.step(Action(field_name="state", value="Karnataka"))
    score = AadhaarGrader().grade(env.state())
    assert score == pytest.approx(3 / 6)


def test_income_grader_with_valid_cross_fields():
    env = GovFormEnv("income_certificate")
    env.reset()
    actions = [
        Action(field_name="applicant_name", value="Rahul Verma"),
        Action(field_name="father_name", value="Suresh Verma"),
        Action(field_name="date_of_birth", value="1990-05-15"),
        Action(field_name="address", value="123 Gandhi Nagar New Delhi 110001"),
        Action(field_name="annual_income", value="150000"),
        Action(field_name="certificate_type", value="BPL"),
        Action(field_name="employment_type", value="Salaried"),
        Action(field_name="employer_name", value="Tata Consultancy Services"),
        Action(field_name="state", value="Delhi"),
        Action(field_name="mobile_number", value="8765432109"),
    ]
    for a in actions:
        env.step(a)
    score = IncomeGrader().grade(env.state())
    assert score == pytest.approx(1.0)


def test_passport_grader_full():
    env = GovFormEnv("passport_renewal")
    env.reset()
    actions = [
        Action(field_name="applicant_name", value="Ananya Patel"),
        Action(field_name="date_of_birth", value="1990-03-20"),
        Action(field_name="application_date", value="2026-04-08"),
        Action(field_name="existing_passport_number", value="J1234567"),
        Action(field_name="existing_passport_expiry", value="2025-12-31"),
        Action(field_name="place_of_birth", value="Mumbai"),
        Action(field_name="address", value="45 Marine Drive Mumbai Maharashtra 400001"),
        Action(field_name="pincode", value="400001"),
        Action(field_name="mobile_number", value="9988776655"),
        Action(field_name="email", value="ananya.patel@example.com"),
        Action(field_name="emergency_contact_name", value="Vikram Patel"),
        Action(field_name="emergency_contact_phone", value="9876543210"),
        Action(field_name="renewal_reason", value="Expiry"),
        Action(field_name="police_clearance", value="Yes"),
    ]
    for a in actions:
        env.step(a)
    score = PassportGrader().grade(env.state())
    assert score == pytest.approx(1.0)


# ── Reward recovery test ────────────────────────────────────────────────


def test_reward_recovery_signal():
    """INVALID → VALID should give +0.05 recovery reward."""
    env = GovFormEnv("aadhaar_update")
    env.reset()
    # First submit invalid
    env.step(Action(field_name="aadhaar_number", value="bad"))
    # Then fix it
    _, reward, _, _ = env.step(Action(field_name="aadhaar_number", value="123456789012"))
    assert reward == pytest.approx(0.05)


def test_noop_penalty():
    """Re-submitting same valid value should incur -0.03 penalty."""
    env = GovFormEnv("aadhaar_update")
    env.reset()
    env.step(Action(field_name="full_name", value="Ravi Kumar"))
    _, reward, _, _ = env.step(Action(field_name="full_name", value="Ravi Kumar"))
    assert reward == pytest.approx(-0.03)
