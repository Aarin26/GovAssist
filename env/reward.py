"""Reward function with partial signals for the GovForm environment."""

from __future__ import annotations

from env.models import Action, FieldStatus, Observation


def compute_reward(
    prev_obs: Observation,
    action: Action,
    new_obs: Observation,
    done: bool,
) -> float:
    """
    Compute the step reward.

    Reward breakdown (sums to 1.0 at perfect completion):
      +0.10  for each field transition  EMPTY  → VALID
      +0.05  for each field transition  INVALID → VALID  (recovery)
      -0.02  for each field transition  VALID  → INVALID (regression)
      -0.05  for attempting to fill a non-existent field_name
      -0.03  for re-submitting an already-VALID field with same value
      +0.20  bonus when done=True (all required fields VALID)
    """
    reward = 0.0

    # Build lookup of previous field statuses & values
    prev_status: dict[str, FieldStatus] = {}
    prev_values: dict[str, str | None] = {}
    for f in prev_obs.fields:
        prev_status[f.name] = f.status
        prev_values[f.name] = f.value

    # Build lookup of new field statuses & values
    new_status: dict[str, FieldStatus] = {}
    new_values: dict[str, str | None] = {}
    for f in new_obs.fields:
        new_status[f.name] = f.status
        new_values[f.name] = f.value

    target_field = action.field_name

    # Penalty: non-existent field
    if target_field not in new_status:
        reward -= 0.05
        if done:
            reward += 0.20
        return reward

    # Penalty: re-submitting an already-VALID field with the same value
    if (
        prev_status.get(target_field) == FieldStatus.VALID
        and prev_values.get(target_field) == action.value
    ):
        reward -= 0.03
        if done:
            reward += 0.20
        return reward

    # Transition rewards (only for the targeted field)
    old = prev_status.get(target_field, FieldStatus.EMPTY)
    new = new_status.get(target_field, FieldStatus.EMPTY)

    if old in (FieldStatus.EMPTY, FieldStatus.FILLED) and new == FieldStatus.VALID:
        reward += 0.10
    elif old == FieldStatus.INVALID and new == FieldStatus.VALID:
        reward += 0.05
    elif old == FieldStatus.VALID and new == FieldStatus.INVALID:
        reward -= 0.02

    # Done bonus
    if done:
        reward += 0.20

    return reward


def max_possible_reward(n_required_fields: int) -> float:
    """Maximum possible episode reward for score normalisation."""
    return (0.10 * n_required_fields) + 0.20


def normalise_score(total_reward: float, n_required_fields: int) -> float:
    """Map cumulative reward to [0.0, 1.0]."""
    max_r = max_possible_reward(n_required_fields)
    if max_r <= 0:
        return 0.0
    return max(0.0, min(1.0, total_reward / max_r))
