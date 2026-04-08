"""Strict OpenEnv inference script following mandatory STDOUT formatting."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, List, Optional

import httpx
from openai import OpenAI

from env.graders.task1_aadhaar import AadhaarGrader
from env.graders.task2_income import IncomeGrader
from env.graders.task3_passport import PassportGrader
from env.reward import max_possible_reward

# ── Environment & Config ──────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN # Alias for consistency with some clients

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
BENCHMARK = "govform-openenv"

GRADERS = {
    "aadhaar_update": AadhaarGrader(),
    "income_certificate": IncomeGrader(),
    "passport_renewal": PassportGrader(),
}

MAX_STEPS_MAP = {
    "aadhaar_update": 20,
    "income_certificate": 20,
    "passport_renewal": 25
}

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Indian government form-filling assistant.
    You will receive the current state of a form with each field's status.

    RULES:
    1. ONLY fill fields whose status is "empty" or "invalid". NEVER re-submit a field that is already "valid".
    2. Pick the FIRST field in the list that is "empty" or "invalid".
    3. If a field is "invalid", read its error message carefully and provide a corrected value.
    4. Use realistic Indian data. For names use Indian names (letters and spaces only).
    5. For dates use YYYY-MM-DD. Aadhaar: 12 digits. Mobile: 10 digits starting 6-9.
    6. For enum fields, use EXACTLY one of the allowed values listed in the error message.
    7. Return ONLY JSON: {"field_name": "...", "value": "...", "reasoning": "..."}
""")

# ── Logging Helpers (MANDATORY FORMAT) ───────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Internal Helpers ──────────────────────────────────────────────────────


def _post(path: str, body: dict) -> dict:
    r = httpx.post(f"{SERVER_URL}{path}", json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _get(path: str) -> dict:
    r = httpx.get(f"{SERVER_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def _build_user_prompt(obs: dict) -> str:
    needs_attention = []
    completed = []

    for f in obs.get("fields", []):
        status = f["status"]
        val = f.get("value") or "(empty)"
        err = f.get("error_message") or ""
        req = "REQUIRED" if f.get("required", True) else "optional"
        label = f.get("label", f["name"])

        if status in ("empty", "invalid"):
            line = f"  ❌ {f['name']} (label: {label}) [{status}, {req}]: {val}"
            if err: line += f"  ⚠ ERROR: {err}"
            needs_attention.append(line)
        else:
            completed.append(f"  ✅ {f['name']}: {val} [VALID]")

    lines = ["=== FIELDS THAT NEED ACTION ==="] + needs_attention + ["\n=== COMPLETED ==="] + completed
    lines.append(f"\nProgress: {obs['valid_count']}/{obs['total_required']} | Step: {obs['step_number']}")
    return "\n".join(lines)


def _parse_action(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")]
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        if "field_name" in obj and "value" in obj:
            return obj
    except:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(raw[start:end+1])
                if "field_name" in obj and "value" in obj: return obj
            except: pass
    return None

# ── Main Task Runner ───────────────────────────────────────────────────────


def run_task(task_id: str, client: OpenAI) -> None:
    max_steps = MAX_STEPS_MAP.get(task_id, 20)
    grader = GRADERS[task_id]
    
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    score = 0.0
    action_str = "null"
    last_error = None
    
    try:
        obs_data = _post("/reset", {"task_id": task_id})
        done = False
        
        for step_num in range(1, max_steps + 1):
            if done: break
            
            user_prompt = _build_user_prompt(obs_data)
            
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                raw_text = response.choices[0].message.content or ""
            except Exception as exc:
                last_error = str(exc)
                break

            action = _parse_action(raw_text)
            if action is None:
                last_error = "json_parse_failed"
                action_str = "invalid_json"
                # Emit a dummy step log for visibility of failure
                log_step(step=step_num, action=action_str, reward=0.0, done=False, error=last_error)
                continue

            action_str = f"{action['field_name']}='{action['value']}'"
            
            result = _post("/step", {
                "field_name": action["field_name"],
                "value": str(action["value"]),
                "reasoning": action.get("reasoning"),
            })

            obs_data = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            
            rewards.append(reward)
            steps_taken = step_num
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=None)

            if done: break

        # Final Scoring
        final_state = _get("/state")
        score = float(grader.grade(final_state))
        success = score >= 0.99  # Consider successful if perfect or near perfect

    except Exception as e:
        last_error = str(e)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in ["aadhaar_update", "income_certificate", "passport_renewal"]:
        try:
            run_task(task_id, client)
        except Exception as exc:
            print(f"[DEBUG] Global failure for {task_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
