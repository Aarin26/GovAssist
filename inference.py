"""
Inference Script — GovForm OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

from govform_env import GovFormAction, GovFormEnv
from govform_env.graders.task1_aadhaar import AadhaarGrader
from govform_env.graders.task2_income import IncomeGrader
from govform_env.graders.task3_passport import PassportGrader

# ── Environment & Config ──────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:7860")
BENCHMARK = "govform-openenv"

TASKS = ["aadhaar_update", "income_certificate", "passport_renewal"]

GRADERS = {
    "aadhaar_update": AadhaarGrader(),
    "income_certificate": IncomeGrader(),
    "passport_renewal": PassportGrader(),
}

MAX_STEPS_MAP = {
    "aadhaar_update": 20,
    "income_certificate": 20,
    "passport_renewal": 25,
}

TEMPERATURE = 0.0
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.5

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

# ── Logging Helpers (MANDATORY FORMAT) ────────────────────────────────────


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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompt Building ──────────────────────────────────────────────────────


def build_user_prompt(obs: dict) -> str:
    needs_attention = []
    completed = []

    for f in obs.get("fields", []):
        status = f["status"]
        val = f.get("value") or "(empty)"
        err = f.get("error_message") or ""
        req = "REQUIRED" if f.get("required", True) else "optional"
        label = f.get("label", f["name"])

        if status in ("empty", "invalid"):
            line = f"  - {f['name']} (label: {label}) [{status}, {req}]: {val}"
            if err:
                line += f"  ERROR: {err}"
            needs_attention.append(line)
        else:
            completed.append(f"  + {f['name']}: {val} [VALID]")

    lines = ["=== FIELDS THAT NEED ACTION ==="] + needs_attention
    lines += ["\n=== COMPLETED ==="] + completed
    lines.append(f"\nProgress: {obs['valid_count']}/{obs['total_required']} | Step: {obs['step_number']}")
    return "\n".join(lines)


# ── LLM Interaction ──────────────────────────────────────────────────────


def get_model_response(client: OpenAI, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"field_name": "unknown", "value": "hello"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"field_name": "unknown", "value": "hello"}'


def parse_action(raw: str) -> Optional[dict]:
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
    except Exception:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(raw[start : end + 1])
                if "field_name" in obj and "value" in obj:
                    return obj
            except Exception:
                pass
    return None


# ── Main Task Runner (Async) ─────────────────────────────────────────────


async def run_task(task_id: str, env: GovFormEnv, llm_client: OpenAI) -> None:
    max_steps = MAX_STEPS_MAP.get(task_id, 20)
    grader = GRADERS[task_id]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        result = await env.reset(task_id=task_id)
        obs_data = result.observation
        done = result.done

        for step_num in range(1, max_steps + 1):
            if done:
                break

            user_prompt = build_user_prompt(obs_data)
            raw_text = get_model_response(llm_client, user_prompt)
            action_dict = parse_action(raw_text)

            if action_dict is None:
                log_step(step=step_num, action="invalid_json", reward=0.0, done=False, error="json_parse_failed")
                continue

            action = GovFormAction(
                field_name=action_dict["field_name"],
                value=str(action_dict["value"]),
                reasoning=action_dict.get("reasoning"),
            )

            action_str = f"{action.field_name}='{action.value}'"

            step_result = await env.step(action)

            obs_data = step_result.observation
            reward = step_result.reward
            done = step_result.done
            error = step_result.last_action_error

            rewards.append(reward)
            steps_taken = step_num

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Final Scoring via Grader
        final_state = await env.get_state()
        score = float(grader.grade(final_state))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", file=sys.stderr)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry Point ──────────────────────────────────────────────────────────


async def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Choose connection mode: Docker image or direct server URL
    if IMAGE_NAME:
        env = await GovFormEnv.from_docker_image(IMAGE_NAME)
    else:
        env = await GovFormEnv.from_server_url(SERVER_URL)

    try:
        for task_id in TASKS:
            await run_task(task_id, env, llm_client)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
