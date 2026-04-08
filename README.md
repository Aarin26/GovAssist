---
title: Govform Assist
sdk: docker
emoji: 🏛️
colorFrom: blue
colorTo: indigo
app_port: 7860
---

# GovForm Assistant — OpenEnv Environment

An AI agent fills Indian government forms correctly by interpreting free-text user inputs, mapping them to structured form fields, validating each field, detecting conflicts or missing data, and guiding the user toward a complete, valid submission.

---

## 1. Environment Description & Motivation

Filling bureaucratic government forms is a task millions of people perform daily.  
This environment models three real Indian government forms with increasing complexity:

| Task | Form | Fields | Difficulty | Key Challenge |
|------|------|--------|------------|---------------|
| 1 | Aadhaar Address Update | 6 | Easy | Straightforward regex validation |
| 2 | Income Certificate | 10 | Medium | Cross-field consistency rules |
| 3 | Passport Renewal | 14 | Hard | Pre-seeded conflicts to detect and resolve |

The agent must fill every required field with a valid value. The environment returns granular feedback after each step, enabling iterative improvement.

---

## 2. Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `form_id` | `str` | Identifier of the current form |
| `task_id` | `str` | Task identifier (e.g. `aadhaar_update`) |
| `fields` | `List[FormField]` | Current state of every field (name, label, value, status, error) |
| `last_agent_action` | `str \| null` | String representation of the last action taken |
| `last_system_message` | `str` | Human-readable feedback to the agent |
| `filled_count` | `int` | Number of fields with a value |
| `valid_count` | `int` | Number of required fields with `VALID` status |
| `total_required` | `int` | Total number of required fields |
| `step_number` | `int` | Current step in the episode |

Each `FormField` contains:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Machine-readable field name |
| `label` | `str` | Human-readable label |
| `value` | `str \| null` | Current value |
| `status` | `FieldStatus` | One of: `empty`, `filled`, `invalid`, `valid` |
| `error_message` | `str \| null` | Validation error, if any |
| `required` | `bool` | Whether this field is required |

---

## 3. Action Space

| Field | Type | Description |
|-------|------|-------------|
| `field_name` | `str` | Which field to fill |
| `value` | `str` | Proposed value (raw string) |
| `reasoning` | `str \| null` | Agent's explanation (logged, not graded) |

---

## 4. Task Descriptions

### Task 1: Aadhaar Address Update (Easy)
- **Fields:** 6 (full name, Aadhaar number, address, PIN code, state, mobile)
- **Validation:** Simple regex and enum checks
- **Grading:** `score = valid_fields / total_required_fields`
- **Expected baseline:** ~1.0 (straightforward for capable LLMs)

### Task 2: Income Certificate (Medium)
- **Fields:** 10 (applicant info, income, employment, certificate type)
- **Validation:** Regex + 3 cross-field rules:
  - BPL certificate → income < ₹3,00,000
  - EWS certificate → income < ₹8,00,000
  - Salaried employment → employer name required
- **Grading:** `(field_score × 0.6) + (cross_field × 0.4)`
- **Expected baseline:** ~0.7–0.9

### Task 3: Passport Renewal (Hard)
- **Fields:** 14 (passport details, personal info, emergency contact)
- **Validation:** Regex + 3 cross-field conflict rules:
  - Applicant must be ≥18 years old
  - Emergency contact ≠ applicant name
  - Passport expiry within 3 years of application date
- **Grading:** `(field_score × 0.5) + (conflict × 0.3) + (completeness × 0.2)`
- **Expected baseline:** ~0.6–0.8

---

## 5. Reward Function Breakdown

| Signal | Reward |
|--------|--------|
| Field transition EMPTY → VALID | +0.10 |
| Field transition INVALID → VALID (recovery) | +0.05 |
| Field transition VALID → INVALID (regression) | −0.02 |
| Non-existent field name | −0.05 |
| Re-submitting a VALID field with same value | −0.03 |
| Episode complete (all required fields valid) | +0.20 |

**Normalised episode score:**  
`score = total_reward / max_possible_reward`  
where `max_possible_reward = (0.10 × n_required_fields) + 0.20`

---

## 6. Setup

### Local (uvicorn)

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t govform-openenv .
docker run -p 7860:7860 govform-openenv
```

### Hugging Face Space

Push this repo to a HF Space with the `Dockerfile` runtime. The server will start on port 7860 automatically.

---

## 7. Running Inference

```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Start the server in one terminal
uvicorn app:app --port 7860

# Run inference in another terminal
python inference.py
```

The script will:
1. Reset each task sequentially
2. Loop up to `max_steps` per task, calling the LLM for each action
3. Grade the final state and print `[START]` / `[STEP]` / `[END]` logs

---

## 8. Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| aadhaar_update | Qwen2.5-72B-Instruct | **1.0000** |
| income_certificate | Qwen2.5-72B-Instruct | **1.0000** |
| passport_renewal | Qwen2.5-72B-Instruct | **1.0000** |
| **Average** | | **1.0000** |

---

## API Reference

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `GET` | `/health` | — | `{"status": "ok"}` |
| `GET` | `/tasks` | — | List of task descriptions |
| `POST` | `/reset` | `{"task_id": "..."}` | Observation JSON |
| `POST` | `/step` | `{"field_name": "...", "value": "...", "reasoning": "..."}` | `{observation, reward, done, info}` |
| `GET` | `/state` | — | Full state snapshot |
