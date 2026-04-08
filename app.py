"""FastAPI server for the GovForm OpenEnv environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from govform_env.environment import GovFormEnv
from govform_env.models import Action, Observation

# ── App setup ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="GovForm OpenEnv",
    description="An AI agent fills Indian government forms — Aadhaar update, income certificate, passport renewal.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global stateful env instance ──────────────────────────────────────────

_env: GovFormEnv | None = None

# ── Request / response schemas ────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class StepResponse(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: Dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    difficulty: str
    max_steps: int
    description: str


# ── Available tasks ──────────────────────────────────────────────────────

TASKS: List[TaskInfo] = [
    TaskInfo(
        id="aadhaar_update",
        difficulty="easy",
        max_steps=20,
        description="Fill a 6-field Aadhaar address update form.",
    ),
    TaskInfo(
        id="income_certificate",
        difficulty="medium",
        max_steps=20,
        description="Fill a 10-field income certificate with cross-field validation rules.",
    ),
    TaskInfo(
        id="passport_renewal",
        difficulty="hard",
        max_steps=25,
        description="Fill a 14-field passport renewal form and resolve 2 pre-seeded conflicts.",
    ),
]


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {
        "message": "GovForm OpenEnv is Running!",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/health"],
        "docs": "/docs"
    }


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks():
    return TASKS


@app.post("/reset")
def reset(req: ResetRequest):
    global _env
    task_id = req.task_id or TASKS[0].id
    valid_ids = {t.id for t in TASKS}
    if task_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id '{task_id}'. Choose from: {sorted(valid_ids)}",
        )
    _env = GovFormEnv(task_id=task_id)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def get_state():
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return _env.state()
