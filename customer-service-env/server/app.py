"""
FastAPI application for the Customer Service OpenEnv environment.

Provides REST endpoints for reset, step, state, tasks, grader, baseline,
and health.  Concurrent sessions are managed via a global dict keyed by
episode ID.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from models import (
    CustomerServiceAction,
    CustomerServiceObservation,
    CustomerServiceState,
    GraderResponse,
    TaskDefinition,
)
from server.environment import CustomerServiceEnvironment
from server.graders import GraderRegistry
from server.tasks import TaskRegistry

# ---------------------------------------------------------------------------
# FastAPI app instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Service OpenEnv",
    version="1.0.0",
    description=(
        "A real-world customer service environment where AI agents resolve "
        "support queries using multi-step tool-calling."
    ),
)


@app.get("/")
def root():
    """Landing page with environment info and available endpoints."""
    return {
        "name": "Customer Service OpenEnv",
        "version": "1.0.0",
        "description": "A real-world customer service environment where AI agents resolve support queries using multi-step tool-calling.",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "grader": "/grader",
            "baseline": "/baseline",
            "docs": "/docs",
        },
    }

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

# Maps session_id -> CustomerServiceEnvironment
_sessions: Dict[str, CustomerServiceEnvironment] = {}


def _get_env(session_id: str) -> CustomerServiceEnvironment:
    """Retrieve an environment by session ID or raise 400."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=400,
            detail=f"No active session '{session_id}'. Call /reset first.",
        )
    return env


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_id: str = Field(default="order_status_inquiry", description="Task identifier to start.")
    seed: int = Field(default=42, description="Random seed for reproducibility.")


class ResetResponse(BaseModel):
    """Response from POST /reset."""
    session_id: str
    observation: CustomerServiceObservation


class StepRequest(BaseModel):
    """Body for POST /step."""
    session_id: str = Field(..., description="Session ID from /reset.")
    tool_name: str = Field(..., description="Name of the tool to call.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")
    message: Optional[str] = Field(default=None, description="Optional agent message.")


class StepResponse(BaseModel):
    """Response from POST /step."""
    observation: CustomerServiceObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Response from GET /state."""
    state: CustomerServiceState


class GraderRequest(BaseModel):
    """Body for POST /grader."""
    session_id: str = Field(default="", description="Session ID of the episode to grade.")
    episode_id: str = Field(default="", description="Episode ID (alternative to session_id).")
    task_id: str = Field(default="", description="Task ID (required if trajectory is provided directly).")
    trajectory: List[Dict[str, Any]] = Field(default_factory=list, description="Action history.")


class TasksResponse(BaseModel):
    """Response from GET /tasks."""
    tasks: List[TaskDefinition]
    action_schema: Dict[str, str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset", response_model=ResetResponse)
def reset(body: Optional[ResetRequest] = None) -> ResetResponse:
    """Initialize a new episode and return the initial observation.

    Creates a fresh session with a unique ID.
    """
    if body is None:
        body = ResetRequest()
    env = CustomerServiceEnvironment()
    try:
        obs = env.reset(task_id=body.task_id, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    session_id = str(uuid.uuid4())
    _sessions[session_id] = env

    return ResetResponse(session_id=session_id, observation=obs)


@app.post("/step", response_model=StepResponse)
def step(body: StepRequest) -> StepResponse:
    """Advance the environment by one step."""
    env = _get_env(body.session_id)
    action = CustomerServiceAction(
        tool_name=body.tool_name,
        tool_args=body.tool_args,
        message=body.message,
    )
    obs, reward, done, info = env.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=StateResponse)
def get_state(session_id: str = Query(..., description="Session ID.")) -> StateResponse:
    """Return the full current episode state."""
    env = _get_env(session_id)
    return StateResponse(state=env.state)


@app.get("/tasks", response_model=TasksResponse)
def list_tasks() -> TasksResponse:
    """Return all available tasks and the action schema."""
    tasks = TaskRegistry.list_tasks()
    return TasksResponse(
        tasks=tasks,
        action_schema={"tool_name": "str", "tool_args": "dict"},
    )


@app.post("/grader", response_model=GraderResponse)
def grade(body: GraderRequest) -> GraderResponse:
    """Grade a completed episode trajectory."""
    # Determine task_id and trajectory from session or body
    task_id = body.task_id
    trajectory = body.trajectory

    if body.session_id and body.session_id in _sessions:
        env = _sessions[body.session_id]
        state = env.state
        if not task_id:
            task_id = state.task_id
        if not trajectory:
            trajectory = state.actions_taken
    else:
        state = CustomerServiceState(task_id=task_id)

    if not task_id:
        raise HTTPException(
            status_code=422,
            detail="task_id is required (either in body or via session_id).",
        )

    return GraderRegistry.grade(task_id, trajectory, state)


@app.get("/baseline")
def baseline_info() -> Dict[str, Any]:
    """Return metadata about the baseline agent."""
    return {
        "model": "gpt-4o-mini",
        "script": "baseline.py",
        "description": "OpenAI function-calling baseline that runs all 3 tasks.",
        "reference_scores": {
            "order_status_inquiry": 0.85,
            "return_refund_processing": 0.75,
            "complex_complaint_resolution": 0.65,
        },
        "average_score": 0.75,
    }


def main() -> None:
    """Entry point for the server."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
