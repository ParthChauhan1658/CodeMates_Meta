"""
Pydantic v2 data models for the Customer Service OpenEnv environment.

All base classes are defined inline for portability (no openenv-core dependency).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base classes (compatible with the openenv-core interface)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Base action submitted by an agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Observation(BaseModel):
    """Base observation returned to the agent."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    done: bool = False
    reward: Optional[float] = None


class State(BaseModel):
    """Base state snapshot of the environment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    episode_id: Optional[str] = None
    step_count: int = 0


# ---------------------------------------------------------------------------
# Customer Service specific models
# ---------------------------------------------------------------------------

class CustomerServiceAction(Action):
    """Action that an agent sends to the environment each step.

    Attributes:
        tool_name: Name of the tool to invoke (e.g. ``lookup_order``).
        tool_args: Keyword arguments forwarded to the tool function.
        message: Optional free-text message from the agent (for logging).
    """
    tool_name: str = Field(..., description="Name of the tool to call")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    message: Optional[str] = Field(default=None, description="Optional natural-language context from the agent")


class CustomerServiceObservation(Observation):
    """Observation returned after each environment step.

    Attributes:
        task_id: Identifier of the active task.
        customer_message: The original customer query text.
        tool_result: Structured result from the last tool call, if any.
        available_tools: Names of tools the agent may call.
        steps_remaining: How many steps the agent has left.
        reward_components: Labelled reward breakdown for this step.
        message: Human-readable summary of the step outcome.
    """
    task_id: str = ""
    customer_message: str = ""
    tool_result: Optional[Dict[str, Any]] = None
    available_tools: List[str] = Field(default_factory=list)
    steps_remaining: int = 0
    reward_components: Dict[str, float] = Field(default_factory=dict)
    message: str = ""


class CustomerServiceState(State):
    """Full episode state snapshot.

    Attributes:
        task_id: Active task identifier.
        customer_id: Customer associated with the episode.
        order_id: Primary order associated with the episode.
        tools_called: Ordered list of tool names invoked so far.
        actions_taken: Full action records with tool name, args, and result.
        resolved: Whether the customer issue has been fully resolved.
        escalated: Whether the episode was escalated to a human agent.
        cumulative_reward: Running total of reward accumulated.
        max_steps: Maximum steps allowed for this task.
        resolution_status: Current resolution status of the episode.
    """
    task_id: str = ""
    customer_id: str = ""
    order_id: str = ""
    tools_called: List[str] = Field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    resolved: bool = False
    escalated: bool = False
    cumulative_reward: float = 0.0
    max_steps: int = 0
    resolution_status: Literal[
        "in_progress", "resolved", "escalated", "step_limit_exceeded"
    ] = "in_progress"


# ---------------------------------------------------------------------------
# Supplementary models
# ---------------------------------------------------------------------------

class TaskDefinition(BaseModel):
    """Static metadata describing a single task.

    Attributes:
        id: Unique task identifier (e.g. ``order_status_inquiry``).
        name: Human-readable task name.
        difficulty: One of easy / medium / hard.
        description: Longer description of the scenario.
        max_steps: Maximum steps an agent is allowed.
        required_tools: Ordered list of tools an ideal agent would call.
        scoring_criteria: Mapping of grader dimension name to its weight.
    """
    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str = ""
    max_steps: int = 5
    required_tools: List[str] = Field(default_factory=list)
    scoring_criteria: Dict[str, float] = Field(default_factory=dict)


class RewardSignal(BaseModel):
    """Internal reward signal returned by the reward calculator.

    Attributes:
        step_reward: Delta reward for this single step.
        cumulative_reward: Running total (clamped to [0, 1] at episode end).
        components: Labelled breakdown of reward contributions.
    """
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    components: Dict[str, float] = Field(default_factory=dict)


class GraderRequest(BaseModel):
    """Request payload for the ``/grader`` endpoint.

    Attributes:
        episode_id: The episode to grade.
        task_id: Which task was played.
        trajectory: List of action records for the episode.
    """
    episode_id: str = ""
    task_id: str = ""
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)


class GraderResponse(BaseModel):
    """Response from the grader.

    Attributes:
        task_id: Which task was graded.
        score: Aggregate score in [0.0, 1.0].
        breakdown: Per-dimension scores.
        explanation: Human-readable explanation of the grading.
    """
    task_id: str = ""
    score: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
    explanation: str = ""


class StepResult(BaseModel):
    """Convenience wrapper for the tuple returned by ``environment.step()``.

    Attributes:
        observation: The observation after the step.
        reward: Scalar reward delta for this step.
        done: Whether the episode has ended.
        info: Additional metadata including reward components.
    """
    observation: CustomerServiceObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
