"""
Core environment for the Customer Service OpenEnv.

Orchestrates episode lifecycle (reset / step / state), delegates tool
execution to the ToolRegistry, computes rewards per step, and enforces
task step limits.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import (
    CustomerServiceAction,
    CustomerServiceObservation,
    CustomerServiceState,
    RewardSignal,
)
from server.fixtures import load_fixtures
from server.tasks import TaskRegistry
from server.tools import TOOLS, ToolRegistry


# ---------------------------------------------------------------------------
# Internal mutable episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """Mutable state for a single episode."""

    episode_id: str = ""
    task_id: str = ""
    step_count: int = 0
    max_steps: int = 5
    done: bool = False
    resolution_status: str = "in_progress"
    cumulative_reward: float = 0.0

    customer_id: str = ""
    order_id: str = ""

    fixtures: Dict[str, Any] = field(default_factory=dict)
    customer_message: str = ""
    conflict_conditions: List[str] = field(default_factory=list)

    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    tools_called: List[str] = field(default_factory=list)

    # Tracking flags for reward computation
    refund_eligible: Optional[bool] = None
    policy_checked: bool = False


# ---------------------------------------------------------------------------
# Reward computation helpers
# ---------------------------------------------------------------------------

# Per-task reward tables.  Maps (task_id, tool_name) -> base reward.
_REWARD_TABLE: Dict[str, Dict[str, float]] = {
    "order_status_inquiry": {
        "lookup_order": 0.30,
        "lookup_customer": 0.30,
        "send_notification": 0.40,
    },
    "return_refund_processing": {
        "lookup_order": 0.20,
        "lookup_customer": 0.10,
        "check_return_policy": 0.20,
        "initiate_refund": 0.20,
        "send_notification": 0.30,
    },
    "complex_complaint_resolution": {
        "lookup_order": 0.10,
        "lookup_customer": 0.10,
        "check_return_policy": 0.15,
        "initiate_refund": 0.15,
        "apply_compensation": 0.20,
        "send_notification": 0.15,
    },
}

# Tools that finish the episode when called correctly per task
_TERMINAL_TOOLS: Dict[str, str] = {
    "order_status_inquiry": "send_notification",
    "return_refund_processing": "send_notification",
    "complex_complaint_resolution": "send_notification",
}


def _compute_step_reward(
    episode: EpisodeState,
    tool_name: str,
    tool_result: Dict[str, Any],
) -> RewardSignal:
    """Compute the reward delta for a single step.

    Applies positive rewards for correct tool calls and negative penalties
    for validation errors and policy violations.
    """
    components: Dict[str, float] = {}
    step_reward = 0.0
    task_id = episode.task_id

    result_status = tool_result.get("status", "")

    # --- Penalty: tool validation / unknown tool errors ---
    if result_status == "error":
        error_type = tool_result.get("error_type", "")
        if error_type == "validation_error":
            step_reward -= 0.10
            components["wrong_tool_args"] = -0.10
        # tool_error (e.g. not found) is not penalised -- agent may retry
        episode.cumulative_reward += step_reward
        return RewardSignal(
            step_reward=step_reward,
            cumulative_reward=episode.cumulative_reward,
            components=components,
        )

    # --- Positive reward: correct tool call ---
    task_rewards = _REWARD_TABLE.get(task_id, {})
    if tool_name in task_rewards and tool_name not in episode.tools_called:
        base = task_rewards[tool_name]
        step_reward += base
        components[f"correct_{tool_name}"] = base

    # --- Penalty: unnecessary escalation on task 1 or 2 ---
    if tool_name == "escalate_to_human" and task_id in (
        "order_status_inquiry",
        "return_refund_processing",
    ):
        step_reward -= 0.30
        components["unnecessary_escalation"] = -0.30

    # --- Penalty: policy violation (refund without policy check) ---
    if tool_name == "initiate_refund" and not episode.policy_checked:
        step_reward -= 0.20
        components["policy_violation"] = -0.20

    episode.cumulative_reward += step_reward
    return RewardSignal(
        step_reward=step_reward,
        cumulative_reward=episode.cumulative_reward,
        components=components,
    )


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class CustomerServiceEnvironment:
    """OpenEnv-compliant customer service environment.

    Supports concurrent sessions: each call to ``reset`` creates a fresh
    ``EpisodeState`` identified by a UUID episode ID.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        self._episode: Optional[EpisodeState] = None

    # ----- reset ----------------------------------------------------------

    def reset(
        self,
        task_id: str,
        seed: int = 42,  # noqa: ARG002  (kept for OpenEnv compat)
    ) -> CustomerServiceObservation:
        """Start a new episode for the given *task_id*.

        Returns the initial observation containing the customer message
        and available tools.
        """
        task_def = TaskRegistry.get_task(task_id)  # raises ValueError
        fixtures = TaskRegistry.build_fixtures(task_id)

        # Determine the primary customer / order IDs from fixture data
        customer_id = next(iter(fixtures["customers"]), "")
        order_id = next(iter(fixtures["orders"]), "")

        self._episode = EpisodeState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            max_steps=task_def.max_steps,
            fixtures=fixtures,
            customer_message=fixtures["customer_message"],
            customer_id=customer_id,
            order_id=order_id,
            conflict_conditions=fixtures.get("conflict_conditions", []),
        )

        return CustomerServiceObservation(
            done=False,
            reward=0.0,
            task_id=task_id,
            customer_message=fixtures["customer_message"],
            tool_result=None,
            available_tools=sorted(TOOLS.keys()),
            steps_remaining=task_def.max_steps,
            reward_components={},
            message=fixtures["customer_message"],
        )

    # ----- step -----------------------------------------------------------

    def step(
        self,
        action: CustomerServiceAction,
    ) -> Tuple[CustomerServiceObservation, float, bool, Dict[str, Any]]:
        """Advance the episode by one step.

        Returns ``(observation, reward, done, info)``.
        """
        ep = self._episode
        if ep is None:
            raise RuntimeError("No active episode. Call reset() first.")

        # If already done, return the terminal observation again
        if ep.done:
            obs = self._make_observation(
                tool_result=None,
                reward_signal=RewardSignal(
                    cumulative_reward=ep.cumulative_reward,
                ),
                message="Episode already finished.",
            )
            return obs, 0.0, True, {"reward_components": {}}

        # Dispatch the tool call
        tool_name = action.tool_name
        tool_args = action.tool_args
        tool_result = ToolRegistry.dispatch(tool_name, tool_args, ep.fixtures)

        # Track policy checks
        if tool_name == "check_return_policy" and tool_result.get("status") == "success":
            ep.policy_checked = True
            ep.refund_eligible = tool_result.get("eligible", False)

        # Handle escalation
        if tool_name == "escalate_to_human" and tool_result.get("status") == "escalated":
            ep.resolution_status = "escalated"

        # Compute reward
        reward_signal = _compute_step_reward(ep, tool_name, tool_result)

        # Record action
        action_record = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result,
            "step": ep.step_count,
        }
        ep.actions_taken.append(action_record)
        if tool_result.get("status") != "error":
            ep.tools_called.append(tool_name)

        # Increment step count
        ep.step_count += 1

        # Check terminal conditions
        done = False

        # 1. Task completed (terminal tool called successfully)
        terminal_tool = _TERMINAL_TOOLS.get(ep.task_id)
        if (
            tool_name == terminal_tool
            and tool_result.get("status") == "delivered"
        ):
            ep.done = True
            ep.resolution_status = "resolved"
            done = True

        # 2. Escalation also terminates
        if tool_name == "escalate_to_human" and tool_result.get("status") == "escalated":
            ep.done = True
            done = True

        # 3. Step limit reached
        if ep.step_count >= ep.max_steps and not ep.done:
            ep.done = True
            ep.resolution_status = "step_limit_exceeded"
            done = True

        # Clamp cumulative reward at episode end to the open interval (0, 1)
        # so downstream validators never see boundary scores.
        if ep.done:
            ep.cumulative_reward = max(0.001, min(0.999, ep.cumulative_reward))
            reward_signal.cumulative_reward = ep.cumulative_reward

        message = (
            tool_result.get("message", "")
            if isinstance(tool_result.get("message"), str)
            else str(tool_result)
        )

        obs = self._make_observation(
            tool_result=tool_result,
            reward_signal=reward_signal,
            message=message,
        )

        info: Dict[str, Any] = {
            "reward_components": reward_signal.components,
            "cumulative_reward": reward_signal.cumulative_reward,
        }

        return obs, reward_signal.step_reward, done, info

    # ----- state ----------------------------------------------------------

    @property
    def state(self) -> CustomerServiceState:
        """Return the full current episode state as a Pydantic model."""
        ep = self._episode
        if ep is None:
            return CustomerServiceState()
        return CustomerServiceState(
            episode_id=ep.episode_id,
            step_count=ep.step_count,
            task_id=ep.task_id,
            customer_id=ep.customer_id,
            order_id=ep.order_id,
            tools_called=list(ep.tools_called),
            actions_taken=list(ep.actions_taken),
            resolved=ep.resolution_status == "resolved",
            escalated=ep.resolution_status == "escalated",
            cumulative_reward=ep.cumulative_reward,
            max_steps=ep.max_steps,
            resolution_status=ep.resolution_status,  # type: ignore[arg-type]
        )

    # ----- internal helpers -----------------------------------------------

    def _make_observation(
        self,
        tool_result: Optional[Dict[str, Any]],
        reward_signal: RewardSignal,
        message: str = "",
    ) -> CustomerServiceObservation:
        """Build an observation from current episode state."""
        ep = self._episode
        assert ep is not None
        return CustomerServiceObservation(
            done=ep.done,
            reward=reward_signal.step_reward,
            task_id=ep.task_id,
            customer_message=ep.customer_message,
            tool_result=tool_result,
            available_tools=sorted(TOOLS.keys()),
            steps_remaining=max(0, ep.max_steps - ep.step_count),
            reward_components=reward_signal.components,
            message=message,
        )
