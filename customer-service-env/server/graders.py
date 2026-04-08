"""
Post-hoc trajectory graders for all three customer service tasks.

Each grader examines the full action history and episode state and returns
a deterministic score in [0.0, 1.0] together with a per-dimension breakdown.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models import CustomerServiceState, GraderResponse


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _tools_called(actions: List[Dict[str, Any]]) -> List[str]:
    """Extract the ordered list of tool names from an action history."""
    return [a.get("tool_name", "") for a in actions]


def _has_tool(actions: List[Dict[str, Any]], tool_name: str) -> bool:
    """Return True if *tool_name* appears anywhere in the action history."""
    return tool_name in _tools_called(actions)


def _tool_result(actions: List[Dict[str, Any]], tool_name: str) -> Dict[str, Any]:
    """Return the result dict of the first occurrence of *tool_name*."""
    for a in actions:
        if a.get("tool_name") == tool_name:
            return a.get("tool_result", {})
    return {}


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.001, 0.999)."""
    return max(0.001, min(0.999, score))


def _efficiency_score(steps_used: int, max_steps: int, ideal_steps: int) -> float:
    """Score how efficiently the agent used its step budget.

    Returns 1.0 when *steps_used* == *ideal_steps* and degrades linearly
    toward 0.0 as *steps_used* approaches *max_steps*.
    """
    if steps_used <= ideal_steps:
        return 1.0
    if steps_used >= max_steps:
        return 0.0
    return max(0.0, 1.0 - (steps_used - ideal_steps) / (max_steps - ideal_steps))


# ---------------------------------------------------------------------------
# Task 1 grader -- Order Status Inquiry
# ---------------------------------------------------------------------------

def grade_task1(actions: List[Dict[str, Any]], state: CustomerServiceState) -> Dict[str, Any]:
    """Grade Task 1 trajectory.

    Dimensions:
      - tool_correctness (0.35): Did the agent call lookup_order and
        lookup_customer with correct arguments?
      - resolution (0.40): Did the agent send a notification with the
        correct order status?
      - efficiency (0.25): How many steps relative to the ideal (3)?
    """
    tool_correctness = 0.0
    resolution = 0.0

    # Check lookup_order called with correct order ID
    if _has_tool(actions, "lookup_order"):
        for a in actions:
            if a.get("tool_name") == "lookup_order":
                if a.get("tool_args", {}).get("order_id") == "ORD-1001":
                    tool_correctness += 0.5
                break

    # Check lookup_customer called with correct customer ID
    if _has_tool(actions, "lookup_customer"):
        for a in actions:
            if a.get("tool_name") == "lookup_customer":
                if a.get("tool_args", {}).get("customer_id") == "C001":
                    tool_correctness += 0.5
                break

    # Check send_notification called
    if _has_tool(actions, "send_notification"):
        for a in actions:
            if a.get("tool_name") == "send_notification":
                result = a.get("tool_result", {})
                if result.get("status") == "delivered":
                    resolution = 1.0
                break

    steps_used = len(actions)
    efficiency = _efficiency_score(steps_used, 5, 3)

    score = (
        tool_correctness * 0.35
        + resolution * 0.40
        + efficiency * 0.25
    )
    score = _clamp(score)

    return {
        "score": round(score, 4),
        "breakdown": {
            "tool_correctness": round(tool_correctness, 4),
            "resolution": round(resolution, 4),
            "efficiency": round(efficiency, 4),
        },
    }


# ---------------------------------------------------------------------------
# Task 2 grader -- Return & Refund Processing
# ---------------------------------------------------------------------------

def grade_task2(actions: List[Dict[str, Any]], state: CustomerServiceState) -> Dict[str, Any]:
    """Grade Task 2 trajectory.

    Dimensions:
      - policy_compliance (0.30): Did the agent check return policy before
        initiating a refund?
      - correct_outcome (0.40): Was the refund correctly initiated or
        denied?  Was the customer notified?
      - efficiency (0.30): Steps used vs ideal (5).
    """
    policy_compliance = 0.0
    correct_outcome = 0.0

    tools_in_order = _tools_called(actions)

    # Policy compliance: check_return_policy must come before initiate_refund
    if "check_return_policy" in tools_in_order:
        policy_compliance += 0.5
        if "initiate_refund" in tools_in_order:
            policy_idx = tools_in_order.index("check_return_policy")
            refund_idx = tools_in_order.index("initiate_refund")
            if policy_idx < refund_idx:
                policy_compliance += 0.5

    # Correct outcome: refund approved + notification
    if _has_tool(actions, "initiate_refund"):
        result = _tool_result(actions, "initiate_refund")
        if result.get("status") == "approved":
            correct_outcome += 0.5

    if _has_tool(actions, "send_notification"):
        result = _tool_result(actions, "send_notification")
        if result.get("status") == "delivered":
            correct_outcome += 0.5

    steps_used = len(actions)
    efficiency = _efficiency_score(steps_used, 10, 5)

    score = (
        policy_compliance * 0.30
        + correct_outcome * 0.40
        + efficiency * 0.30
    )
    score = _clamp(score)

    return {
        "score": round(score, 4),
        "breakdown": {
            "policy_compliance": round(policy_compliance, 4),
            "correct_outcome": round(correct_outcome, 4),
            "efficiency": round(efficiency, 4),
        },
    }


# ---------------------------------------------------------------------------
# Task 3 grader -- Complex Complaint Resolution
# ---------------------------------------------------------------------------

def grade_task3(actions: List[Dict[str, Any]], state: CustomerServiceState) -> Dict[str, Any]:
    """Grade Task 3 trajectory.

    Dimensions:
      - issue_identification (0.25): Did the agent look up the order and
        customer to discover both issues (wrong item + billing overcharge)?
      - resolution_quality (0.30): Was a refund initiated and compensation
        applied?
      - policy_compliance (0.25): Was return policy checked before refund?
        Was escalation used appropriately (if at all)?
      - efficiency (0.20): Steps used vs ideal (6).
    """
    issue_identification = 0.0
    resolution_quality = 0.0
    policy_compliance = 0.0

    # Issue identification
    if _has_tool(actions, "lookup_order"):
        issue_identification += 0.5
    if _has_tool(actions, "lookup_customer"):
        issue_identification += 0.5

    # Resolution quality
    if _has_tool(actions, "initiate_refund"):
        result = _tool_result(actions, "initiate_refund")
        if result.get("status") == "approved":
            resolution_quality += 0.4
    if _has_tool(actions, "apply_compensation"):
        result = _tool_result(actions, "apply_compensation")
        if result.get("status") == "applied":
            resolution_quality += 0.3
    if _has_tool(actions, "send_notification"):
        result = _tool_result(actions, "send_notification")
        if result.get("status") == "delivered":
            resolution_quality += 0.3

    # Policy compliance
    tools_in_order = _tools_called(actions)
    if "check_return_policy" in tools_in_order:
        policy_compliance += 0.5
        if "initiate_refund" in tools_in_order:
            p_idx = tools_in_order.index("check_return_policy")
            r_idx = tools_in_order.index("initiate_refund")
            if p_idx < r_idx:
                policy_compliance += 0.5

    # Penalise unnecessary escalation (escalation is optional on task 3
    # but should only be used if matching a conflict condition)
    if _has_tool(actions, "escalate_to_human"):
        esc_result = _tool_result(actions, "escalate_to_human")
        reason = ""
        for a in actions:
            if a.get("tool_name") == "escalate_to_human":
                reason = a.get("tool_args", {}).get("reason", "")
                break
        valid_reasons = {"wrong_item_delivered", "billing_overcharge"}
        if not any(vr in reason.lower().replace(" ", "_") for vr in valid_reasons):
            # Invalid escalation reason -- penalty
            policy_compliance = max(0.0, policy_compliance - 0.3)

    steps_used = len(actions)
    efficiency = _efficiency_score(steps_used, 15, 6)

    score = (
        issue_identification * 0.25
        + resolution_quality * 0.30
        + policy_compliance * 0.25
        + efficiency * 0.20
    )
    score = _clamp(score)

    return {
        "score": round(score, 4),
        "breakdown": {
            "issue_identification": round(issue_identification, 4),
            "resolution_quality": round(resolution_quality, 4),
            "policy_compliance": round(policy_compliance, 4),
            "efficiency": round(efficiency, 4),
        },
    }


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

_GRADERS = {
    "order_status_inquiry": grade_task1,
    "return_refund_processing": grade_task2,
    "complex_complaint_resolution": grade_task3,
}


class GraderRegistry:
    """Dispatches grading to the appropriate per-task grader."""

    @staticmethod
    def grade(
        task_id: str,
        trajectory: List[Dict[str, Any]],
        state: CustomerServiceState,
    ) -> GraderResponse:
        """Grade a completed episode trajectory.

        Returns a :class:`GraderResponse` with score 0.0 and an error
        explanation when the trajectory is empty or the task ID is unknown.
        """
        if task_id not in _GRADERS:
            return GraderResponse(
                task_id=task_id,
                score=0.001,
                breakdown={},
                explanation=f"Unknown task_id '{task_id}'.",
            )

        if not trajectory:
            return GraderResponse(
                task_id=task_id,
                score=0.001,
                breakdown={},
                explanation="Empty trajectory -- nothing to grade.",
            )

        grader_fn = _GRADERS[task_id]
        result = grader_fn(trajectory, state)

        return GraderResponse(
            task_id=task_id,
            score=result["score"],
            breakdown=result["breakdown"],
            explanation="Grading completed successfully.",
        )
