"""
Task registry for the Customer Service OpenEnv environment.

Defines the three tasks and provides helpers for looking up definitions,
building fixtures, and generating initial observation text.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models import TaskDefinition
from server.fixtures import load_fixtures


# ---------------------------------------------------------------------------
# Static task definitions
# ---------------------------------------------------------------------------

_TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "order_status_inquiry": TaskDefinition(
        id="order_status_inquiry",
        name="Order Status Inquiry",
        difficulty="easy",
        description=(
            "The customer wants to know the current status of their recent "
            "order. The agent must look up the order, retrieve the status, "
            "and notify the customer with the correct information."
        ),
        max_steps=5,
        required_tools=["lookup_order", "lookup_customer", "send_notification"],
        scoring_criteria={
            "tool_correctness": 0.35,
            "resolution": 0.40,
            "efficiency": 0.25,
        },
    ),
    "return_refund_processing": TaskDefinition(
        id="return_refund_processing",
        name="Return & Refund Processing",
        difficulty="medium",
        description=(
            "The customer received a defective product and wants to return it "
            "for a refund. The agent must verify the order, check the return "
            "policy, initiate the refund if eligible, and notify the customer."
        ),
        max_steps=10,
        required_tools=[
            "lookup_order",
            "lookup_customer",
            "check_return_policy",
            "initiate_refund",
            "send_notification",
        ],
        scoring_criteria={
            "policy_compliance": 0.30,
            "correct_outcome": 0.40,
            "efficiency": 0.30,
        },
    ),
    "complex_complaint_resolution": TaskDefinition(
        id="complex_complaint_resolution",
        name="Complex Complaint Resolution",
        difficulty="hard",
        description=(
            "The customer received the wrong item AND was charged twice. The "
            "agent must identify both issues, process a refund, apply "
            "appropriate compensation, and notify the customer with a full "
            "resolution summary."
        ),
        max_steps=15,
        required_tools=[
            "lookup_order",
            "lookup_customer",
            "check_return_policy",
            "initiate_refund",
            "apply_compensation",
            "send_notification",
        ],
        scoring_criteria={
            "issue_identification": 0.25,
            "resolution_quality": 0.30,
            "policy_compliance": 0.25,
            "efficiency": 0.20,
        },
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TaskRegistry:
    """Registry that provides task definitions and fixture data."""

    @staticmethod
    def get_task(task_id: str) -> TaskDefinition:
        """Return the :class:`TaskDefinition` for *task_id*.

        Raises ``ValueError`` when *task_id* is not recognised.
        """
        if task_id not in _TASK_DEFINITIONS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid IDs: {list(_TASK_DEFINITIONS.keys())}"
            )
        return _TASK_DEFINITIONS[task_id]

    @staticmethod
    def list_tasks() -> List[TaskDefinition]:
        """Return all registered task definitions."""
        return list(_TASK_DEFINITIONS.values())

    @staticmethod
    def build_fixtures(task_id: str) -> Dict[str, Any]:
        """Load and return the deterministic fixture bundle for *task_id*."""
        return load_fixtures(task_id)

    @staticmethod
    def get_initial_observation_text(task_id: str) -> str:
        """Return the customer's initial message for the given task."""
        fixtures = load_fixtures(task_id)
        return fixtures["customer_message"]
