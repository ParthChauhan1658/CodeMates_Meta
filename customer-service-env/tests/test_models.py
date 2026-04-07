"""Unit tests for Pydantic data models."""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pydantic import ValidationError

from models import (
    CustomerServiceAction,
    CustomerServiceObservation,
    CustomerServiceState,
    GraderResponse,
    RewardSignal,
    TaskDefinition,
)


class TestCustomerServiceAction:
    """Tests for the Action model."""

    def test_valid_action(self) -> None:
        action = CustomerServiceAction(
            tool_name="lookup_order",
            tool_args={"order_id": "ORD-1001"},
        )
        assert action.tool_name == "lookup_order"
        assert action.tool_args == {"order_id": "ORD-1001"}
        assert action.message is None

    def test_action_with_message(self) -> None:
        action = CustomerServiceAction(
            tool_name="send_notification",
            tool_args={"customer_id": "C001", "message": "Hello"},
            message="Sending greeting",
        )
        assert action.message == "Sending greeting"

    def test_action_missing_tool_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            CustomerServiceAction(tool_args={"order_id": "ORD-1001"})  # type: ignore[call-arg]

    def test_action_default_tool_args(self) -> None:
        action = CustomerServiceAction(tool_name="escalate_to_human")
        assert action.tool_args == {}


class TestCustomerServiceObservation:
    """Tests for the Observation model."""

    def test_default_values(self) -> None:
        obs = CustomerServiceObservation()
        assert obs.done is False
        assert obs.reward is None
        assert obs.task_id == ""
        assert obs.available_tools == []
        assert obs.steps_remaining == 0

    def test_populated_observation(self) -> None:
        obs = CustomerServiceObservation(
            done=True,
            reward=0.5,
            task_id="order_status_inquiry",
            customer_message="Where is my order?",
            tool_result={"status": "success"},
            available_tools=["lookup_order"],
            steps_remaining=3,
            message="Order found",
        )
        assert obs.done is True
        assert obs.reward == 0.5
        assert obs.tool_result == {"status": "success"}


class TestCustomerServiceState:
    """Tests for the State model."""

    def test_default_state(self) -> None:
        state = CustomerServiceState()
        assert state.resolution_status == "in_progress"
        assert state.cumulative_reward == 0.0
        assert state.tools_called == []

    def test_resolution_status_literals(self) -> None:
        for status in ["in_progress", "resolved", "escalated", "step_limit_exceeded"]:
            state = CustomerServiceState(resolution_status=status)  # type: ignore[arg-type]
            assert state.resolution_status == status

    def test_invalid_resolution_status_raises(self) -> None:
        with pytest.raises(ValidationError):
            CustomerServiceState(resolution_status="invalid_status")  # type: ignore[arg-type]


class TestGraderResponse:
    """Tests for the GraderResponse model."""

    def test_default_grader_response(self) -> None:
        resp = GraderResponse()
        assert resp.score == 0.0
        assert resp.breakdown == {}

    def test_populated_grader_response(self) -> None:
        resp = GraderResponse(
            task_id="order_status_inquiry",
            score=0.85,
            breakdown={"tool_correctness": 0.9, "resolution": 1.0, "efficiency": 0.65},
            explanation="Good job",
        )
        assert resp.score == 0.85
        assert "tool_correctness" in resp.breakdown


class TestTaskDefinition:
    """Tests for the TaskDefinition model."""

    def test_valid_task(self) -> None:
        td = TaskDefinition(
            id="order_status_inquiry",
            name="Order Status Inquiry",
            difficulty="easy",
            max_steps=5,
        )
        assert td.difficulty == "easy"
        assert td.max_steps == 5

    def test_invalid_difficulty_raises(self) -> None:
        with pytest.raises(ValidationError):
            TaskDefinition(
                id="test",
                name="Test",
                difficulty="impossible",  # type: ignore[arg-type]
            )


class TestRewardSignal:
    """Tests for the RewardSignal model."""

    def test_defaults(self) -> None:
        rs = RewardSignal()
        assert rs.step_reward == 0.0
        assert rs.cumulative_reward == 0.0
        assert rs.components == {}
