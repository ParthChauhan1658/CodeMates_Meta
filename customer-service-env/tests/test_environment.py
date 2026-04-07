"""Integration tests for the full episode loop across all three tasks."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import CustomerServiceAction
from server.environment import CustomerServiceEnvironment


def _step(env, tool_name, tool_args):
    action = CustomerServiceAction(tool_name=tool_name, tool_args=tool_args)
    return env.step(action)


# ---------------------------------------------------------------------------
# Task 1 -- Order Status Inquiry
# ---------------------------------------------------------------------------

class TestTask1Episode:

    def test_happy_path(self):
        env = CustomerServiceEnvironment()
        obs = env.reset("order_status_inquiry")
        assert obs.task_id == "order_status_inquiry"
        assert "ORD-1001" in obs.customer_message

        obs, r1, done, info = _step(env, "lookup_order", {"order_id": "ORD-1001"})
        assert not done
        assert r1 > 0

        obs, r2, done, info = _step(env, "lookup_customer", {"customer_id": "C001"})
        assert not done

        obs, r3, done, info = _step(
            env, "send_notification",
            {"customer_id": "C001", "message": "Your order ORD-1001 is shipped."},
        )
        assert done is True
        state = env.state
        assert state.resolution_status == "resolved"
        assert state.cumulative_reward > 0

    def test_invalid_order_continues_episode(self):
        env = CustomerServiceEnvironment()
        env.reset("order_status_inquiry")
        obs, reward, done, info = _step(env, "lookup_order", {"order_id": "FAKE"})
        assert not done
        assert obs.tool_result["status"] == "error"

    def test_step_limit_exceeded(self):
        env = CustomerServiceEnvironment()
        env.reset("order_status_inquiry")
        for i in range(5):
            obs, reward, done, info = _step(
                env, "lookup_customer", {"customer_id": "C001"}
            )
        assert done is True
        assert env.state.resolution_status == "step_limit_exceeded"


# ---------------------------------------------------------------------------
# Task 2 -- Return & Refund Processing
# ---------------------------------------------------------------------------

class TestTask2Episode:

    def test_happy_path(self):
        env = CustomerServiceEnvironment()
        env.reset("return_refund_processing")

        _step(env, "lookup_order", {"order_id": "ORD-2001"})
        _step(env, "lookup_customer", {"customer_id": "C002"})
        _step(env, "check_return_policy", {"product_category": "electronics", "order_date": "2024-01-05"})
        _step(env, "initiate_refund", {"order_id": "ORD-2001", "amount": 199.99})
        obs, reward, done, info = _step(
            env, "send_notification",
            {"customer_id": "C002", "message": "Your refund has been approved."},
        )
        assert done is True
        assert env.state.resolution_status == "resolved"
        assert env.state.cumulative_reward > 0.5

    def test_unnecessary_escalation_penalised(self):
        env = CustomerServiceEnvironment()
        env.reset("return_refund_processing")
        obs, reward, done, info = _step(
            env, "escalate_to_human", {"reason": "too hard"}
        )
        assert done is True
        assert env.state.resolution_status == "escalated"
        assert "unnecessary_escalation" in info["reward_components"]

    def test_policy_violation_penalty(self):
        env = CustomerServiceEnvironment()
        env.reset("return_refund_processing")
        _step(env, "lookup_order", {"order_id": "ORD-2001"})
        # Skip check_return_policy -- go straight to refund
        obs, reward, done, info = _step(
            env, "initiate_refund", {"order_id": "ORD-2001", "amount": 199.99}
        )
        assert "policy_violation" in info["reward_components"]


# ---------------------------------------------------------------------------
# Task 3 -- Complex Complaint Resolution
# ---------------------------------------------------------------------------

class TestTask3Episode:

    def test_happy_path(self):
        env = CustomerServiceEnvironment()
        env.reset("complex_complaint_resolution")

        _step(env, "lookup_order", {"order_id": "ORD-3001"})
        _step(env, "lookup_customer", {"customer_id": "C003"})
        _step(env, "check_return_policy", {"product_category": "electronics", "order_date": "2024-01-01"})
        _step(env, "initiate_refund", {"order_id": "ORD-3001", "amount": 899.99})
        _step(env, "apply_compensation", {"customer_id": "C003", "comp_type": "store_credit", "amount": 50.0})
        obs, reward, done, info = _step(
            env, "send_notification",
            {"customer_id": "C003", "message": "Issue resolved. Refund and compensation applied."},
        )
        assert done is True
        assert env.state.resolution_status == "resolved"
        assert env.state.cumulative_reward > 0.5

    def test_escalation_with_valid_reason(self):
        env = CustomerServiceEnvironment()
        env.reset("complex_complaint_resolution")
        _step(env, "lookup_order", {"order_id": "ORD-3001"})
        _step(env, "lookup_customer", {"customer_id": "C003"})
        obs, reward, done, info = _step(
            env, "escalate_to_human",
            {"reason": "wrong_item_delivered and duplicate billing"},
        )
        assert done is True
        assert env.state.resolution_status == "escalated"

    def test_step_after_done_returns_terminal(self):
        env = CustomerServiceEnvironment()
        env.reset("complex_complaint_resolution")
        _step(env, "escalate_to_human", {"reason": "wrong_item_delivered"})
        # Call step again on a finished episode
        obs, reward, done, info = _step(env, "lookup_order", {"order_id": "ORD-3001"})
        assert done is True
        assert reward == 0.0


# ---------------------------------------------------------------------------
# State property
# ---------------------------------------------------------------------------

class TestStateProperty:

    def test_state_before_reset(self):
        env = CustomerServiceEnvironment()
        state = env.state
        assert state.episode_id is None

    def test_state_after_reset(self):
        env = CustomerServiceEnvironment()
        env.reset("order_status_inquiry")
        state = env.state
        assert state.task_id == "order_status_inquiry"
        assert state.episode_id is not None
        assert state.step_count == 0
        assert state.max_steps == 5

    def test_state_tracks_actions(self):
        env = CustomerServiceEnvironment()
        env.reset("order_status_inquiry")
        _step(env, "lookup_order", {"order_id": "ORD-1001"})
        state = env.state
        assert state.step_count == 1
        assert len(state.actions_taken) == 1
        assert "lookup_order" in state.tools_called
