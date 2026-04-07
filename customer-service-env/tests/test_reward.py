"""Unit tests for reward computation logic."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import CustomerServiceAction
from server.environment import CustomerServiceEnvironment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(task_id: str) -> CustomerServiceEnvironment:
    env = CustomerServiceEnvironment()
    env.reset(task_id=task_id)
    return env


def _step(env: CustomerServiceEnvironment, tool_name: str, tool_args: dict):
    action = CustomerServiceAction(tool_name=tool_name, tool_args=tool_args)
    return env.step(action)


# ---------------------------------------------------------------------------
# Task 1 reward tests
# ---------------------------------------------------------------------------

class TestTask1Reward:

    def test_correct_lookup_order_gives_positive_reward(self):
        env = _make_env("order_status_inquiry")
        obs, reward, done, info = _step(env, "lookup_order", {"order_id": "ORD-1001"})
        assert reward > 0
        assert "correct_lookup_order" in info["reward_components"]

    def test_correct_lookup_customer_gives_positive_reward(self):
        env = _make_env("order_status_inquiry")
        _step(env, "lookup_order", {"order_id": "ORD-1001"})
        obs, reward, done, info = _step(env, "lookup_customer", {"customer_id": "C001"})
        assert reward > 0
        assert "correct_lookup_customer" in info["reward_components"]

    def test_send_notification_completes_episode(self):
        env = _make_env("order_status_inquiry")
        _step(env, "lookup_order", {"order_id": "ORD-1001"})
        _step(env, "lookup_customer", {"customer_id": "C001"})
        obs, reward, done, info = _step(
            env, "send_notification",
            {"customer_id": "C001", "message": "Your order ORD-1001 has been shipped."},
        )
        assert done is True
        assert reward > 0

    def test_unnecessary_escalation_penalised(self):
        env = _make_env("order_status_inquiry")
        obs, reward, done, info = _step(
            env, "escalate_to_human", {"reason": "I don't know what to do"}
        )
        assert reward < 0
        assert "unnecessary_escalation" in info["reward_components"]

    def test_validation_error_penalised(self):
        env = _make_env("order_status_inquiry")
        obs, reward, done, info = _step(env, "lookup_order", {})  # missing order_id
        assert reward < 0
        assert "wrong_tool_args" in info["reward_components"]


# ---------------------------------------------------------------------------
# Task 2 reward tests
# ---------------------------------------------------------------------------

class TestTask2Reward:

    def test_refund_without_policy_check_penalised(self):
        env = _make_env("return_refund_processing")
        _step(env, "lookup_order", {"order_id": "ORD-2001"})
        obs, reward, done, info = _step(
            env, "initiate_refund", {"order_id": "ORD-2001", "amount": 199.99}
        )
        # Should get positive reward for correct tool MINUS policy violation
        assert "policy_violation" in info["reward_components"]

    def test_policy_then_refund_no_penalty(self):
        env = _make_env("return_refund_processing")
        _step(env, "lookup_order", {"order_id": "ORD-2001"})
        _step(env, "lookup_customer", {"customer_id": "C002"})
        _step(env, "check_return_policy", {"product_category": "electronics", "order_date": "2024-01-05"})
        obs, reward, done, info = _step(
            env, "initiate_refund", {"order_id": "ORD-2001", "amount": 199.99}
        )
        assert "policy_violation" not in info["reward_components"]


# ---------------------------------------------------------------------------
# Cumulative reward clamping
# ---------------------------------------------------------------------------

class TestRewardClamping:

    def test_cumulative_reward_clamped_to_zero(self):
        env = _make_env("order_status_inquiry")
        # Burn steps with errors to accumulate negative reward
        for _ in range(4):
            _step(env, "lookup_order", {})  # validation errors

        # Last step triggers step limit
        obs, reward, done, info = _step(env, "lookup_order", {})
        assert done is True
        state = env.state
        assert state.cumulative_reward >= 0.0

    def test_cumulative_reward_clamped_to_one(self):
        env = _make_env("order_status_inquiry")
        _step(env, "lookup_order", {"order_id": "ORD-1001"})
        _step(env, "lookup_customer", {"customer_id": "C001"})
        obs, reward, done, info = _step(
            env, "send_notification",
            {"customer_id": "C001", "message": "Shipped"},
        )
        assert done is True
        state = env.state
        assert state.cumulative_reward <= 1.0
