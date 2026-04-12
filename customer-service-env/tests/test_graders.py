"""Unit tests for the episode graders."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import CustomerServiceState
from server.graders import GraderRegistry, grade_task1, grade_task2, grade_task3


# ---------------------------------------------------------------------------
# Task 1 grader
# ---------------------------------------------------------------------------

class TestGradeTask1:

    def test_perfect_trajectory(self):
        actions = [
            {
                "tool_name": "lookup_order",
                "tool_args": {"order_id": "ORD-1001"},
                "tool_result": {"status": "success", "order": {"id": "ORD-1001", "status": "shipped"}},
            },
            {
                "tool_name": "lookup_customer",
                "tool_args": {"customer_id": "C001"},
                "tool_result": {"status": "success", "customer": {"id": "C001"}},
            },
            {
                "tool_name": "send_notification",
                "tool_args": {"customer_id": "C001", "message": "Your order has shipped."},
                "tool_result": {"status": "delivered"},
            },
        ]
        state = CustomerServiceState(task_id="order_status_inquiry")
        result = grade_task1(actions, state)
        assert result["score"] == 0.999
        assert result["breakdown"]["tool_correctness"] == 1.0
        assert result["breakdown"]["resolution"] == 1.0
        assert result["breakdown"]["efficiency"] == 1.0

    def test_missing_notification_scores_lower(self):
        actions = [
            {
                "tool_name": "lookup_order",
                "tool_args": {"order_id": "ORD-1001"},
                "tool_result": {"status": "success"},
            },
            {
                "tool_name": "lookup_customer",
                "tool_args": {"customer_id": "C001"},
                "tool_result": {"status": "success"},
            },
        ]
        state = CustomerServiceState(task_id="order_status_inquiry")
        result = grade_task1(actions, state)
        assert result["score"] < 1.0
        assert result["breakdown"]["resolution"] == 0.0

    def test_empty_trajectory_via_registry(self):
        state = CustomerServiceState(task_id="order_status_inquiry")
        resp = GraderRegistry.grade("order_status_inquiry", [], state)
        assert resp.score == 0.001
        assert "empty" in resp.explanation.lower() or "Empty" in resp.explanation


# ---------------------------------------------------------------------------
# Task 2 grader
# ---------------------------------------------------------------------------

class TestGradeTask2:

    def test_perfect_trajectory(self):
        actions = [
            {
                "tool_name": "lookup_order",
                "tool_args": {"order_id": "ORD-2001"},
                "tool_result": {"status": "success"},
            },
            {
                "tool_name": "lookup_customer",
                "tool_args": {"customer_id": "C002"},
                "tool_result": {"status": "success"},
            },
            {
                "tool_name": "check_return_policy",
                "tool_args": {"product_category": "electronics", "order_date": "2024-01-05"},
                "tool_result": {"status": "success", "eligible": True},
            },
            {
                "tool_name": "initiate_refund",
                "tool_args": {"order_id": "ORD-2001", "amount": 199.99},
                "tool_result": {"status": "approved", "refund_id": "REF-2001"},
            },
            {
                "tool_name": "send_notification",
                "tool_args": {"customer_id": "C002", "message": "Refund approved."},
                "tool_result": {"status": "delivered"},
            },
        ]
        state = CustomerServiceState(task_id="return_refund_processing")
        result = grade_task2(actions, state)
        assert result["score"] == 0.999

    def test_missing_policy_check_lowers_compliance(self):
        actions = [
            {
                "tool_name": "lookup_order",
                "tool_args": {"order_id": "ORD-2001"},
                "tool_result": {"status": "success"},
            },
            {
                "tool_name": "initiate_refund",
                "tool_args": {"order_id": "ORD-2001", "amount": 199.99},
                "tool_result": {"status": "approved"},
            },
            {
                "tool_name": "send_notification",
                "tool_args": {"customer_id": "C002", "message": "Done"},
                "tool_result": {"status": "delivered"},
            },
        ]
        state = CustomerServiceState(task_id="return_refund_processing")
        result = grade_task2(actions, state)
        assert result["breakdown"]["policy_compliance"] < 1.0


# ---------------------------------------------------------------------------
# Task 3 grader
# ---------------------------------------------------------------------------

class TestGradeTask3:

    def test_perfect_trajectory(self):
        actions = [
            {"tool_name": "lookup_order", "tool_args": {"order_id": "ORD-3001"}, "tool_result": {"status": "success"}},
            {"tool_name": "lookup_customer", "tool_args": {"customer_id": "C003"}, "tool_result": {"status": "success"}},
            {"tool_name": "check_return_policy", "tool_args": {"product_category": "electronics", "order_date": "2024-01-01"}, "tool_result": {"status": "success", "eligible": True}},
            {"tool_name": "initiate_refund", "tool_args": {"order_id": "ORD-3001", "amount": 899.99}, "tool_result": {"status": "approved"}},
            {"tool_name": "apply_compensation", "tool_args": {"customer_id": "C003", "comp_type": "store_credit", "amount": 50.0}, "tool_result": {"status": "applied"}},
            {"tool_name": "send_notification", "tool_args": {"customer_id": "C003", "message": "Resolved"}, "tool_result": {"status": "delivered"}},
        ]
        state = CustomerServiceState(task_id="complex_complaint_resolution")
        result = grade_task3(actions, state)
        assert result["score"] == 0.999

    def test_valid_escalation_accepted(self):
        actions = [
            {"tool_name": "lookup_order", "tool_args": {"order_id": "ORD-3001"}, "tool_result": {"status": "success"}},
            {"tool_name": "lookup_customer", "tool_args": {"customer_id": "C003"}, "tool_result": {"status": "success"}},
            {"tool_name": "escalate_to_human", "tool_args": {"reason": "wrong_item_delivered and billing issue"}, "tool_result": {"status": "escalated"}},
        ]
        state = CustomerServiceState(task_id="complex_complaint_resolution")
        result = grade_task3(actions, state)
        # Escalation with valid reason should not penalise policy_compliance
        assert result["breakdown"]["policy_compliance"] >= 0.0

    def test_invalid_escalation_penalised(self):
        actions = [
            {"tool_name": "lookup_order", "tool_args": {"order_id": "ORD-3001"}, "tool_result": {"status": "success"}},
            {"tool_name": "escalate_to_human", "tool_args": {"reason": "I'm tired"}, "tool_result": {"status": "escalated"}},
        ]
        state = CustomerServiceState(task_id="complex_complaint_resolution")
        result = grade_task3(actions, state)
        # Invalid escalation penalises policy compliance
        assert result["breakdown"]["policy_compliance"] <= 0.0

    def test_unknown_task_returns_zero(self):
        state = CustomerServiceState(task_id="unknown")
        resp = GraderRegistry.grade("unknown", [{"tool_name": "x"}], state)
        assert resp.score == 0.001


# ---------------------------------------------------------------------------
# GraderRegistry integration
# ---------------------------------------------------------------------------

class TestGraderRegistry:

    def test_grade_task1_via_registry(self):
        actions = [
            {"tool_name": "lookup_order", "tool_args": {"order_id": "ORD-1001"}, "tool_result": {"status": "success", "order": {"id": "ORD-1001", "status": "shipped"}}},
            {"tool_name": "lookup_customer", "tool_args": {"customer_id": "C001"}, "tool_result": {"status": "success", "customer": {"id": "C001"}}},
            {"tool_name": "send_notification", "tool_args": {"customer_id": "C001", "message": "Shipped"}, "tool_result": {"status": "delivered"}},
        ]
        state = CustomerServiceState(task_id="order_status_inquiry")
        resp = GraderRegistry.grade("order_status_inquiry", actions, state)
        assert resp.score == 0.999
        assert resp.task_id == "order_status_inquiry"
        assert "breakdown" in resp.model_dump() or resp.breakdown is not None
