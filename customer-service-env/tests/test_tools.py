"""Unit tests for the simulated tools and ToolRegistry."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from server.fixtures import load_fixtures
from server.tools import (
    ToolError,
    ToolRegistry,
    apply_compensation,
    check_return_policy,
    escalate_to_human,
    get_openai_tool_definitions,
    initiate_refund,
    lookup_customer,
    lookup_order,
    send_notification,
)


# ---------------------------------------------------------------------------
# Fixtures (pytest)
# ---------------------------------------------------------------------------

@pytest.fixture
def task1_fixtures():
    return load_fixtures("order_status_inquiry")


@pytest.fixture
def task2_fixtures():
    return load_fixtures("return_refund_processing")


@pytest.fixture
def task3_fixtures():
    return load_fixtures("complex_complaint_resolution")


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------

class TestLookupOrder:

    def test_valid_order(self, task1_fixtures):
        result = lookup_order("ORD-1001", task1_fixtures)
        assert result["status"] == "success"
        assert result["order"]["id"] == "ORD-1001"
        assert result["order"]["status"] == "shipped"

    def test_invalid_order(self, task1_fixtures):
        with pytest.raises(ToolError, match="not found"):
            lookup_order("ORD-9999", task1_fixtures)


# ---------------------------------------------------------------------------
# lookup_customer
# ---------------------------------------------------------------------------

class TestLookupCustomer:

    def test_valid_customer(self, task1_fixtures):
        result = lookup_customer("C001", task1_fixtures)
        assert result["status"] == "success"
        assert result["customer"]["name"] == "Alice Johnson"

    def test_invalid_customer(self, task1_fixtures):
        with pytest.raises(ToolError, match="not found"):
            lookup_customer("C999", task1_fixtures)


# ---------------------------------------------------------------------------
# check_return_policy
# ---------------------------------------------------------------------------

class TestCheckReturnPolicy:

    def test_eligible(self, task2_fixtures):
        result = check_return_policy("electronics", "2024-01-05", task2_fixtures)
        assert result["status"] == "success"
        assert result["eligible"] is True
        assert result["return_window_days"] == 30

    def test_unknown_category(self, task2_fixtures):
        result = check_return_policy("alien_tech", "2024-01-05", task2_fixtures)
        assert result["eligible"] is False

    def test_invalid_date_format(self, task2_fixtures):
        with pytest.raises(ToolError, match="Invalid order_date"):
            check_return_policy("electronics", "not-a-date", task2_fixtures)


# ---------------------------------------------------------------------------
# initiate_refund
# ---------------------------------------------------------------------------

class TestInitiateRefund:

    def test_approved_refund(self, task2_fixtures):
        result = initiate_refund("ORD-2001", 199.99, task2_fixtures)
        assert result["status"] == "approved"
        assert result["amount"] == 199.99
        assert "REF-" in result["refund_id"]

    def test_refund_denied_when_ineligible(self, task2_fixtures):
        result = initiate_refund("ORD-2001", 199.99, task2_fixtures, _refund_eligible=False)
        assert result["status"] == "denied"

    def test_refund_unknown_order(self, task2_fixtures):
        with pytest.raises(ToolError, match="not found"):
            initiate_refund("ORD-FAKE", 100.0, task2_fixtures)


# ---------------------------------------------------------------------------
# send_notification
# ---------------------------------------------------------------------------

class TestSendNotification:

    def test_send_ok(self, task1_fixtures):
        result = send_notification("C001", "Your order has shipped.", task1_fixtures)
        assert result["status"] == "delivered"
        assert result["customer_id"] == "C001"

    def test_send_unknown_customer(self, task1_fixtures):
        with pytest.raises(ToolError, match="not found"):
            send_notification("C999", "Hello", task1_fixtures)


# ---------------------------------------------------------------------------
# escalate_to_human
# ---------------------------------------------------------------------------

class TestEscalateToHuman:

    def test_escalation(self):
        result = escalate_to_human("Customer is very upset")
        assert result["status"] == "escalated"
        assert result["escalation_id"] == "ESC-001"


# ---------------------------------------------------------------------------
# apply_compensation
# ---------------------------------------------------------------------------

class TestApplyCompensation:

    def test_valid_compensation(self, task3_fixtures):
        result = apply_compensation("C003", "store_credit", 50.0, task3_fixtures)
        assert result["status"] == "applied"
        assert result["amount"] == 50.0

    def test_invalid_type(self, task3_fixtures):
        with pytest.raises(ToolError, match="Invalid compensation type"):
            apply_compensation("C003", "free_pizza", 10.0, task3_fixtures)

    def test_unknown_customer(self, task3_fixtures):
        with pytest.raises(ToolError, match="not found"):
            apply_compensation("C999", "store_credit", 10.0, task3_fixtures)


# ---------------------------------------------------------------------------
# ToolRegistry.dispatch
# ---------------------------------------------------------------------------

class TestToolRegistry:

    def test_dispatch_valid(self, task1_fixtures):
        result = ToolRegistry.dispatch(
            "lookup_order", {"order_id": "ORD-1001"}, task1_fixtures
        )
        assert result["status"] == "success"

    def test_dispatch_unknown_tool(self, task1_fixtures):
        result = ToolRegistry.dispatch("magic_wand", {}, task1_fixtures)
        assert result["status"] == "error"
        assert result["error_type"] == "unknown_tool"

    def test_dispatch_missing_arg(self, task1_fixtures):
        result = ToolRegistry.dispatch("lookup_order", {}, task1_fixtures)
        assert result["status"] == "error"
        assert result["error_type"] == "validation_error"

    def test_dispatch_wrong_arg_type(self, task1_fixtures):
        result = ToolRegistry.dispatch("lookup_order", {"order_id": 12345}, task1_fixtures)
        assert result["status"] == "error"
        assert result["error_type"] == "validation_error"

    def test_dispatch_tool_error_not_crash(self, task1_fixtures):
        result = ToolRegistry.dispatch(
            "lookup_order", {"order_id": "FAKE"}, task1_fixtures
        )
        assert result["status"] == "error"
        assert result["error_type"] == "tool_error"


# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------

class TestOpenAIToolDefinitions:

    def test_returns_7_tools(self):
        defs = get_openai_tool_definitions()
        assert len(defs) == 7

    def test_schema_structure(self):
        defs = get_openai_tool_definitions()
        for d in defs:
            assert d["type"] == "function"
            assert "name" in d["function"]
            assert "parameters" in d["function"]
