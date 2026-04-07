"""
Simulated customer-service tools and the ToolRegistry dispatcher.

Each tool is a pure function that operates on in-memory fixture data and
returns a structured result dict.  The ``ToolRegistry`` validates arguments,
dispatches by name, and converts errors into structured ``ToolError`` dicts
so the server never crashes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Custom exception for tool-level validation errors
# ---------------------------------------------------------------------------

class ToolError(Exception):
    """Raised when a tool encounters an expected error condition."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def lookup_order(order_id: str, fixtures: Dict[str, Any]) -> Dict[str, Any]:
    """Look up an order by its ID.

    Returns order details on success or raises ``ToolError`` when the
    order is not found in the fixture data.
    """
    orders = fixtures.get("orders", {})
    if order_id not in orders:
        raise ToolError(f"Order '{order_id}' not found.")
    order = orders[order_id]
    return {
        "status": "success",
        "order": order,
    }


def lookup_customer(customer_id: str, fixtures: Dict[str, Any]) -> Dict[str, Any]:
    """Look up a customer by their ID.

    Returns customer profile on success or raises ``ToolError``.
    """
    customers = fixtures.get("customers", {})
    if customer_id not in customers:
        raise ToolError(f"Customer '{customer_id}' not found.")
    customer = customers[customer_id]
    return {
        "status": "success",
        "customer": customer,
    }


def check_return_policy(
    product_category: str,
    order_date: str,
    fixtures: Dict[str, Any],
) -> Dict[str, Any]:
    """Check whether a product category is eligible for return.

    Uses the fixture return-policy table and compares the order date
    against the return window.  Returns eligibility status, window, and
    reason.
    """
    policies = fixtures.get("return_policies", {})
    policy = policies.get(product_category)
    if policy is None:
        return {
            "status": "success",
            "eligible": False,
            "return_window_days": 0,
            "reason": f"No return policy found for category '{product_category}'.",
        }

    try:
        order_dt = datetime.strptime(order_date, "%Y-%m-%d")
    except ValueError:
        raise ToolError(
            f"Invalid order_date format '{order_date}'. Expected YYYY-MM-DD."
        )

    window_days = policy["return_window_days"]
    # Use a fixed "today" for determinism.  Chosen so that all fixture
    # orders (earliest: 2024-01-01) are within the 30-day return window.
    reference_date = datetime(2024, 1, 20)
    deadline = order_dt + timedelta(days=window_days)

    eligible = reference_date <= deadline
    reason = (
        f"Within {window_days}-day return window."
        if eligible
        else f"Outside {window_days}-day return window (expired {deadline.date()})."
    )

    return {
        "status": "success",
        "eligible": eligible,
        "return_window_days": window_days,
        "reason": reason,
    }


def initiate_refund(
    order_id: str,
    amount: float,
    fixtures: Dict[str, Any],
    *,
    _refund_eligible: Optional[bool] = None,
) -> Dict[str, Any]:
    """Initiate a refund for the specified order and amount.

    If ``_refund_eligible`` is explicitly ``False`` (set by the environment
    when policy was checked and failed), the refund is denied.  Otherwise
    the refund is approved with a deterministic confirmation ID.
    """
    orders = fixtures.get("orders", {})
    if order_id not in orders:
        raise ToolError(f"Order '{order_id}' not found.")

    if _refund_eligible is False:
        return {
            "status": "denied",
            "reason": "Order is not eligible for refund per return policy.",
        }

    return {
        "status": "approved",
        "refund_id": f"REF-{order_id[-4:]}",
        "amount": amount,
        "estimated_processing_days": 5,
        "message": f"Refund of ${amount:.2f} approved for order {order_id}. "
                   f"Estimated processing time: 5 business days.",
    }


def send_notification(
    customer_id: str,
    message: str,
    fixtures: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a notification message to the customer.

    Always succeeds in the simulated environment and returns a
    delivery confirmation.
    """
    customers = fixtures.get("customers", {})
    if customer_id not in customers:
        raise ToolError(f"Customer '{customer_id}' not found.")
    return {
        "status": "delivered",
        "notification_id": f"NOTIF-{customer_id}",
        "customer_id": customer_id,
        "message_preview": message[:120],
    }


def escalate_to_human(reason: str, **kwargs: Any) -> Dict[str, Any]:
    """Escalate the case to a human agent.

    Returns an escalation confirmation with a deterministic ticket ID.
    """
    return {
        "status": "escalated",
        "escalation_id": "ESC-001",
        "reason": reason,
        "message": "Case has been escalated to a human agent.",
    }


def apply_compensation(
    customer_id: str,
    comp_type: str,
    amount: float,
    fixtures: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a compensation credit or discount to the customer account.

    ``comp_type`` should be one of ``store_credit``, ``discount``, or
    ``refund_bonus``.
    """
    customers = fixtures.get("customers", {})
    if customer_id not in customers:
        raise ToolError(f"Customer '{customer_id}' not found.")
    valid_types = {"store_credit", "discount", "refund_bonus"}
    if comp_type not in valid_types:
        raise ToolError(
            f"Invalid compensation type '{comp_type}'. "
            f"Must be one of {sorted(valid_types)}."
        )
    return {
        "status": "applied",
        "compensation_id": f"COMP-{customer_id}",
        "type": comp_type,
        "amount": amount,
        "customer_id": customer_id,
        "message": f"${amount:.2f} {comp_type} applied to customer {customer_id}.",
    }


# ---------------------------------------------------------------------------
# Tool registry & dispatcher
# ---------------------------------------------------------------------------

# Mapping of tool name to implementation function
TOOLS: Dict[str, Any] = {
    "lookup_order": lookup_order,
    "lookup_customer": lookup_customer,
    "check_return_policy": check_return_policy,
    "initiate_refund": initiate_refund,
    "send_notification": send_notification,
    "escalate_to_human": escalate_to_human,
    "apply_compensation": apply_compensation,
}

# Required argument schemas per tool (name -> list of (arg_name, type))
_TOOL_ARG_SCHEMAS: Dict[str, List[tuple]] = {
    "lookup_order": [("order_id", str)],
    "lookup_customer": [("customer_id", str)],
    "check_return_policy": [("product_category", str), ("order_date", str)],
    "initiate_refund": [("order_id", str), ("amount", (int, float))],
    "send_notification": [("customer_id", str), ("message", str)],
    "escalate_to_human": [("reason", str)],
    "apply_compensation": [
        ("customer_id", str),
        ("comp_type", str),
        ("amount", (int, float)),
    ],
}


class ToolRegistry:
    """Validates arguments, dispatches tool calls, and catches errors."""

    @staticmethod
    def dispatch(
        tool_name: str,
        args: Dict[str, Any],
        fixtures: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dispatch a tool call and return the structured result.

        Returns a dict with at least a ``"status"`` key.  If the tool name
        is unknown or arguments are invalid, the returned dict has
        ``"status": "error"`` and an ``"error_type"`` field.
        """
        if tool_name not in TOOLS:
            return {
                "status": "error",
                "error_type": "unknown_tool",
                "message": f"Unknown tool '{tool_name}'. "
                           f"Available: {sorted(TOOLS.keys())}",
            }

        # Validate required arguments
        schema = _TOOL_ARG_SCHEMAS.get(tool_name, [])
        for arg_name, expected_type in schema:
            if arg_name not in args:
                return {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": f"Missing required argument '{arg_name}' for tool '{tool_name}'.",
                }
            if not isinstance(args[arg_name], expected_type):
                return {
                    "status": "error",
                    "error_type": "validation_error",
                    "message": (
                        f"Argument '{arg_name}' must be of type "
                        f"{expected_type}, got {type(args[arg_name]).__name__}."
                    ),
                }

        try:
            func = TOOLS[tool_name]
            if tool_name == "escalate_to_human":
                # escalate_to_human does not need fixtures
                return func(reason=args["reason"])
            elif tool_name == "lookup_order":
                return func(order_id=args["order_id"], fixtures=fixtures)
            elif tool_name == "lookup_customer":
                return func(customer_id=args["customer_id"], fixtures=fixtures)
            elif tool_name == "check_return_policy":
                return func(
                    product_category=args["product_category"],
                    order_date=args["order_date"],
                    fixtures=fixtures,
                )
            elif tool_name == "initiate_refund":
                return func(
                    order_id=args["order_id"],
                    amount=args["amount"],
                    fixtures=fixtures,
                )
            elif tool_name == "send_notification":
                return func(
                    customer_id=args["customer_id"],
                    message=args["message"],
                    fixtures=fixtures,
                )
            elif tool_name == "apply_compensation":
                return func(
                    customer_id=args["customer_id"],
                    comp_type=args["comp_type"],
                    amount=args["amount"],
                    fixtures=fixtures,
                )
            else:
                return {
                    "status": "error",
                    "error_type": "unknown_tool",
                    "message": f"Tool '{tool_name}' has no dispatch handler.",
                }
        except ToolError as exc:
            return {
                "status": "error",
                "error_type": "tool_error",
                "message": exc.message,
            }
        except Exception as exc:
            return {
                "status": "error",
                "error_type": "internal_error",
                "message": f"Internal error in tool '{tool_name}': {exc}",
            }


# ---------------------------------------------------------------------------
# OpenAI-compatible function definitions
# ---------------------------------------------------------------------------

def get_openai_tool_definitions() -> List[Dict[str, Any]]:
    """Return OpenAI function-calling compatible tool schemas."""
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order by its order ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID to look up (e.g. ORD-1001).",
                        },
                    },
                    "required": ["order_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_customer",
                "description": "Look up a customer by their customer ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer ID to look up (e.g. C001).",
                        },
                    },
                    "required": ["customer_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_return_policy",
                "description": (
                    "Check the return policy for a product category and order date. "
                    "Returns eligibility, return window, and reason."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_category": {
                            "type": "string",
                            "description": "Product category (e.g. electronics, books).",
                        },
                        "order_date": {
                            "type": "string",
                            "description": "Order date in YYYY-MM-DD format.",
                        },
                    },
                    "required": ["product_category", "order_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "initiate_refund",
                "description": (
                    "Initiate a refund for a specific order. Returns approval or denial."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The order ID to refund.",
                        },
                        "amount": {
                            "type": "number",
                            "description": "The refund amount in dollars.",
                        },
                    },
                    "required": ["order_id", "amount"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_notification",
                "description": (
                    "Send a notification message to the customer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer ID to notify.",
                        },
                        "message": {
                            "type": "string",
                            "description": "The notification message content.",
                        },
                    },
                    "required": ["customer_id", "message"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": (
                    "Escalate the case to a human agent when the issue cannot "
                    "be resolved automatically."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Reason for escalation.",
                        },
                    },
                    "required": ["reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_compensation",
                "description": (
                    "Apply a compensation (store credit, discount, or refund bonus) "
                    "to a customer account."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {
                            "type": "string",
                            "description": "The customer ID to compensate.",
                        },
                        "comp_type": {
                            "type": "string",
                            "enum": ["store_credit", "discount", "refund_bonus"],
                            "description": "Type of compensation.",
                        },
                        "amount": {
                            "type": "number",
                            "description": "Compensation amount in dollars.",
                        },
                    },
                    "required": ["customer_id", "comp_type", "amount"],
                },
            },
        },
    ]
