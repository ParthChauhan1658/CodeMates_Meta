"""
Deterministic fixture data for all three customer service tasks.

Every value is hard-coded so that episodes are fully reproducible regardless
of environment or platform.
"""

from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# Task 1 -- Order Status Inquiry (easy)
# ---------------------------------------------------------------------------

TASK1_CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "C001": {
        "id": "C001",
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "tier": "standard",
        "order_history": ["ORD-1001"],
    },
}

TASK1_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-1001": {
        "id": "ORD-1001",
        "customer_id": "C001",
        "status": "shipped",
        "items": [{"name": "Wireless Headphones", "qty": 1, "price": 79.99}],
        "total": 79.99,
        "order_date": "2024-01-10",
        "estimated_delivery": "2024-01-15",
        "tracking_number": "TRK123456",
        "category": "electronics",
    },
}

TASK1_RETURN_POLICIES: Dict[str, Dict[str, Any]] = {
    "electronics": {
        "product_category": "electronics",
        "return_window_days": 30,
        "exclusions": [],
    },
}

TASK1_CUSTOMER_MESSAGE = (
    "Hi, I ordered some headphones last week (order #ORD-1001). "
    "Can you tell me the current status of my order?"
)

TASK1_BILLING: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Task 2 -- Return & Refund Processing (medium)
# ---------------------------------------------------------------------------

TASK2_CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "C002": {
        "id": "C002",
        "name": "Bob Smith",
        "email": "bob@example.com",
        "tier": "premium",
        "order_history": ["ORD-2001"],
    },
}

TASK2_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-2001": {
        "id": "ORD-2001",
        "customer_id": "C002",
        "status": "delivered",
        "items": [{"name": "Smart Watch", "qty": 1, "price": 199.99}],
        "total": 199.99,
        "order_date": "2024-01-05",
        "delivered_date": "2024-01-09",
        "category": "electronics",
    },
}

TASK2_RETURN_POLICIES: Dict[str, Dict[str, Any]] = {
    "electronics": {
        "product_category": "electronics",
        "return_window_days": 30,
        "exclusions": ["opened_software"],
    },
}

TASK2_CUSTOMER_MESSAGE = (
    "I received my smart watch (order #ORD-2001) but it's not working "
    "properly. I'd like to return it and get a refund."
)

TASK2_BILLING: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Task 3 -- Complex Complaint Resolution (hard)
# ---------------------------------------------------------------------------

TASK3_CUSTOMERS: Dict[str, Dict[str, Any]] = {
    "C003": {
        "id": "C003",
        "name": "Carol White",
        "email": "carol@example.com",
        "tier": "vip",
        "order_history": ["ORD-3001"],
    },
}

TASK3_ORDERS: Dict[str, Dict[str, Any]] = {
    "ORD-3001": {
        "id": "ORD-3001",
        "customer_id": "C003",
        "status": "delivered",
        "items": [{"name": "Laptop", "qty": 1, "price": 899.99}],
        "total": 899.99,
        "order_date": "2024-01-01",
        "delivered_date": "2024-01-05",
        "category": "electronics",
        # Scenario details: wrong item was actually delivered
        "delivered_item": "Keyboard",
        "expected_item": "Laptop",
    },
}

TASK3_RETURN_POLICIES: Dict[str, Dict[str, Any]] = {
    "electronics": {
        "product_category": "electronics",
        "return_window_days": 30,
        "exclusions": [],
    },
}

TASK3_BILLING: Dict[str, Dict[str, Any]] = {
    "TXN-001": {
        "transaction_id": "TXN-001",
        "order_id": "ORD-3001",
        "customer_id": "C003",
        "amount": 899.99,
        "status": "duplicate_charge",
    },
}

TASK3_CONFLICT_CONDITIONS = ["wrong_item_delivered", "billing_overcharge"]

TASK3_CUSTOMER_MESSAGE = (
    "This is unacceptable! I ordered a laptop (order #ORD-3001) but received "
    "a keyboard instead! And I've been charged TWICE for this! I want this "
    "resolved immediately!"
)


# ---------------------------------------------------------------------------
# Aggregated fixture loader
# ---------------------------------------------------------------------------

def load_fixtures(task_id: str) -> Dict[str, Any]:
    """Return the complete fixture bundle for the given *task_id*.

    The returned dictionary always contains the keys:
    ``customers``, ``orders``, ``return_policies``, ``billing``,
    ``customer_message``, and ``conflict_conditions``.
    """
    if task_id == "order_status_inquiry":
        return {
            "customers": TASK1_CUSTOMERS,
            "orders": TASK1_ORDERS,
            "return_policies": TASK1_RETURN_POLICIES,
            "billing": TASK1_BILLING,
            "customer_message": TASK1_CUSTOMER_MESSAGE,
            "conflict_conditions": [],
        }
    elif task_id == "return_refund_processing":
        return {
            "customers": TASK2_CUSTOMERS,
            "orders": TASK2_ORDERS,
            "return_policies": TASK2_RETURN_POLICIES,
            "billing": TASK2_BILLING,
            "customer_message": TASK2_CUSTOMER_MESSAGE,
            "conflict_conditions": [],
        }
    elif task_id == "complex_complaint_resolution":
        return {
            "customers": TASK3_CUSTOMERS,
            "orders": TASK3_ORDERS,
            "return_policies": TASK3_RETURN_POLICIES,
            "billing": TASK3_BILLING,
            "customer_message": TASK3_CUSTOMER_MESSAGE,
            "conflict_conditions": TASK3_CONFLICT_CONDITIONS,
        }
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
