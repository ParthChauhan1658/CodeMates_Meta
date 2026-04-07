"""Integration tests for the FastAPI endpoints."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_valid_task(self):
        resp = client.post("/reset", json={"task_id": "order_status_inquiry"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["observation"]["task_id"] == "order_status_inquiry"
        assert len(data["observation"]["available_tools"]) == 7

    def test_reset_unknown_task(self):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 422

    def test_reset_missing_task_id(self):
        resp = client.post("/reset", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

class TestStep:

    def _reset(self, task_id="order_status_inquiry"):
        resp = client.post("/reset", json={"task_id": task_id})
        return resp.json()["session_id"]

    def test_step_valid(self):
        sid = self._reset()
        resp = client.post("/step", json={
            "session_id": sid,
            "tool_name": "lookup_order",
            "tool_args": {"order_id": "ORD-1001"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_without_reset(self):
        resp = client.post("/step", json={
            "session_id": "fake-session-id",
            "tool_name": "lookup_order",
            "tool_args": {"order_id": "ORD-1001"},
        })
        assert resp.status_code == 400

    def test_step_missing_fields(self):
        resp = client.post("/step", json={"session_id": "fake"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /state
# ---------------------------------------------------------------------------

class TestState:

    def test_state_after_reset(self):
        resp = client.post("/reset", json={"task_id": "order_status_inquiry"})
        sid = resp.json()["session_id"]
        resp = client.get("/state", params={"session_id": sid})
        assert resp.status_code == 200
        state = resp.json()["state"]
        assert state["task_id"] == "order_status_inquiry"
        assert state["step_count"] == 0

    def test_state_unknown_session(self):
        resp = client.get("/state", params={"session_id": "unknown"})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

class TestTasks:

    def test_list_tasks(self):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 3
        ids = {t["id"] for t in data["tasks"]}
        assert "order_status_inquiry" in ids
        assert "return_refund_processing" in ids
        assert "complex_complaint_resolution" in ids
        assert data["action_schema"] == {"tool_name": "str", "tool_args": "dict"}


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------

class TestGrader:

    def test_grader_with_session(self):
        # Run a full task 1 episode, then grade it
        resp = client.post("/reset", json={"task_id": "order_status_inquiry"})
        sid = resp.json()["session_id"]

        client.post("/step", json={
            "session_id": sid, "tool_name": "lookup_order",
            "tool_args": {"order_id": "ORD-1001"},
        })
        client.post("/step", json={
            "session_id": sid, "tool_name": "lookup_customer",
            "tool_args": {"customer_id": "C001"},
        })
        client.post("/step", json={
            "session_id": sid, "tool_name": "send_notification",
            "tool_args": {"customer_id": "C001", "message": "Shipped"},
        })

        resp = client.post("/grader", json={"session_id": sid})
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] == 1.0

    def test_grader_empty_trajectory(self):
        resp = client.post("/grader", json={
            "task_id": "order_status_inquiry",
            "trajectory": [],
        })
        assert resp.status_code == 200
        assert resp.json()["score"] == 0.0


# ---------------------------------------------------------------------------
# /baseline
# ---------------------------------------------------------------------------

class TestBaseline:

    def test_baseline_info(self):
        resp = client.get("/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert "model" in data
        assert "reference_scores" in data


# ---------------------------------------------------------------------------
# Full episode via API
# ---------------------------------------------------------------------------

class TestFullEpisodeViaAPI:

    def test_task2_full_episode(self):
        resp = client.post("/reset", json={"task_id": "return_refund_processing"})
        sid = resp.json()["session_id"]

        for tool_name, tool_args in [
            ("lookup_order", {"order_id": "ORD-2001"}),
            ("lookup_customer", {"customer_id": "C002"}),
            ("check_return_policy", {"product_category": "electronics", "order_date": "2024-01-05"}),
            ("initiate_refund", {"order_id": "ORD-2001", "amount": 199.99}),
            ("send_notification", {"customer_id": "C002", "message": "Refund done"}),
        ]:
            resp = client.post("/step", json={
                "session_id": sid,
                "tool_name": tool_name,
                "tool_args": tool_args,
            })

        data = resp.json()
        assert data["done"] is True

        # Grade it
        resp = client.post("/grader", json={"session_id": sid})
        assert resp.json()["score"] == 1.0
