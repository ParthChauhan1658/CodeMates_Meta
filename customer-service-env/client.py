"""
HTTP client wrapper for the Customer Service OpenEnv REST API.

Provides a Pythonic interface backed by ``httpx`` so that ``baseline.py``
and external evaluators can interact with the environment without
importing the server directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

from models import (
    CustomerServiceAction,
    CustomerServiceObservation,
    CustomerServiceState,
    GraderResponse,
    TaskDefinition,
)


class CustomerServiceClient:
    """Synchronous HTTP client for the Customer Service OpenEnv API.

    Args:
        base_url: Root URL of the running server (e.g. ``http://localhost:7860``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)
        self._session_id: Optional[str] = None

    # ----- lifecycle -------------------------------------------------------

    def reset(
        self,
        task_id: str,
        seed: int = 42,
    ) -> CustomerServiceObservation:
        """Start a new episode and return the initial observation."""
        resp = self._client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data["session_id"]
        return CustomerServiceObservation(**data["observation"])

    def step(
        self,
        action: CustomerServiceAction,
    ) -> Tuple[CustomerServiceObservation, float, bool, Dict[str, Any]]:
        """Take one step in the active episode."""
        if self._session_id is None:
            raise RuntimeError("No active session. Call reset() first.")
        resp = self._client.post(
            "/step",
            json={
                "session_id": self._session_id,
                "tool_name": action.tool_name,
                "tool_args": action.tool_args,
                "message": action.message,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        obs = CustomerServiceObservation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]

    def state(self) -> CustomerServiceState:
        """Return the current full episode state."""
        if self._session_id is None:
            raise RuntimeError("No active session. Call reset() first.")
        resp = self._client.get("/state", params={"session_id": self._session_id})
        resp.raise_for_status()
        data = resp.json()
        return CustomerServiceState(**data["state"])

    # ----- informational ---------------------------------------------------

    def get_tasks(self) -> List[TaskDefinition]:
        """Return all available task definitions."""
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        data = resp.json()
        return [TaskDefinition(**t) for t in data["tasks"]]

    def grade(self) -> GraderResponse:
        """Grade the current episode using the server-side grader."""
        if self._session_id is None:
            raise RuntimeError("No active session. Call reset() first.")
        resp = self._client.post(
            "/grader",
            json={"session_id": self._session_id},
        )
        resp.raise_for_status()
        return GraderResponse(**resp.json())

    # ----- properties ------------------------------------------------------

    @property
    def session_id(self) -> Optional[str]:
        """The active session ID, or ``None`` if no episode is running."""
        return self._session_id

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "CustomerServiceClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
