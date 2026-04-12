"""
Inference script for Customer Service OpenEnv.

Required environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    (Optional) Docker image name to spin up the environment.

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=hf_...
    export LOCAL_IMAGE_NAME=customer-service-env:latest  # optional
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Environment server config
ENV_PORT = 7860
ENV_BASE_URL = f"http://localhost:{ENV_PORT}"
HF_SPACE_URL = "https://parthchauhan3-customer-service-env.hf.space"

# Episode config
MAX_STEPS = 15
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

TASK_IDS = [
    "order_status_inquiry",
    "return_refund_processing",
    "complex_complaint_resolution",
]

BENCHMARK = "customer-service-env"

# ---------------------------------------------------------------------------
# Structured logging — [START] / [STEP] / [END]
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error.replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    rewards: List[float],
) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Docker container lifecycle
# ---------------------------------------------------------------------------

_container_id: Optional[str] = None


def start_docker_container(image: str) -> str:
    """Spin up the environment docker container. Returns the base URL."""
    global _container_id

    result = subprocess.run(
        ["docker", "run", "-d", "-p", f"{ENV_PORT}:{ENV_PORT}", image],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(1)

    _container_id = result.stdout.strip()

    # Wait for server to be ready
    _wait_for_server(ENV_BASE_URL)
    return ENV_BASE_URL


def stop_docker_container() -> None:
    """Stop and remove the docker container."""
    global _container_id
    if _container_id:
        subprocess.run(["docker", "stop", _container_id], capture_output=True)
        subprocess.run(["docker", "rm", _container_id], capture_output=True)
        _container_id = None


def _wait_for_server(base_url: str, timeout: int = 60) -> None:
    """Poll the health endpoint until the server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=3)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(http: httpx.Client, task_id: str) -> Dict[str, Any]:
    resp = http.post("/reset", json={"task_id": task_id, "seed": 42})
    resp.raise_for_status()
    return resp.json()


def env_step(http: httpx.Client, session_id: str, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
    resp = http.post("/step", json={
        "session_id": session_id,
        "tool_name": tool_name,
        "tool_args": tool_args,
    })
    resp.raise_for_status()
    return resp.json()


def env_grader(http: httpx.Client, session_id: str) -> Dict[str, Any]:
    resp = http.post("/grader", json={"session_id": session_id})
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tool definitions for OpenAI function calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up order details by order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The order ID to look up."}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up customer profile and order history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "The customer ID."}
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_policy",
            "description": "Check return policy eligibility for a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_category": {"type": "string"},
                    "order_date": {"type": "string", "description": "Order date in YYYY-MM-DD format."},
                },
                "required": ["product_category", "order_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "initiate_refund",
            "description": "Initiate a refund for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Refund amount in dollars."},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Send a notification message to a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "message": {"type": "string", "description": "The message to send."},
                },
                "required": ["customer_id", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the issue to a human agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Reason for escalation."}
                },
                "required": ["reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_compensation",
            "description": "Apply compensation to a customer account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "type": {"type": "string", "enum": ["credit", "discount", "refund", "voucher"]},
                    "amount": {"type": "number"},
                },
                "required": ["customer_id", "type", "amount"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def get_model_response(
    client: OpenAI,
    messages: List[Dict[str, Any]],
) -> Any:
    try:
        return client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single task runner
# ---------------------------------------------------------------------------

async def run_task(
    task_id: str,
    llm_client: OpenAI,
    base_url: str,
) -> Dict[str, Any]:
    """Run a full episode for one task. Returns result dict."""

    rewards: List[float] = []
    steps_taken = 0
    task_score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    with httpx.Client(base_url=base_url, timeout=30) as http:
        try:
            # Reset the environment
            reset_data = env_reset(http, task_id)
            session_id = reset_data["session_id"]
            obs = reset_data["observation"]
            done = obs.get("done", False)

            system_prompt = (
                "You are a helpful customer service agent. Use the available tools "
                "to resolve the customer's issue step by step. Be efficient and "
                "follow company policies."
            )
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": obs.get("customer_message", "")},
            ]

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                response = get_model_response(llm_client, messages)

                if response is None:
                    log_step(step=step, action="[error]", reward=0.0, done=True, error="LLM call failed")
                    break

                msg = response.choices[0].message

                if not msg.tool_calls:
                    log_step(step=step, action="[no_tool_call]", reward=0.0, done=True, error=None)
                    break

                for tool_call in msg.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    step_data = env_step(http, session_id, fn_name, fn_args)
                    raw_reward = float(step_data.get("reward", 0.0))
                    reward = raw_reward
                    done = step_data.get("done", False)
                    tool_result = step_data.get("observation", {}).get("tool_result") or {}
                    error = step_data.get("info", {}).get("error")
                    if error is None and tool_result.get("status") == "error":
                        error = tool_result.get("message")

                    rewards.append(reward)
                    steps_taken = step

                    compact_args = json.dumps(fn_args, separators=(",", ":"))
                    action_str = f"{fn_name}({compact_args})"
                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    # Add to message history
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": tool_call.function.arguments,
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(step_data.get("observation", {}).get("tool_result") or {}),
                    })

                    if done:
                        break

            # Compute final score
            total_reward = sum(rewards)
            task_score = min(max(total_reward / MAX_TOTAL_REWARD, 0.001), 0.999)
            success = task_score >= SUCCESS_SCORE_THRESHOLD

        except Exception:
            pass

        finally:
            log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "score": task_score,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")

    # Build OpenAI client using the required variables
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    # Determine environment base URL
    if LOCAL_IMAGE_NAME:
        base_url = start_docker_container(LOCAL_IMAGE_NAME)
    else:
        base_url = HF_SPACE_URL

    all_results = []
    try:
        for task_id in TASK_IDS:
            result = await run_task(task_id, llm_client, base_url)
            all_results.append(result)

    finally:
        if LOCAL_IMAGE_NAME:
            stop_docker_container()

if __name__ == "__main__":
    asyncio.run(main())
