"""
Baseline inference script using OpenAI function calling.

Runs a full episode for each of the three tasks, prints per-task scores,
and outputs an overall average.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py

The script can run against a live server (default http://localhost:7860) or
directly against the environment in-process when --local is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from models import CustomerServiceAction, CustomerServiceObservation
from server.environment import CustomerServiceEnvironment
from server.tools import get_openai_tool_definitions

TASK_IDS = [
    "order_status_inquiry",
    "return_refund_processing",
    "complex_complaint_resolution",
]


def _build_system_prompt() -> str:
    return (
        "You are a helpful customer service agent. You have access to tools "
        "for looking up orders, customers, checking return policies, "
        "initiating refunds, sending notifications, escalating to humans, "
        "and applying compensation. Use the tools to resolve the customer's "
        "issue step by step. Be efficient and follow company policies."
    )


def run_baseline_local(
    task_id: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Run a full episode locally (in-process) using OpenAI function calling.

    Returns a dict with task_id, score, steps, and the full trajectory.
    """
    from openai import OpenAI

    client = OpenAI()
    env = CustomerServiceEnvironment()
    obs = env.reset(task_id=task_id)

    tools = get_openai_tool_definitions()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": obs.message},
    ]

    trajectory: List[Dict[str, Any]] = []
    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # If the model doesn't call a tool, we're done or it wants to respond
        if not msg.tool_calls:
            # Send a final notification as a fallback
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            action = CustomerServiceAction(tool_name=fn_name, tool_args=fn_args)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            trajectory.append({
                "tool_name": fn_name,
                "tool_args": fn_args,
                "tool_result": obs.tool_result,
                "reward": reward,
                "step": steps,
            })

            # Add assistant message with tool call
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

            # Add tool response
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(obs.tool_result or {}),
            })

            if done:
                break

    final_reward = max(0.001, min(0.999, total_reward))
    return {
        "task_id": task_id,
        "score": final_reward,
        "steps": steps,
        "trajectory": trajectory,
    }


def run_baseline_http(
    task_id: str,
    base_url: str = "http://localhost:7860",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Run a full episode via the HTTP API using OpenAI function calling."""
    from openai import OpenAI

    from client import CustomerServiceClient

    openai_client = OpenAI()

    with CustomerServiceClient(base_url=base_url) as env_client:
        obs = env_client.reset(task_id=task_id)

        tools = get_openai_tool_definitions()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": obs.message},
        ]

        trajectory: List[Dict[str, Any]] = []
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                action = CustomerServiceAction(tool_name=fn_name, tool_args=fn_args)
                obs, reward, done, info = env_client.step(action)
                total_reward += reward
                steps += 1

                trajectory.append({
                    "tool_name": fn_name,
                    "tool_args": fn_args,
                    "tool_result": obs.tool_result,
                    "reward": reward,
                    "step": steps,
                })

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
                    "content": json.dumps(obs.tool_result or {}),
                })

                if done:
                    break

    final_reward = max(0.001, min(0.999, total_reward))
    return {
        "task_id": task_id,
        "score": final_reward,
        "steps": steps,
        "trajectory": trajectory,
    }


def main() -> None:
    """Entry point for the baseline script."""
    parser = argparse.ArgumentParser(description="Customer Service OpenEnv Baseline")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run in-process instead of via HTTP API.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the running server (ignored with --local).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "Please set it before running the baseline:\n"
            "  export OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)

    run_fn = run_baseline_local if args.local else run_baseline_http

    results = []
    for task_id in TASK_IDS:
        print(f"Running task: {task_id} ...")
        result = run_fn(task_id, model=args.model) if args.local else run_fn(
            task_id, base_url=args.base_url, model=args.model
        )
        results.append(result)
        print(f"  Task: {task_id} | Score: {result['score']:.2f} | Steps: {result['steps']}")

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\nOverall baseline score: {avg:.2f}")


if __name__ == "__main__":
    main()
