---
title: Customer Service OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Customer Service OpenEnv

A real-world customer service environment where AI agents resolve support queries using multi-step tool-calling. Built to the [OpenEnv](https://github.com/openenv) specification.

## Motivation

Customer service is one of the most common and impactful applications of AI agents. This environment provides a controlled, reproducible benchmark for evaluating how well an agent can:

- Use the right tools in the right order
- Follow company policies (e.g. checking return eligibility before issuing refunds)
- Handle increasingly complex, multi-issue scenarios
- Communicate resolutions clearly to customers

## Tasks

| ID | Name | Difficulty | Max Steps | Description |
|----|------|-----------|-----------|-------------|
| `order_status_inquiry` | Order Status Inquiry | Easy | 5 | Look up an order and notify the customer of its status. |
| `return_refund_processing` | Return & Refund Processing | Medium | 10 | Verify order, check return policy, initiate refund if eligible, notify customer. |
| `complex_complaint_resolution` | Complex Complaint Resolution | Hard | 15 | Resolve a wrong-item delivery AND duplicate billing charge with refund, compensation, and notification. |

## Action Space

Each step, the agent submits an **Action** with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | Name of the tool to call (one of the 7 available tools). |
| `tool_args` | `dict` | Keyword arguments for the tool. |
| `message` | `str` (optional) | Free-text context from the agent. |

### Available Tools

1. **lookup_order(order_id: str)** -- Retrieve order details by ID.
2. **lookup_customer(customer_id: str)** -- Retrieve customer profile by ID.
3. **check_return_policy(product_category: str, order_date: str)** -- Check if a product is eligible for return.
4. **initiate_refund(order_id: str, amount: float)** -- Initiate a refund (must check policy first).
5. **send_notification(customer_id: str, message: str)** -- Send a message to the customer.
6. **escalate_to_human(reason: str)** -- Escalate to a human agent (penalised on easy/medium tasks).
7. **apply_compensation(customer_id: str, comp_type: str, amount: float)** -- Apply store credit, discount, or refund bonus.

## Observation Space

Each step returns an **Observation** with:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task identifier. |
| `customer_message` | `str` | The original customer query. |
| `tool_result` | `dict` or `null` | Structured result from the last tool call. |
| `available_tools` | `list[str]` | Names of callable tools. |
| `steps_remaining` | `int` | Steps left before the episode is terminated. |
| `reward_components` | `dict[str, float]` | Labelled reward breakdown for this step. |
| `message` | `str` | Human-readable step outcome summary. |
| `done` | `bool` | Whether the episode has ended. |
| `reward` | `float` or `null` | Step reward delta. |

## Reward Function

Rewards are awarded incrementally per step:

- **Correct tool call**: +0.10 to +0.40 depending on tool importance for the task.
- **Policy compliance**: Implicit in tool reward (checking policy before refund).
- **Penalties**: -0.10 for wrong arguments, -0.20 for policy violations, -0.30 for unnecessary escalation.
- **Final reward**: Clamped to [0.0, 1.0] at episode end.

## Baseline Scores

| Task | gpt-4o-mini (reference) |
|------|------------------------|
| Order Status Inquiry | ~0.85 |
| Return & Refund Processing | ~0.75 |
| Complex Complaint Resolution | ~0.65 |
| **Average** | **~0.75** |

## Setup

### Local Development

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t customer-service-env .
docker run -p 7860:7860 customer-service-env
```

### Hugging Face Spaces

Push this repository to a Hugging Face Space with Docker SDK. The Dockerfile is pre-configured for port 7860.

### Running Tests

```bash
pytest tests/ -v
```

### Running the Baseline

```bash
export OPENAI_API_KEY=sk-...

# Against a running server
python baseline.py

# In-process (no server needed)
python baseline.py --local
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check. |
| POST | `/reset` | Start a new episode. Body: `{"task_id": "...", "seed": 42}` |
| POST | `/step` | Take one step. Body: `{"session_id": "...", "tool_name": "...", "tool_args": {...}}` |
| GET | `/state?session_id=...` | Get full episode state. |
| GET | `/tasks` | List all tasks with metadata and action schema. |
| POST | `/grader` | Grade an episode. Body: `{"session_id": "..."}` or `{"task_id": "...", "trajectory": [...]}` |
| GET | `/baseline` | Baseline agent metadata and reference scores. |
| GET | `/docs` | Auto-generated OpenAPI documentation. |

## Architecture

```
customer-service-env/
  models.py          -- Pydantic v2 data models (Action, Observation, State)
  client.py          -- HTTP client wrapper for the REST API
  baseline.py        -- OpenAI function-calling baseline script
  server/
    app.py           -- FastAPI application with all endpoints
    environment.py   -- Core environment (reset/step/state lifecycle)
    tools.py         -- 7 simulated tools + ToolRegistry + OpenAI schemas
    tasks.py         -- Task definitions and registry
    graders.py       -- Post-hoc trajectory graders
    fixtures.py      -- Deterministic test data for all 3 tasks
  tests/
    test_models.py   -- Pydantic model validation tests
    test_tools.py    -- Tool unit tests
    test_reward.py   -- Reward computation tests
    test_graders.py  -- Grader unit tests
    test_environment.py -- Full episode integration tests
    test_api.py      -- FastAPI endpoint integration tests
```

<<<<<<< HEAD
<<<<<<< HEAD
=======
## License

MIT
>>>>>>> 9cd60de (stage 2)
=======
>>>>>>> c766c1461ec8362e7146cf180e15166dbcbc89b9
