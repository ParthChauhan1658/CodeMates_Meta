# CodeMates Meta - Customer Service AI Agent

An AI-powered customer service environment for evaluating and benchmarking LLM agents on real-world support scenarios. Built for the Meta CodeMates hackathon using the OpenEnv specification.

## Overview

This project implements a comprehensive customer service simulation where AI agents handle support queries through multi-step tool-calling. The environment tests an agent's ability to:

- Execute tools in the correct sequence
- Follow company policies and procedures
- Handle complex multi-issue scenarios
- Communicate effectively with customers

## Project Structure

```
CodeMates_Meta/
├── customer-service-env/          # Main environment implementation
│   ├── server/                    # FastAPI backend
│   │   ├── app.py                # API endpoints
│   │   ├── environment.py        # Core environment logic
│   │   ├── tools.py              # Customer service tools
│   │   ├── tasks.py              # Task definitions
│   │   ├── graders.py            # Evaluation logic
│   │   └── fixtures.py           # Test data
│   ├── tests/                     # Comprehensive test suite
│   ├── baseline.py                # Reference implementation
│   ├── inference.py               # LLM inference script
│   ├── client.py                  # HTTP client wrapper
│   ├── models.py                  # Data models
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Container configuration
│   └── README.md                  # Detailed documentation
└── README.md                      # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerized deployment)
- OpenAI API key or compatible LLM endpoint

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ParthChauhan1658/CodeMates_Meta.git
cd CodeMates_Meta
```

2. Set up virtual environment:
```bash
cd customer-service-env
python -m venv .venv

# On Windows (bash)
source .venv/Scripts/activate

# On Windows (PowerShell)
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Environment

#### Option 1: Local Server

```bash
cd customer-service-env
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Access the API at `http://localhost:7860` and interactive docs at `http://localhost:7860/docs`

#### Option 2: Docker

```bash
cd customer-service-env
docker build -t customer-service-env .
docker run -p 7860:7860 customer-service-env
```

#### Option 3: Hugging Face Spaces

The environment is deployed at: `https://parthchauhan3-customer-service-env.hf.space`

### Running Inference

Set up your environment variables:

```bash
# On bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here

# On PowerShell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="your_token_here"
```

Run inference:

```bash
cd customer-service-env
python inference.py
```

### Running Tests

```bash
cd customer-service-env
pytest tests/ -v
```

## Tasks

The environment includes three tasks of increasing difficulty:

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| Order Status Inquiry | Easy | 5 | Look up order and notify customer |
| Return & Refund Processing | Medium | 10 | Verify eligibility and process refund |
| Complex Complaint Resolution | Hard | 15 | Handle multiple issues with compensation |

## Available Tools

The agent has access to 7 customer service tools:

1. `lookup_order` - Retrieve order details
2. `lookup_customer` - Get customer profile
3. `check_return_policy` - Verify return eligibility
4. `initiate_refund` - Process refunds
5. `send_notification` - Message customers
6. `escalate_to_human` - Escalate complex issues
7. `apply_compensation` - Apply credits/discounts

## API Endpoints

- `GET /health` - Health check
- `POST /reset` - Start new episode
- `POST /step` - Execute action
- `GET /state` - Get episode state
- `GET /tasks` - List all tasks
- `POST /grader` - Evaluate performance
- `GET /baseline` - Reference scores
- `GET /docs` - API documentation

## Evaluation

Agents are scored on:
- Tool selection accuracy
- Policy compliance
- Efficiency (fewer steps = better)
- Task completion

Rewards range from 0.0 to 1.0, with baseline scores around 0.75 average.

## Deployment

### Upload to Hugging Face

```bash
cd customer-service-env

# Login to Hugging Face
huggingface-cli login

# Upload files
python upload_to_hf.py
```

## Technologies Used

- FastAPI - Web framework
- Pydantic - Data validation
- OpenAI - LLM integration
- httpx - HTTP client
- pytest - Testing
- Docker - Containerization
- Hugging Face Spaces - Deployment

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT

## Authors

- Parth Chauhan (@ParthChauhan3)

## Acknowledgments

Built for the Meta CodeMates hackathon using the OpenEnv specification.

---

For detailed documentation on the environment implementation, see [customer-service-env/README.md](customer-service-env/README.md)
