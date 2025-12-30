# 🛢️ Oil Price Prediction Agent

A multi-agent system powered by **Google ADK** (Agent Development Kit) that monitors oil market factors, analyzes trends, and predicts oil price movements with clear reasoning.

Built following patterns from the **Kaggle 5-Day GenAI Intensive Course**.

## 📋 Project Overview

**Platform**: Kaggle Notebook / Local Python  
**Framework**: Google ADK (Agent Development Kit)  
**Model**: `gemini-2.5-flash-lite`  
**Submission**: 5-Day Kaggle Course Capstone Project

### Problem Statement
Oil price volatility affects global markets, energy companies, and consumers. Manually tracking all influencing factors (geopolitics, supply/demand, economic indicators, market sentiment) is time-consuming and often reactive.

### Solution
A multi-agent system that:
1. Continuously monitors oil market factors across 4 domains using `google_search`
2. Analyzes trends using `BuiltInCodeExecutor` for statistical methods
3. Generates price predictions with confidence levels via `FunctionTool`
4. Learns from prediction accuracy using `LoopAgent` patterns
5. Provides actionable insights with clear reasoning

## 🏗️ Architecture (Google ADK Patterns)

This project implements patterns from the Kaggle 5-Day GenAI Intensive Course:

- **Day 1**: Agent architectures (ParallelAgent, SequentialAgent, LoopAgent)
- **Day 2**: Tools (google_search, FunctionTool, BuiltInCodeExecutor)
- **Day 3**: Sessions & Memory (InMemorySessionService, InMemoryMemoryService)
- **Day 4**: Observability (callbacks, logging)
- **Day 5**: Deployment patterns

### Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT (LlmAgent)                         │
│                    model: gemini-2.5-flash-lite                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ ParallelAgent │       │SequentialAgent│       │   LoopAgent   │
│   Research    │       │   Analysis    │       │  Evaluation   │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
   ┌────┼────┐             ┌────┼────┐                  │
   │    │    │             │    │    │                  │
   ▼    ▼    ▼             ▼    ▼    ▼                  ▼
┌────┐┌────┐┌────┐     ┌────┐┌────┐┌────┐         ┌─────────┐
│Geo ││S&D ││Econ│     │Aggr││Trnd││Pred│         │Evaluator│
│    ││    ││    │     │    ││    ││    │         │(2 iter) │
└────┘└────┘└────┘     └────┘└────┘└────┘         └─────────┘
  │     │     │           │     │     │                │
  └──┬──┴──┬──┘           └──┬──┴──┬──┘                │
     │     │                 │     │                   │
[google_search]        [FunctionTool]           [FunctionTool]
                       [CodeExecutor]           [update_weights]
```

### ADK Components Used

| Component | Import | Usage |
|-----------|--------|-------|
| `Agent` | `google.adk.agents` | Individual specialized agents with instructions |
| `ParallelAgent` | `google.adk.agents` | Concurrent research execution (4 agents) |
| `SequentialAgent` | `google.adk.agents` | Ordered analysis pipeline (3 stages) |
| `LoopAgent` | `google.adk.agents` | Iterative evaluation (max 2 iterations) |
| `Gemini` | `google.adk.models.google_llm` | Model wrapper with retry options |
| `google_search` | `google.adk.tools` | Real-time market data retrieval |
| `FunctionTool` | `google.adk.tools` | Python functions for state management |
| `AgentTool` | `google.adk.tools` | Agent wrapping for orchestration |
| `BuiltInCodeExecutor` | `google.adk.code_executors` | Statistical analysis |
| `InMemoryRunner` | `google.adk.runners` | Agent execution runtime |
| `InMemorySessionService` | `google.adk.sessions` | Session state management |
| `InMemoryMemoryService` | `google.adk.memory` | Long-term memory storage |
| `HttpRetryOptions` | `google.genai.types` | Robust API retry handling (429, 500, 503, 504) |
| Callbacks | `google.adk.agents.callback_context` | before/after agent callbacks |

## ✅ Key Concepts Implemented (6/6 from Course)

| Day | Concept | Implementation |
|-----|---------|---------------|
| 1 | Multi-agent System | `ParallelAgent` (4 research) + `SequentialAgent` (3 analysis) + `LoopAgent` (evaluation) |
| 2 | Built-in Tools | `google_search` for research + `BuiltInCodeExecutor` for statistics |
| 3a | Sessions | `InMemorySessionService` for conversation state |
| 3b | Memory | `InMemoryMemoryService` for prediction history |
| 4 | Observability | `before_agent_callback`, `after_agent_callback`, DEBUG logging |
| 5 | Evaluation | Accuracy tracking + learned weight adjustment via `LoopAgent` |

### Code Pattern Examples

```python
# Day 1: Agent with tools (from day-1a-from-prompt-to-action.ipynb)
research_agent = Agent(
    name="geopolitical_monitor",
    model=model,
    instruction="Monitor OPEC decisions and geopolitical events...",
    tools=[google_search]
)

# Day 1b: ParallelAgent for concurrent execution
research_parallel = ParallelAgent(
    name="research_coordinator",
    sub_agents=[geo_agent, supply_agent, econ_agent, sentiment_agent]
)

# Day 2: FunctionTool for state management
store_prediction_tool = FunctionTool(func=store_prediction)

# Day 3: Session and Memory services
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService()

# Day 4: InMemoryRunner with services
runner = InMemoryRunner(
    agent=orchestrator,
    app_name="oil-price-prediction",
    session_service=session_service,
    memory_service=memory_service,
)

# Run with async
async for event in runner.run_async(user_id, session_id, content):
    print(event.content.parts[0].text)
```

## 📁 Project Structure

```
oilprice/
├── 📓 oil_price_prediction_agent.ipynb  # Main Kaggle notebook (self-contained)
├── 🐍 main.py                           # CLI entry point with ADK runner
├── 📦 pyproject.toml                    # Project config (uv)
├── 📦 requirements.txt                  # Dependencies (pip)
│
├── 🤖 agents/                           # ADK Agent implementations
│   ├── __init__.py                      # Exports all agent factories
│   ├── research_agents.py               # ParallelAgent + 4 research agents
│   ├── analysis_agents.py               # SequentialAgent + 3 analysis agents
│   └── evaluation_agent.py              # LoopAgent + evaluator
│
├── 🧠 memory/                           # Session & Memory (Day 3)
│   ├── __init__.py
│   ├── memory_bank.py                   # Long-term prediction storage
│   └── session_manager.py               # Session context tracking
│
├── 📊 models/                           # Data Models
│   ├── __init__.py
│   └── data_models.py                   # Dataclasses for predictions
│
├── 🔧 utils/                            # Utilities (Day 4)
│   ├── __init__.py
│   ├── observability.py                 # Callbacks, logging, tracing
│   └── visualization.py                 # Charts and dashboards
│
├── ⚙️ config/                           # Configuration
│   ├── __init__.py
│   └── agent_config.py                  # Agent prompts & settings
│
└── 📚 docs/                             # Documentation
    └── PROJECT_GUIDE.md                 # Detailed guide
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### 1. Install uv (if not already installed)
```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Up the Project
```bash
# Clone or navigate to the project directory
cd oilprice

# Create virtual environment and install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### 3. Set Up API Key
```bash
# Option 1: Environment variable
export GOOGLE_API_KEY="your-api-key"

# Option 2: .env file
echo "GOOGLE_API_KEY=your-api-key" > .env

# Option 3: Kaggle Secrets (for Kaggle notebooks)
# Add GOOGLE_API_KEY to your Kaggle Secrets
```

### 4. Run the Agent
```bash
# Using uv (recommended)
uv run python main.py

# Or start Jupyter
uv run jupyter lab

# Or open the notebook directly
uv run jupyter notebook oil_price_prediction_agent.ipynb
```

### Alternative: Using pip
```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Running on Kaggle

1. Upload `oil_price_prediction_agent.ipynb` to Kaggle
2. Add `GOOGLE_API_KEY` to Kaggle Secrets
3. Enable Internet access in notebook settings
4. Run all cells

The notebook auto-detects Kaggle environment:
```python
try:
    from kaggle_secrets import UserSecretsClient
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
except:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
```

## 📊 Sample Output

```
🛢️ OIL PRICE PREDICTION REPORT
═══════════════════════════════════════════════════════════

📅 Date: November 27, 2025

📈 PREDICTIONS (7-Day Horizon)
───────────────────────────────────────────────────────────
WTI Crude Oil:
  Current Price:    $76.20
  Predicted Price:  $78.50 (+2.3%)
  Price Range:      $77.00 - $80.00
  Confidence:       75%

Brent Crude Oil:
  Current Price:    $80.50
  Predicted Price:  $82.20 (+1.8%)
  Price Range:      $80.50 - $84.00
  Confidence:       72%

🔴 BULLISH FACTORS
───────────────────────────────────────────────────────────
1. [HIGH] OPEC+ Production Cut Extension
   Saudi Arabia extending 1M bpd cut through Q1 2026

2. [MEDIUM] Strong Asian Demand
   China manufacturing PMI up 3.2%

🟢 BEARISH FACTORS
───────────────────────────────────────────────────────────
1. [MEDIUM] USD Strengthening
   DXY index up 1.5% this week

📊 RECOMMENDATION: BUY
───────────────────────────────────────────────────────────
Strong fundamentals outweigh technical concerns.
```

## 📈 Performance Metrics

The evaluation agent tracks these metrics via `FunctionTool`:

| Metric | Description | Target |
|--------|-------------|--------|
| Mean Absolute Error | Average $ deviation from actual | < $3.00 |
| Directional Accuracy | Correct up/down predictions | > 70% |
| Confidence Calibration | Predicted vs actual confidence | > 75% |

### Learning Loop (LoopAgent)

The `LoopAgent` runs the evaluation agent up to 2 iterations:
1. **Iteration 1**: Calculate accuracy, identify best-performing factors
2. **Iteration 2**: Adjust weights, validate improvements

```python
evaluation_loop = LoopAgent(
    name="evaluation_loop",
    sub_agents=[evaluation_agent],
    max_iterations=2
)
```

## 🎥 Video Demo

[Link to video walkthrough - Add your video link here]

## 📝 Submission

This project is submitted for the **Kaggle 5-Day GenAI Intensive Course Capstone**.

**Deadline**: December 1, 2025 11:59 AM Pacific Time

## 📚 References

- [Kaggle 5-Day GenAI Intensive Course](https://www.kaggle.com/learn-guide/5-day-genai-intensive-course)
- [Google ADK Documentation](https://google.github.io/adk-docs/)

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- Google ADK Team
- Kaggle Learn Team
- 5-Day GenAI Intensive Course Instructors
