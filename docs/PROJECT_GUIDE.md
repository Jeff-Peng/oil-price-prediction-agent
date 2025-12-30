# 🛢️ Oil Price Prediction Agent - Complete Project Guide

A comprehensive guide for understanding, running, and extending the Oil Price Prediction Agent built with **Google ADK** (Agent Development Kit).

This project follows patterns from the **Kaggle 5-Day GenAI Intensive Course**.

---

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Key Concepts (from Course)](#key-concepts)
5. [How It Works](#how-it-works)
6. [Running Locally](#running-locally)
7. [Running on Kaggle](#running-on-kaggle)
8. [Understanding the Code](#understanding-the-code)
9. [Extending the Project](#extending-the-project)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is this project?

This is a **multi-agent AI system** that predicts oil prices (WTI and Brent crude) by:
- Gathering real-time market intelligence from multiple sources
- Analyzing trends using statistical methods
- Generating predictions with confidence levels
- Learning from past prediction accuracy

### Why was it built?

This project was created as a **Kaggle 5-Day GenAI Intensive Course Capstone Project** to demonstrate key concepts from each day of the course:

| Day | Concept | ADK Components Used |
|-----|---------|--------------------|
| **Day 1** | Multi-Agent Architectures | `Agent`, `ParallelAgent`, `SequentialAgent`, `LoopAgent` |
| **Day 2** | Built-in Tools | `google_search`, `FunctionTool`, `BuiltInCodeExecutor` |
| **Day 3a** | Sessions | `InMemorySessionService` |
| **Day 3b** | Memory | `InMemoryMemoryService` |
| **Day 4** | Observability | Callbacks (`before_agent_callback`, `after_agent_callback`), Logging |
| **Day 5** | Evaluation & Deployment | Accuracy tracking, weight adjustment via `LoopAgent` |

### Who is this for?

- Students learning about AI agents
- Developers exploring Google ADK
- Anyone interested in financial prediction systems
- Kaggle course participants

---

## Architecture Overview

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                                     │
│                    "Predict oil prices"                                  │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: RESEARCH (Parallel)                          │
│                                                                          │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│   │ Geopolitical │  │Supply/Demand │  │  Economic    │  │  Sentiment   ││
│   │   Monitor    │  │   Tracker    │  │  Indicators  │  │   Analyzer   ││
│   │              │  │              │  │              │  │              ││
│   │ • OPEC news  │  │ • Production │  │ • USD index  │  │ • Futures    ││
│   │ • Sanctions  │  │ • Inventory  │  │ • Inflation  │  │ • Headlines  ││
│   │ • Conflicts  │  │ • Demand     │  │ • Interest   │  │ • Technical  ││
│   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘│
│          │                 │                 │                 │        │
│          └────────────────┼─────────────────┼─────────────────┘        │
│                           │                 │                           │
└───────────────────────────┼─────────────────┼───────────────────────────┘
                            │                 │
                            ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: ANALYSIS (Sequential)                        │
│                                                                          │
│   ┌──────────────────┐                                                  │
│   │  Data Aggregator │ ──► Combines all research findings               │
│   └────────┬─────────┘     Weights by importance                        │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │  Trend Analyzer  │ ──► Statistical analysis                         │
│   └────────┬─────────┘     Moving averages, support/resistance          │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │  Price Predictor │ ──► Final prediction with reasoning              │
│   └────────┬─────────┘     Confidence scores                            │
│            │                                                             │
└────────────┼────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: EVALUATION (Loop)                            │
│                                                                          │
│   ┌──────────────────────┐                                              │
│   │ Performance Evaluator│ ──► Compare predictions to actual prices     │
│   │                      │     Adjust factor weights                    │
│   │   Learning Loop      │     Improve over time                        │
│   └──────────────────────┘                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                           │
│                                                                          │
│   📊 Price Prediction Report                                            │
│   • WTI: $76.20 → $78.50 (+2.3%) | Confidence: 75%                     │
│   • Brent: $80.50 → $82.20 (+1.8%) | Confidence: 72%                   │
│   • Recommendation: BUY                                                  │
│   • Key Factors: OPEC cuts, Strong Asian demand                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Types (from Day 1b)

Google ADK provides three composite agent types:

| Agent Type | ADK Class | Execution | Purpose | Our Usage |
|------------|-----------|-----------|---------|----------|
| **Parallel** | `ParallelAgent` | All at once | Gather independent data | 4 research agents |
| **Sequential** | `SequentialAgent` | One after another | Process data in stages | 3 analysis agents |
| **Loop** | `LoopAgent` | Repeating (max N iterations) | Learn and improve | Evaluation (2 iterations) |

```python
# From day-1b-agent-architectures.ipynb
from google.adk.agents import Agent, ParallelAgent, SequentialAgent, LoopAgent

parallel = ParallelAgent(name="research", sub_agents=[a1, a2, a3, a4])
sequential = SequentialAgent(name="analysis", sub_agents=[agg, trend, pred])
loop = LoopAgent(name="eval", sub_agents=[evaluator], max_iterations=2)
```

---

## Project Structure

```
oilprice/
│
├── 📓 oil_price_prediction_agent.ipynb   # Main Kaggle notebook
├── 🐍 main.py                             # CLI entry point
├── 📋 README.md                           # Quick start guide
├── 📦 pyproject.toml                      # Project config (uv)
├── 📦 requirements.txt                    # Dependencies (pip)
│
├── 🤖 agents/                             # AI Agents
│   ├── __init__.py
│   ├── research_agents.py                 # 4 parallel research agents
│   ├── analysis_agents.py                 # 3 sequential analysis agents
│   └── evaluation_agent.py                # Learning loop agent
│
├── ⚙️ config/                             # Configuration
│   ├── __init__.py
│   └── agent_config.py                    # Agent prompts & settings
│
├── 🧠 memory/                             # Memory & Sessions
│   ├── __init__.py
│   ├── memory_bank.py                     # Long-term storage
│   └── session_manager.py                 # Session tracking
│
├── 📊 models/                             # Data Models
│   ├── __init__.py
│   └── data_models.py                     # Dataclasses & types
│
├── 🔧 utils/                              # Utilities
│   ├── __init__.py
│   ├── observability.py                   # Logging, tracing, metrics
│   └── visualization.py                   # Charts & dashboards
│
└── 📚 docs/                               # Documentation
    └── PROJECT_GUIDE.md                   # This file!
```

### File Descriptions

#### Core Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `main.py` | CLI entry point | `main()`, `run_prediction_pipeline()` |
| `oil_price_prediction_agent.ipynb` | Kaggle notebook | Complete self-contained implementation |

#### Agents (`agents/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `research_agents.py` | Gather market intelligence | `GeopoliticalMonitor`, `SupplyDemandTracker`, `EconomicIndicatorMonitor`, `MarketSentimentAnalyzer` |
| `analysis_agents.py` | Process and analyze data | `DataAggregator`, `TrendAnalyzer`, `PricePredictor` |
| `evaluation_agent.py` | Learn from results | `PerformanceEvaluator` |

#### Memory (`memory/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `memory_bank.py` | Store predictions & learn | `MemoryBank` |
| `session_manager.py` | Track analysis sessions | `SessionManager`, `SessionContext` |

#### Models (`models/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `data_models.py` | Data structures | `PricePrediction`, `AnalysisReport`, `TrendAnalysis`, `ResearchResult` |

---

## Key Concepts

### 1. Multi-Agent System (Day 1)

**What:** Multiple specialized AI agents working together instead of one general agent.

**Why:** 
- Each agent can focus on what it does best
- Parallel execution = faster results
- Easier to debug and improve individual agents

**In this project (ADK pattern from day-1b):**
```python
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

# Create individual agents with specific instructions
geopolitical_agent = Agent(
    name="geopolitical_monitor",
    model=model,
    instruction="Monitor OPEC decisions, sanctions, and conflicts...",
    tools=[google_search]
)

# 4 research agents run in PARALLEL (all at once)
research_parallel = ParallelAgent(
    name="research_coordinator",
    sub_agents=[geopolitical_agent, supply_agent, economic_agent, sentiment_agent]
)

# 3 analysis agents run SEQUENTIALLY (one after another)
analysis_sequential = SequentialAgent(
    name="analysis_pipeline",
    sub_agents=[data_aggregator, trend_analyzer, price_predictor]
)
```

### 2. Built-in Tools (Day 2)

**What:** Pre-built capabilities that agents can use.

**ADK Tool Types (from day-2a-agent-tools.ipynb):**
- `google_search`: Built-in search tool for real-time data
- `FunctionTool`: Wrap Python functions for agent use
- `BuiltInCodeExecutor`: Execute Python code for analysis
- `AgentTool`: Wrap an agent as a tool for another agent

**In this project:**
```python
from google.adk.tools import google_search, FunctionTool, AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

# Research agents use google_search directly
research_agent = Agent(
    name="researcher",
    model=model,
    tools=[google_search]  # Built-in search
)

# Wrap Python functions as FunctionTool
def store_prediction(wti: float, brent: float, confidence: float) -> str:
    prediction_history.append({"wti": wti, "brent": brent, "confidence": confidence})
    return f"Stored prediction: WTI=${wti}, Brent=${brent}"

store_tool = FunctionTool(func=store_prediction)

# Analysis agents can use code execution
analysis_agent = Agent(
    name="trend_analyzer",
    model=model,
    tools=[BuiltInCodeExecutor()]  # Run Python code
)

# Wrap ParallelAgent as a tool for orchestrator
research_tool = AgentTool(agent=research_parallel)
```

### 3. Sessions & Memory (Day 3)

**What:** Remembering context across interactions.

**ADK Services (from day-3a and day-3b):**
- `InMemorySessionService`: Track conversation state within a session
- `InMemoryMemoryService`: Store long-term memory across sessions

**Session (from day-3a-agent-sessions.ipynb):**
```python
from google.adk.sessions import InMemorySessionService

# Create session service
session_service = InMemorySessionService()

# Create a session for this prediction run
session = session_service.create_session(
    app_name="oil-price-prediction",
    user_id="oil_analyst"
)
SESSION_ID = session.id
print(f"Session created: {SESSION_ID}")
```

**Memory (from day-3b-agent-memory.ipynb):**
```python
from google.adk.memory import InMemoryMemoryService

# Create memory service
memory_service = InMemoryMemoryService()

# InMemoryRunner uses both services
runner = InMemoryRunner(
    agent=orchestrator_agent,
    app_name="oil-price-prediction",
    session_service=session_service,
    memory_service=memory_service
)
```

**Custom FunctionTools for State Management:**
```python
# Global state (managed via FunctionTools)
prediction_history: List[Dict] = []
learned_weights: Dict[str, float] = {
    "geopolitical": 0.25,
    "supply_demand": 0.30,
    "economic": 0.25,
    "sentiment": 0.20
}

def get_learned_weights() -> Dict[str, float]:
    """Get current factor weights for prediction."""
    return learned_weights

def update_weights(new_weights: Dict[str, float]) -> str:
    """Update factor weights based on evaluation."""
    global learned_weights
    learned_weights = new_weights
    return f"Weights updated: {new_weights}"
```

### 4. Context Engineering (FunctionTools)

**What:** Carefully managing what information each agent sees.

**Why:** 
- Too much context = confused agent
- Too little context = uninformed agent
- Right context = accurate predictions

**In ADK, FunctionTools manage state:**
```python
from google.adk.tools import FunctionTool

# Global state
prediction_history: List[Dict] = []
learned_weights: Dict[str, float] = {
    "geopolitical": 0.25,
    "supply_demand": 0.30,
    "economic": 0.25,
    "sentiment": 0.20
}

# FunctionTools expose specific state to agents
def get_learned_weights() -> Dict[str, float]:
    """Returns current factor weights."""
    return learned_weights

def get_recent_predictions(n: int = 5) -> List[Dict]:
    """Returns last N predictions."""
    return prediction_history[-n:]

# Wrap as tools - agents only see what they need
get_weights_tool = FunctionTool(func=get_learned_weights)
get_predictions_tool = FunctionTool(func=get_recent_predictions)

# Data Aggregator gets weights, not raw predictions
aggregator_agent = Agent(
    name="data_aggregator",
    model=model,
    tools=[get_weights_tool]  # Only needs current weights
)

# Evaluator gets both weights and history
evaluator_agent = Agent(
    name="evaluator",
    model=model,
    tools=[get_weights_tool, get_predictions_tool]  # Needs both
)
```

### 5. Observability (Day 4)

**What:** Monitoring what agents are doing.

**ADK Callbacks (from day-4a-agent-observability.ipynb):**
- `before_agent_callback`: Called before an agent runs
- `after_agent_callback`: Called after an agent completes
- `before_model_callback`: Called before model invocation
- `after_model_callback`: Called after model response

**In this project:**
```python
from google.adk.agents.callback_context import CallbackContext
import logging

# Configure DEBUG logging for full tracing
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("OilPriceAgent")

# Callback functions
def before_agent_callback(ctx: CallbackContext) -> None:
    """Log before each agent runs."""
    logger.info(f"🚀 Starting agent: {ctx.agent_name}")
    logger.debug(f"   Input: {ctx.input_text[:100]}...")

def after_agent_callback(ctx: CallbackContext) -> None:
    """Log after each agent completes."""
    logger.info(f"✅ Completed agent: {ctx.agent_name}")
    logger.debug(f"   Output length: {len(ctx.output_text)} chars")

# Note: Callbacks are passed to runner or agent configuration
# The ADK automatically calls these during execution
```

**Performance Metrics via FunctionTools:**
```python
def calculate_accuracy() -> Dict[str, float]:
    """Calculate prediction accuracy metrics."""
    if not prediction_history:
        return {"mae": 0.0, "directional_accuracy": 0.0}
    
    errors = [abs(p["predicted"] - p["actual"]) for p in prediction_history if p.get("actual")]
    mae = sum(errors) / len(errors) if errors else 0.0
    
    return {"mae": mae, "predictions_count": len(prediction_history)}

accuracy_tool = FunctionTool(func=calculate_accuracy)
```

### 6. Agent Evaluation (Day 4b & Day 5)

**What:** Measuring prediction accuracy and learning from mistakes.

**LoopAgent for Iterative Evaluation:**
```python
from google.adk.agents import LoopAgent

# Evaluation agent with learning tools
evaluation_agent = Agent(
    name="performance_evaluator",
    model=model,
    instruction="""Evaluate prediction accuracy and adjust weights.
    1. Call get_recent_predictions to review history
    2. Call calculate_accuracy to get metrics
    3. Identify best-performing factors
    4. Call update_weights with improved weights""",
    tools=[
        get_recent_predictions_tool,
        calculate_accuracy_tool,
        get_weights_tool,
        update_weights_tool
    ]
)

# Wrap in LoopAgent for iterative refinement (from day-1b)
evaluation_loop = LoopAgent(
    name="evaluation_loop",
    sub_agents=[evaluation_agent],
    max_iterations=2  # Run evaluation up to 2 times
)
```

**Accuracy Metrics:**
```python
def calculate_accuracy() -> Dict[str, float]:
    """Calculate prediction accuracy metrics."""
    mae = mean_absolute_error(predicted, actual)  # Average $ error
    rmse = root_mean_square_error(predicted, actual)  # Penalizes big errors
    directional = correct_direction / total  # Up/down accuracy
    return {"mae": mae, "rmse": rmse, "directional_accuracy": directional}

def update_weights(new_weights: Dict[str, float]) -> str:
    """Update factor weights based on performance."""
    global learned_weights
    learned_weights = new_weights
    return f"Weights updated to: {new_weights}"
```

---

## How It Works

### Step-by-Step Flow

```
1. USER: "Predict tomorrow's oil prices"
        │
        ▼
2. PIPELINE STARTS
   • Create new session
   • Start trace for observability
   • Get current prices (WTI: $76.20, Brent: $80.50)
        │
        ▼
3. RESEARCH PHASE (Parallel - ~30 seconds)
   │
   ├─► Geopolitical Monitor
   │   • Searches: "OPEC oil production news"
   │   • Finds: "Saudi Arabia extends 1M bpd cut"
   │   • Impact: HIGH, Sentiment: BULLISH
   │
   ├─► Supply/Demand Tracker
   │   • Searches: "crude oil inventory levels"
   │   • Finds: "US crude stocks down 2.1M barrels"
   │   • Impact: MEDIUM, Sentiment: BULLISH
   │
   ├─► Economic Indicators
   │   • Searches: "USD index oil correlation"
   │   • Finds: "Dollar strengthens 1.5%"
   │   • Impact: MEDIUM, Sentiment: BEARISH
   │
   └─► Sentiment Analyzer
       • Searches: "oil futures market sentiment"
       • Finds: "Bullish positioning at 3-month high"
       • Impact: MEDIUM, Sentiment: BULLISH
        │
        ▼
4. ANALYSIS PHASE (Sequential - ~20 seconds)
   │
   ├─► Data Aggregator
   │   • Combines 4 research results
   │   • Weights: Geopolitical 25%, Supply 30%, Economic 25%, Sentiment 20%
   │   • Net sentiment: BULLISH (3 bullish, 1 bearish)
   │
   ├─► Trend Analyzer
   │   • Calculates 7-day and 30-day moving averages
   │   • Identifies support ($74.50) and resistance ($79.00)
   │   • Trend: UP, Strength: 0.65
   │
   └─► Price Predictor
       • Combines all factors
       • Generates prediction with confidence
       • Creates reasoning explanation
        │
        ▼
5. OUTPUT
   ┌─────────────────────────────────────────┐
   │ 🛢️ OIL PRICE PREDICTION                │
   │                                         │
   │ WTI Crude:                             │
   │   Current:   $76.20                    │
   │   Predicted: $78.50 (+2.3%)            │
   │   Range:     $77.00 - $80.00           │
   │   Confidence: 75%                      │
   │                                         │
   │ Brent Crude:                           │
   │   Current:   $80.50                    │
   │   Predicted: $82.20 (+1.8%)            │
   │   Range:     $80.50 - $84.00           │
   │   Confidence: 72%                      │
   │                                         │
   │ 📊 RECOMMENDATION: BUY                  │
   │                                         │
   │ Key Factors:                           │
   │ • OPEC+ production cuts extended       │
   │ • US inventory drawdown                │
   │ • Strong Asian demand signals          │
   └─────────────────────────────────────────┘
        │
        ▼
6. EVALUATION (Background)
   • Store prediction in memory bank
   • Compare with actual prices (when available)
   • Update factor weights for future predictions
```

---

## Running Locally

### Prerequisites

- Python 3.13+ installed
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Google API key with Generative AI access

### Step 1: Clone/Download the Project

```bash
# If using git
git clone <repository-url>
cd oilprice

# Or download and extract the ZIP file
```

### Step 2: Install uv (Recommended)

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 3: Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Step 4: Set Up API Key

```bash
# Option 1: Environment variable
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-api-key-here"

# macOS/Linux
export GOOGLE_API_KEY="your-api-key-here"

# Option 2: Create .env file
echo "GOOGLE_API_KEY=your-api-key-here" > .env
```

### Step 5: Run the Agent

```bash
# Using uv
uv run python main.py

# Using pip/venv
python main.py
```

### Step 6: Run the Jupyter Notebook (Optional)

```bash
# Using uv
uv run jupyter lab

# Then open oil_price_prediction_agent.ipynb
```

### Expected Output

```
============================================================
🚀 Starting Oil Price Prediction Agent
============================================================
✅ Google API configured successfully

============================================================
🛢️ OIL PRICE PREDICTION AGENT
============================================================
📅 2025-11-27 10:30:45
💰 Current Prices: WTI $76.20, Brent $80.50

📡 Phase 1: Running research agents in parallel...
   ✅ Completed 4/4 research agents

🔄 Phase 2: Aggregating research data...
   ✅ Data aggregation complete

📊 Phase 3: Running sequential analysis...
   ✅ Trend analysis complete
   ✅ Price prediction complete

📈 Phase 4: Running evaluation loop...
   ✅ Evaluation complete

============================================================
🎉 PREDICTION CYCLE COMPLETE
   WTI: $76.20 → $78.50
   Brent: $80.50 → $82.20
   Recommendation: BUY
   Confidence: 75%
============================================================
```

---

## Running on Kaggle

### Step 1: Create a Kaggle Account

Go to [kaggle.com](https://www.kaggle.com) and sign up if you haven't already.

### Step 2: Get a Google API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Click "Get API Key"
3. Create a new API key
4. Copy the key

### Step 3: Add API Key to Kaggle Secrets

1. Go to your Kaggle notebook
2. Click the "Add-ons" menu (🧩 icon)
3. Select "Secrets"
4. Click "Add a new secret"
5. Name: `GOOGLE_API_KEY`
6. Value: (paste your API key)
7. Click "Save"

### Step 4: Upload the Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Click "File" → "Import Notebook"
4. Upload `oil_price_prediction_agent.ipynb`

### Step 5: Run the Notebook

1. Make sure GPU/TPU is NOT required (CPU is fine)
2. Make sure Internet is ENABLED (needed for Google Search)
3. Click "Run All" (▶️▶️)
4. Wait for all cells to complete (~2-3 minutes)

### Kaggle-Specific Code

The notebook automatically detects Kaggle environment:

```python
# This code runs automatically in the notebook
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
    print("✅ API key loaded from Kaggle Secrets")
except:
    # Fall back to environment variable for local dev
    from dotenv import load_dotenv
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
```

---

## Understanding the Code

### Data Models (`models/data_models.py`)

The project uses Python dataclasses for type safety:

```python
@dataclass
class CrudeOilPrediction:
    """Prediction for a single crude oil type"""
    crude_type: str              # "WTI" or "Brent"
    current_price: float         # Current price in USD
    predicted_price: float       # Predicted price
    predicted_change_pct: float  # Percent change
    predicted_range: Tuple[float, float]  # Low-high range
    confidence: float            # 0.0 to 1.0

@dataclass
class PricePrediction:
    """Complete price prediction"""
    prediction_id: str
    timestamp: datetime
    wti_crude: CrudeOilPrediction
    brent_crude: CrudeOilPrediction
    overall_confidence: float
    recommendation: Recommendation  # BUY, SELL, or HOLD
```

### Research Agent Example (ADK Pattern)

```python
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types

# Configure model with retry options (from class samples)
HTTP_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

model = Gemini(
    model="gemini-2.5-flash-lite",
    http_options=types.HttpOptions(retry_options=HTTP_RETRY_OPTIONS)
)

# Create agent with google_search tool (from day-1a)
geopolitical_agent = Agent(
    name="geopolitical_monitor",
    model=model,
    description="Monitors geopolitical events affecting oil markets",
    instruction="""You are a geopolitical analyst specializing in oil markets.
Your role is to identify and analyze geopolitical events that impact oil prices.

Use google_search to find current news about:
- OPEC+ production decisions and compliance
- Middle East tensions and supply disruptions
- Sanctions and trade policies affecting oil
- Political instability in major oil-producing regions

For each event found, assess:
- Impact level (HIGH/MEDIUM/LOW)
- Sentiment for oil prices (BULLISH/BEARISH/NEUTRAL)
- Time horizon of impact

Provide your analysis in a structured format.""",
    tools=[google_search]  # Built-in ADK tool
)
```

### State Management via FunctionTools

```python
from google.adk.tools import FunctionTool
from typing import Dict, List

# Global state (managed by FunctionTools)
prediction_history: List[Dict] = []
learned_weights: Dict[str, float] = {
    "geopolitical": 0.25,
    "supply_demand": 0.30,
    "economic": 0.25,
    "sentiment": 0.20
}

def store_prediction(
    wti_current: float,
    wti_predicted: float,
    brent_current: float,
    brent_predicted: float,
    confidence: float,
    recommendation: str,
    reasoning: str
) -> str:
    """Store a prediction for future evaluation."""
    prediction_history.append({
        "timestamp": datetime.now().isoformat(),
        "wti_current": wti_current,
        "wti_predicted": wti_predicted,
        "brent_current": brent_current,
        "brent_predicted": brent_predicted,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "actual_wti": None,  # Filled in later
        "actual_brent": None
    })
    return f"Prediction stored. History count: {len(prediction_history)}"

def update_weights(new_weights: Dict[str, float]) -> str:
    """Update factor weights based on evaluation."""
    global learned_weights
    learned_weights = new_weights
    return f"Weights updated: {new_weights}"

# Wrap as FunctionTools for agents
store_prediction_tool = FunctionTool(func=store_prediction)
update_weights_tool = FunctionTool(func=update_weights)
```

---

## Extending the Project

### Adding a New Research Agent

1. **Create the agent class** in `agents/research_agents.py`:

```python
class TechnicalAnalysisMonitor:
    """Monitors technical indicators"""
    
    def __init__(self):
        self.name = "technical_analysis"
        self.search_topics = [
            "oil price RSI indicator",
            "crude oil MACD signals",
            "oil futures open interest"
        ]
    
    async def research(self, model=None) -> ResearchResult:
        # Implementation here
        pass
```

2. **Add configuration** in `config/agent_config.py`:

```python
TECHNICAL_AGENT = {
    "name": "technical_analysis",
    "description": "Monitors technical trading indicators",
    "search_topics": [...],
    "weight": 0.15
}
```

3. **Include in pipeline** in `agents/__init__.py` and `main.py`

### Adding New Metrics

In `utils/observability.py`:

```python
# Add new metric
metrics.record("prediction_spread", high - low)

# Add new counter
metrics.increment("api_calls")

# Add custom aggregation
def get_average_confidence():
    return metrics.get_average("confidence_score")
```

### Improving Predictions

1. **Add more data sources**: Weather, shipping data, refinery status
2. **Enhance the model**: Fine-tune prompts, add few-shot examples
3. **Improve evaluation**: Track more metrics, longer evaluation periods
4. **Add ensemble methods**: Combine multiple prediction strategies

---

## Troubleshooting

### Common Issues

#### "API key not found"

```
⚠️ GOOGLE_API_KEY not found. Running in mock mode.
```

**Solution:**
- Check that you've set the environment variable correctly
- On Kaggle, ensure the secret is named exactly `GOOGLE_API_KEY`
- Try restarting your terminal/kernel

#### "google-generativeai not installed"

```
Warning: google-generativeai not installed
```

**Solution:**
```bash
uv add google-genai
# or
pip install google-genai
```

#### "Rate limit exceeded"

```
Error: 429 Resource exhausted
```

**Solution:**
- Wait a few minutes and try again
- Reduce the number of parallel requests
- Use a paid API tier for higher limits

#### Import errors

```
ModuleNotFoundError: No module named 'agents'
```

**Solution:**
- Make sure you're running from the project root directory
- Check that `__init__.py` files exist in all directories
- Try: `cd c:\kaggle\oilprice` before running

#### Matplotlib not showing charts

**Solution:**
- In Jupyter: Add `%matplotlib inline` at the top
- Check matplotlib is installed: `uv add matplotlib`

### Getting Help

1. **Check the logs**: Look at the console output for error messages
2. **Enable debug logging**: Set `logging.DEBUG` in your code
3. **Review class notebooks**: Check `class/` folder for reference patterns
4. **Check ADK docs**: [Google ADK Documentation](https://google.github.io/adk-docs/)

---

## Summary

This project demonstrates how to build a sophisticated multi-agent AI system using **Google ADK**, following patterns from the **Kaggle 5-Day GenAI Intensive Course**.

### Key ADK Patterns Used:

| Day | Pattern | ADK Component |
|-----|---------|---------------|
| 1 | Agent architectures | `ParallelAgent`, `SequentialAgent`, `LoopAgent` |
| 2 | Built-in tools | `google_search`, `FunctionTool`, `BuiltInCodeExecutor` |
| 3 | Sessions & Memory | `InMemorySessionService`, `InMemoryMemoryService` |
| 4 | Observability | Callbacks, DEBUG logging |
| 5 | Evaluation | `LoopAgent` with accuracy tracking |

### Key Takeaways:

1. **`ParallelAgent`**: Run independent agents concurrently (4 research agents)
2. **`SequentialAgent`**: Chain agents for ordered processing (aggregator → trend → predictor)
3. **`LoopAgent`**: Iterate for refinement (evaluation runs up to 2x)
4. **`FunctionTool`**: Expose Python functions to agents for state management
5. **`InMemoryRunner`**: Execute agents with session and memory services

### Course Reference:

The `class/` folder contains all course notebooks for reference:
- `day-1a`: Basic agent with `google_search`
- `day-1b`: `ParallelAgent`, `SequentialAgent`, `LoopAgent`
- `day-2a`: `FunctionTool`, `AgentTool`, `BuiltInCodeExecutor`
- `day-3a/3b`: `InMemorySessionService`, `InMemoryMemoryService`
- `day-4a/4b`: Observability and evaluation
- `day-5b`: Deployment patterns

Happy coding! 🚀

---

*Last updated: November 2025*
*Built for: Kaggle 5-Day GenAI Intensive Course Capstone*
