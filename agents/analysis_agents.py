"""
Analysis Agents for Oil Price Prediction

Uses Google ADK Agent pattern with SequentialAgent.
Contains 3 sequential analysis agents:
1. Data Aggregator
2. Trend Analyzer
3. Price Predictor
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Google ADK imports
try:
    from google.adk.agents import Agent, SequentialAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.tools import FunctionTool
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    Agent = None
    SequentialAgent = None
    Gemini = None
    FunctionTool = None
    BuiltInCodeExecutor = None
    types = None


# Default model configuration
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"


def _get_retry_options() -> Optional[Any]:
    """Get HTTP retry options if ADK is available."""
    if not ADK_AVAILABLE or types is None:
        return None
    return types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )


def create_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[Any]:
    """Create a Gemini model with retry configuration."""
    if not ADK_AVAILABLE or Gemini is None or types is None:
        return None
    retry_options = _get_retry_options()
    return Gemini(
        model=model_name,
        http_options=types.HttpOptions(retry_options=retry_options)
    )


# --- Global State (used by FunctionTools) ---

learned_weights: Dict[str, float] = {
    "geopolitical": 0.25,
    "supply_demand": 0.30,
    "economic": 0.25,
    "sentiment": 0.20
}

current_prices: Dict[str, float] = {
    "wti": 75.50,
    "brent": 80.25
}

prediction_history: List[Dict] = []


# --- Function Tools ---

def get_learned_weights() -> Dict[str, float]:
    """Get the current learned factor weights for price prediction."""
    return learned_weights


def update_weights(
    geopolitical: float,
    supply_demand: float,
    economic: float,
    sentiment: float
) -> str:
    """Update the factor weights based on evaluation feedback."""
    global learned_weights
    total = geopolitical + supply_demand + economic + sentiment
    if total > 0:
        learned_weights = {
            "geopolitical": geopolitical / total,
            "supply_demand": supply_demand / total,
            "economic": economic / total,
            "sentiment": sentiment / total
        }
    return f"Weights updated: {learned_weights}"


def get_current_prices() -> Dict[str, float]:
    """Get the current WTI and Brent crude oil prices."""
    return current_prices


def store_prediction(
    wti_current: float,
    wti_predicted: float,
    brent_current: float,
    brent_predicted: float,
    confidence: float,
    recommendation: str,
    reasoning: str,
    key_factors: List[str]
) -> str:
    """Store a new price prediction in the prediction history."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "wti_current": wti_current,
        "wti_predicted": wti_predicted,
        "brent_current": brent_current,
        "brent_predicted": brent_predicted,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "key_factors": key_factors,
        "evaluated": False
    }
    prediction_history.append(record)
    return f"Prediction stored. Total predictions: {len(prediction_history)}"


def get_recent_predictions(n: int = 5) -> List[Dict]:
    """Get the most recent N predictions from history."""
    return prediction_history[-n:] if prediction_history else []


def calculate_accuracy() -> Dict[str, Any]:
    """Calculate prediction accuracy from evaluated predictions."""
    evaluated = [p for p in prediction_history if p.get("evaluated")]
    if not evaluated:
        return {"total": 0, "message": "No evaluated predictions yet"}

    errors_wti = []
    errors_brent = []
    for p in evaluated:
        if "actual_wti" in p:
            errors_wti.append(abs(p["wti_predicted"] - p["actual_wti"]))
        if "actual_brent" in p:
            errors_brent.append(abs(p["brent_predicted"] - p["actual_brent"]))

    import numpy as np
    return {
        "total_evaluated": len(evaluated),
        "avg_error_wti": float(np.mean(errors_wti)) if errors_wti else 0,
        "avg_error_brent": float(np.mean(errors_brent)) if errors_brent else 0
    }


# --- Agent Instructions ---

AGGREGATOR_INSTRUCTION = """You are a data aggregation specialist.
Your role is to combine outputs from multiple research agents.

Use get_learned_weights to retrieve current factor weights.

Tasks:
1. Normalize factor impacts to a common scale (0-1)
2. Resolve conflicting assessments between agents
3. Apply the learned weights to each factor category
4. Identify consensus views and outliers

Synthesize all research findings into a unified analysis with:
- Weighted factors from each category
- Overall market direction (BULLISH/BEARISH/NEUTRAL)
- Confidence level
- Key drivers

Pass this aggregated analysis to the next agent in the pipeline."""

TREND_ANALYZER_INSTRUCTION = """You are a quantitative trend analyst.
Your role is to analyze oil price trends and identify patterns.

Use the aggregated analysis from the previous step.

Perform:
1. Trend direction analysis (UP/DOWN/STABLE)
2. Trend strength measurement (0-1 scale)
3. Support and resistance level identification
4. Pattern recognition

Provide structured trend analysis with:
- Direction and strength
- Key support/resistance levels
- Technical indicators
- Trend summary

Pass your analysis to the price predictor."""

PREDICTOR_INSTRUCTION = """You are the final prediction agent for oil prices.
Your role is to synthesize all analysis into actionable predictions.

Use these tools:
- get_current_prices: Get current WTI and Brent prices
- calculate_accuracy: Review historical prediction accuracy
- store_prediction: Save your final prediction

Generate predictions for WTI and Brent crude oil (24-hour outlook).

Provide:
- Current and predicted prices
- Change percentages
- Confidence level (0-1)
- Key factors driving the prediction
- Recommendation (BUY/SELL/HOLD)
- Reasoning

IMPORTANT: Call store_prediction with your final values."""


def create_analysis_agents(
    model: Optional[Any] = None
) -> Dict[str, Optional[Any]]:
    """
    Create all analysis agents using Google ADK Agent pattern.

    Args:
        model: Optional Gemini model instance

    Returns:
        Dictionary of agent name to Agent instance
    """
    if not ADK_AVAILABLE or Agent is None or FunctionTool is None:
        return {
            "aggregator": None,
            "trend_analyzer": None,
            "predictor": None
        }

    active_model = model or create_model()

    # Create FunctionTools
    get_weights_tool = FunctionTool(get_learned_weights)
    get_prices_tool = FunctionTool(get_current_prices)
    store_pred_tool = FunctionTool(store_prediction)
    calc_accuracy_tool = FunctionTool(calculate_accuracy)

    agents = {
        "aggregator": Agent(
            name="data_aggregator",
            model=active_model,
            description="Aggregates and normalizes research data",
            instruction=AGGREGATOR_INSTRUCTION,
            tools=[get_weights_tool]
        ),
        "trend_analyzer": Agent(
            name="trend_analyzer",
            model=active_model,
            description="Analyzes price trends and patterns",
            instruction=TREND_ANALYZER_INSTRUCTION,
            tools=[BuiltInCodeExecutor()]
        ),
        "predictor": Agent(
            name="price_predictor",
            model=active_model,
            description="Generates final price predictions",
            instruction=PREDICTOR_INSTRUCTION,
            tools=[
                get_prices_tool,
                calc_accuracy_tool,
                store_pred_tool,
                BuiltInCodeExecutor()
            ]
        )
    }

    return agents


def create_sequential_analysis_agent(
    model: Optional[Any] = None
) -> Optional[Any]:
    """
    Create a SequentialAgent for the analysis pipeline.

    Args:
        model: Optional Gemini model instance

    Returns:
        SequentialAgent instance or None if ADK not available
    """
    if not ADK_AVAILABLE or SequentialAgent is None:
        return None

    agents = create_analysis_agents(model)
    sub_agents = [
        agents["aggregator"],
        agents["trend_analyzer"],
        agents["predictor"]
    ]

    if None in sub_agents:
        return None

    return SequentialAgent(
        name="analysis_pipeline",
        description="Sequential analysis: Aggregation → Trends → Prediction",
        sub_agents=sub_agents
    )


# --- Helper Functions ---

def set_current_prices(wti: float, brent: float) -> None:
    """Update current prices."""
    global current_prices
    current_prices = {"wti": wti, "brent": brent}


def get_prediction_history() -> List[Dict]:
    """Get all stored predictions."""
    return prediction_history


def clear_prediction_history() -> None:
    """Clear prediction history."""
    global prediction_history
    prediction_history = []
