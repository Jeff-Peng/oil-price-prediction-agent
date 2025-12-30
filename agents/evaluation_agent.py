"""
Evaluation Agent for Oil Price Prediction

Uses Google ADK Agent pattern with LoopAgent for iterative refinement.
Tracks prediction accuracy and improves the system over time.
"""

import os
import sys
from typing import Any, Dict, Optional

from agents.analysis_agents import (
    learned_weights,
    prediction_history,
    get_learned_weights,
    update_weights,
    get_recent_predictions,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Google ADK imports
try:
    from google.adk.agents import Agent, LoopAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.tools import FunctionTool
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    Agent = None
    LoopAgent = None
    Gemini = None
    FunctionTool = None
    types = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


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


# Default model configuration

def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of prediction performance for evaluation."""
    evaluated = [p for p in prediction_history if p.get("evaluated")]
    total = len(prediction_history)

    if not evaluated:
        return {
            "total_predictions": total,
            "evaluated_predictions": 0,
            "message": "No predictions have been evaluated yet"
        }

    errors_wti = []
    errors_brent = []
    correct_directions = 0

    for p in evaluated:
        if "actual_wti" in p:
            error = abs(p["wti_predicted"] - p["actual_wti"])
            errors_wti.append(error)
            # Check direction
            pred_up = p["wti_predicted"] > p["wti_current"]
            actual_up = p["actual_wti"] > p["wti_current"]
            if pred_up == actual_up:
                correct_directions += 1
        if "actual_brent" in p:
            errors_brent.append(abs(p["brent_predicted"] - p["actual_brent"]))

    if NUMPY_AVAILABLE:
        avg_error_wti = float(np.mean(errors_wti)) if errors_wti else 0
        avg_error_brent = float(np.mean(errors_brent)) if errors_brent else 0
    else:
        avg_error_wti = sum(errors_wti) / len(errors_wti) if errors_wti else 0
        avg_error_brent = (
            sum(errors_brent) / len(errors_brent)
            if errors_brent
            else 0
        )

    direction_accuracy = (
        correct_directions / len(evaluated) if evaluated else 0
    )

    return {
        "total_predictions": total,
        "evaluated_predictions": len(evaluated),
        "avg_error_wti": round(avg_error_wti, 2),
        "avg_error_brent": round(avg_error_brent, 2),
        "direction_accuracy": round(direction_accuracy, 3),
        "current_weights": learned_weights
    }


def mark_prediction_evaluated(
    timestamp: str,
    actual_wti: float,
    actual_brent: float
) -> str:
    """Mark a prediction as evaluated with actual prices."""
    for pred in prediction_history:
        if pred["timestamp"] == timestamp:
            pred["actual_wti"] = actual_wti
            pred["actual_brent"] = actual_brent
            pred["evaluated"] = True
            return f"Prediction at {timestamp} marked as evaluated"
    return f"Prediction at {timestamp} not found"


# --- Agent Instruction ---

EVALUATOR_INSTRUCTION = """You are a performance evaluation specialist.
Your role is to assess prediction accuracy and improve the system.

Use these tools:
- get_recent_predictions: Review recent prediction history
- get_performance_summary: Get accuracy statistics
- get_learned_weights: Get current factor weights
- update_weights: Adjust weights based on analysis

Analyze:
1. Compare predictions to actual outcomes (if available)
2. Calculate error metrics
3. Identify which factors were most predictive
4. Recommend weight adjustments

Learning loop:
- Track accuracy over time
- Identify systematic biases
- Suggest factor weight changes
- Flag anomalous predictions

If you have enough data and high confidence in your analysis,
call update_weights with your recommended new weights.
Weights should sum to 1.0 (100%).

Provide a summary of:
- Current performance metrics
- Factor performance analysis
- Recommended adjustments
- Learning insights"""


def create_evaluation_agent(
    model: Optional[Any] = None
) -> Optional[Any]:
    """
    Create the evaluation agent using Google ADK Agent pattern.

    Args:
        model: Optional Gemini model instance

    Returns:
        Agent instance or None if ADK not available
    """
    if not ADK_AVAILABLE or Agent is None or FunctionTool is None:
        return None

    active_model = model or create_model()

    # Create FunctionTools
    get_predictions_tool = FunctionTool(get_recent_predictions)
    get_performance_tool = FunctionTool(get_performance_summary)
    get_weights_tool = FunctionTool(get_learned_weights)
    update_weights_tool = FunctionTool(update_weights)

    return Agent(
        name="performance_evaluator",
        model=active_model,
        description="Evaluates prediction performance and adjusts weights",
        instruction=EVALUATOR_INSTRUCTION,
        tools=[
            get_predictions_tool,
            get_performance_tool,
            get_weights_tool,
            update_weights_tool
        ]
    )


def create_loop_evaluation_agent(
    model: Optional[Any] = None,
    max_iterations: int = 2
) -> Optional[Any]:
    """
    Create a LoopAgent for iterative evaluation.

    Args:
        model: Optional Gemini model instance
        max_iterations: Maximum iterations for the loop

    Returns:
        LoopAgent instance or None if ADK not available
    """
    if not ADK_AVAILABLE or LoopAgent is None:
        return None

    eval_agent = create_evaluation_agent(model)
    if eval_agent is None:
        return None

    return LoopAgent(
        name="evaluation_loop",
        description="Iterative evaluation and learning loop",
        sub_agents=[eval_agent],
        max_iterations=max_iterations
    )


# --- Helper Functions ---

def simulate_evaluation(
    wti_change: float = 0.5, brent_change: float = 0.5
) -> str:
    """
    Simulate evaluation by adding 'actual' prices to recent predictions.
    Useful for testing the learning loop.

    Args:
        wti_change: Random change range for WTI
        brent_change: Random change range for Brent

    Returns:
        Summary of simulated evaluations
    """
    import random

    evaluated_count = 0
    for pred in prediction_history:
        if not pred.get("evaluated"):
            # Simulate actual prices with some noise
            actual_wti = pred["wti_predicted"] + random.uniform(
                -wti_change, wti_change
            )
            actual_brent = pred["brent_predicted"] + random.uniform(
                -brent_change, brent_change
            )
            pred["actual_wti"] = round(actual_wti, 2)
            pred["actual_brent"] = round(actual_brent, 2)
            pred["evaluated"] = True
            evaluated_count += 1

    return f"Simulated evaluation for {evaluated_count} predictions"
