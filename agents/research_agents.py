"""
Research Agents for Oil Price Prediction

Uses Google ADK Agent pattern with google_search tool.
Contains 4 parallel research agents:
1. Geopolitical Monitor
2. Supply & Demand Tracker
3. Economic Indicator Monitor
4. Market Sentiment Analyzer
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Google ADK imports
try:
    from google.adk.agents import Agent, ParallelAgent
    from google.adk.models.google_llm import Gemini
    from google.adk.tools import google_search
    from google.genai import types

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    Agent = None
    ParallelAgent = None
    Gemini = None
    google_search = None
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


# --- Agent Instructions ---

GEOPOLITICAL_INSTRUCTION = """You are a geopolitical analyst for oil markets.
Use google_search to find current news about:
- OPEC+ production decisions and compliance
- Middle East tensions and supply disruptions
- Sanctions and trade policies affecting oil
- Political instability in major oil-producing regions

Assess each event's impact (HIGH/MEDIUM/LOW) and sentiment
(BULLISH/BEARISH/NEUTRAL).
Provide structured analysis with key factors and overall assessment."""

SUPPLY_DEMAND_INSTRUCTION = """You are a supply/demand analyst for crude oil.
Use google_search to find current data about:
- Global production levels (OPEC, US shale, etc.)
- Inventory levels (US EIA, OECD stocks)
- Demand indicators (refinery runs, imports)
- Seasonal patterns and trends

Quantify changes and assess supply/demand balance outlook.
Provide analysis with data points and market impact assessment."""

ECONOMIC_INSTRUCTION = """You are a macro-economic analyst for energy markets.
Use google_search to find current data about:
- US Dollar strength/weakness (DXY index)
- Interest rate expectations (Fed policy)
- GDP and growth forecasts
- Manufacturing and industrial activity

Analyze correlation with oil prices.
Provide analysis with economic data and oil price implications."""

SENTIMENT_INSTRUCTION = """You are a market sentiment analyst for oil markets.
Use google_search to find current data about:
- Futures positioning (COT data)
- Analyst forecasts and consensus
- Market commentary and narratives
- Technical indicators and patterns

Provide overall sentiment assessment with supporting data."""


def create_research_agents(
    model: Optional[Any] = None
) -> Dict[str, Optional[Any]]:
    """
    Create all research agents using Google ADK Agent pattern.

    Args:
        model: Optional Gemini model instance

    Returns:
        Dictionary of agent name to Agent instance
    """
    if not ADK_AVAILABLE or Agent is None:
        return {
            "geopolitical": None,
            "supply_demand": None,
            "economic": None,
            "sentiment": None
        }

    active_model = model or create_model()

    agents = {
        "geopolitical": Agent(
            name="geopolitical_monitor",
            model=active_model,
            description="Monitors geopolitical events affecting oil markets",
            instruction=GEOPOLITICAL_INSTRUCTION,
            tools=[google_search]
        ),
        "supply_demand": Agent(
            name="supply_demand_tracker",
            model=active_model,
            description="Tracks global oil supply and demand dynamics",
            instruction=SUPPLY_DEMAND_INSTRUCTION,
            tools=[google_search]
        ),
        "economic": Agent(
            name="economic_indicator_monitor",
            model=active_model,
            description="Monitors macro-economic indicators affecting oil",
            instruction=ECONOMIC_INSTRUCTION,
            tools=[google_search]
        ),
        "sentiment": Agent(
            name="sentiment_analyzer",
            model=active_model,
            description="Analyzes market sentiment and positioning",
            instruction=SENTIMENT_INSTRUCTION,
            tools=[google_search]
        )
    }

    return agents


def create_parallel_research_agent(
    model: Optional[Any] = None
) -> Optional[Any]:
    """
    Create a ParallelAgent that runs all research agents concurrently.

    Args:
        model: Optional Gemini model instance

    Returns:
        ParallelAgent instance or None if ADK not available
    """
    if not ADK_AVAILABLE or ParallelAgent is None:
        return None

    agents = create_research_agents(model)
    sub_agents = [a for a in agents.values() if a is not None]

    if not sub_agents:
        return None

    return ParallelAgent(
        name="research_coordinator",
        description="Coordinates parallel research from multiple experts",
        sub_agents=sub_agents
    )


# --- Mock Data for Testing ---

def get_mock_geopolitical_data() -> Dict:
    """Return mock geopolitical data for testing."""
    return {
        "agent_name": "geopolitical_monitor",
        "timestamp": datetime.now().isoformat(),
        "findings": [
            {
                "event": "OPEC+ Production Cut Extension",
                "description": "Saudi Arabia extends voluntary cut",
                "impact": "HIGH",
                "direction": "BULLISH"
            },
            {
                "event": "Middle East Tensions",
                "description": "Red Sea shipping concerns",
                "impact": "MEDIUM",
                "direction": "BULLISH"
            }
        ],
        "summary": "Geopolitical factors bullish for oil prices.",
        "confidence": 0.75
    }


def get_mock_supply_demand_data() -> Dict:
    """Return mock supply/demand data for testing."""
    return {
        "agent_name": "supply_demand_tracker",
        "timestamp": datetime.now().isoformat(),
        "findings": [
            {
                "metric": "US Crude Inventory",
                "value": "-3.2M barrels",
                "impact": "HIGH",
                "direction": "BULLISH"
            },
            {
                "metric": "China Demand",
                "value": "+3.2% YoY",
                "impact": "MEDIUM",
                "direction": "BULLISH"
            }
        ],
        "summary": "Market slightly undersupplied.",
        "confidence": 0.78
    }


def get_mock_economic_data() -> Dict:
    """Return mock economic data for testing."""
    return {
        "agent_name": "economic_indicator_monitor",
        "timestamp": datetime.now().isoformat(),
        "findings": [
            {
                "indicator": "US Dollar Index (DXY)",
                "value": "104.5",
                "impact": "MEDIUM",
                "direction": "BEARISH"
            },
            {
                "indicator": "China Manufacturing PMI",
                "value": "51.2",
                "impact": "MEDIUM",
                "direction": "BULLISH"
            }
        ],
        "summary": "Mixed economic signals.",
        "confidence": 0.72
    }


def get_mock_sentiment_data() -> Dict:
    """Return mock sentiment data for testing."""
    return {
        "agent_name": "sentiment_analyzer",
        "timestamp": datetime.now().isoformat(),
        "findings": [
            {
                "indicator": "COT Report - Managed Money",
                "value": "Net long 285K contracts",
                "impact": "MEDIUM",
                "direction": "BULLISH"
            },
            {
                "indicator": "Technical - RSI",
                "value": "62",
                "impact": "LOW",
                "direction": "BEARISH"
            }
        ],
        "summary": "Market sentiment moderately bullish.",
        "confidence": 0.70
    }


def get_all_mock_research() -> Dict[str, Dict]:
    """Return mock data for all research agents."""
    return {
        "geopolitical": get_mock_geopolitical_data(),
        "supply_demand": get_mock_supply_demand_data(),
        "economic": get_mock_economic_data(),
        "sentiment": get_mock_sentiment_data()
    }
