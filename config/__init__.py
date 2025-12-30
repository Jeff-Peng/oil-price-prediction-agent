"""Config package for Oil Price Prediction Agent"""

from .agent_config import (
    AGGREGATOR_AGENT,
    ALL_AGENTS,
    ANALYSIS_AGENTS,
    DEFAULT_CONFIG,
    ECONOMIC_AGENT,
    EVALUATION_AGENTS,
    EVALUATOR_AGENT,
    GEOPOLITICAL_AGENT,
    PREDICTOR_AGENT,
    RESEARCH_AGENTS,
    SENTIMENT_AGENT,
    SUPPLY_DEMAND_AGENT,
    TREND_ANALYZER_AGENT,
    AgentConfig,
    get_agent_config,
    get_all_analysis_agents,
    get_all_research_agents,
)

__all__ = [
    'AgentConfig',
    'GEOPOLITICAL_AGENT',
    'SUPPLY_DEMAND_AGENT',
    'ECONOMIC_AGENT',
    'SENTIMENT_AGENT',
    'AGGREGATOR_AGENT',
    'TREND_ANALYZER_AGENT',
    'PREDICTOR_AGENT',
    'EVALUATOR_AGENT',
    'RESEARCH_AGENTS',
    'ANALYSIS_AGENTS',
    'EVALUATION_AGENTS',
    'ALL_AGENTS',
    'DEFAULT_CONFIG',
    'get_agent_config',
    'get_all_research_agents',
    'get_all_analysis_agents'
]
