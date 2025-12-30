"""
Agents package for Oil Price Prediction Agent

Uses Google ADK patterns:
- Agent, ParallelAgent, SequentialAgent, LoopAgent
- google_search, FunctionTool, AgentTool
- InMemoryRunner, InMemorySessionService
"""

# Research Agents (Google ADK Agent + google_search)
from .research_agents import (
    create_research_agents,
    create_parallel_research_agent,
    get_all_mock_research,
    ADK_AVAILABLE,
)

# Analysis Agents (Google ADK SequentialAgent)
from .analysis_agents import (
    create_analysis_agents,
    create_sequential_analysis_agent,
    get_learned_weights,
    update_weights,
    get_current_prices,
    set_current_prices,
    store_prediction,
    get_recent_predictions,
    get_prediction_history,
    clear_prediction_history,
)

# Evaluation Agent (Google ADK LoopAgent)
from .evaluation_agent import (
    create_evaluation_agent,
    create_loop_evaluation_agent,
    get_performance_summary,
    simulate_evaluation,
)

__all__ = [
    # ADK Availability
    'ADK_AVAILABLE',

    # Research Agents
    'create_research_agents',
    'create_parallel_research_agent',
    'get_all_mock_research',

    # Analysis Agents
    'create_analysis_agents',
    'create_sequential_analysis_agent',
    'get_learned_weights',
    'update_weights',
    'get_current_prices',
    'set_current_prices',
    'store_prediction',
    'get_recent_predictions',
    'get_prediction_history',
    'clear_prediction_history',

    # Evaluation Agent
    'create_evaluation_agent',
    'create_loop_evaluation_agent',
    'get_performance_summary',
    'simulate_evaluation',
]
