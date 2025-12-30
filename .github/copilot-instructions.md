# Oil Price Prediction Agent - Project Instructions

## Project Overview
This is a multi-agent system for predicting oil prices using Google ADK (Agent Development Kit).
Built for the Kaggle 5-Day Course Capstone Project.

## Key Concepts Implemented
1. Multi-agent system (Parallel + Sequential + Loop agents)
2. Built-in tools (Google Search + Code Execution)
3. Sessions & Memory (InMemorySessionService + Memory Bank)
4. Context Engineering (Compaction strategies)
5. Observability (Logging, Tracing, Metrics)
6. Agent Evaluation (Accuracy tracking, learning)

## Project Structure
- `agents/` - Agent implementations
- `memory/` - Memory bank and session management
- `models/` - Data models
- `utils/` - Observability and visualization utilities
- `config/` - Agent configurations
- `oil_price_prediction_agent.ipynb` - Main notebook

## Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Set up Google API key in environment or Kaggle Secrets
3. Run the Jupyter notebook or execute `main.py`

## Development Notes
- Use Google Search for research agents
- Use Code Execution for analysis agents
- Memory Bank stores historical predictions for learning
- Session manager tracks context across agent interactions
