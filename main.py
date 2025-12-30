"""
Oil Price Prediction Agent - Main Entry Point

A multi-agent system for predicting oil prices using Google ADK.
Uses ADK patterns: Agent, ParallelAgent, SequentialAgent, LoopAgent.
"""

import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from agents import (
    ADK_AVAILABLE as AGENTS_ADK_AVAILABLE,
    create_parallel_research_agent,
    create_sequential_analysis_agent,
    create_loop_evaluation_agent,
    get_all_mock_research,
    set_current_prices,
    get_learned_weights,
    get_performance_summary,
)

# Load environment variables
load_dotenv()

# Try to import Google ADK
try:
    from google.genai import types
    from google.adk.agents import Agent
    from google.adk.models.google_llm import Gemini
    from google.adk.runners import InMemoryRunner
    from google.adk.sessions import InMemorySessionService
    from google.adk.memory import InMemoryMemoryService

    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False
    print("⚠️  google-adk not installed. Running in mock mode.")

# Constants
APP_NAME = "oil-price-prediction"
USER_ID = "oil_analyst"
MODEL_NAME = "gemini-2.5-flash-lite"

# HTTP Retry configuration
HTTP_RETRY_OPTIONS = None
if ADK_AVAILABLE:
    HTTP_RETRY_OPTIONS = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )


def create_model():
    """Create a Gemini model with retry configuration."""
    if not ADK_AVAILABLE:
        return None

    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY not found. Running in mock mode.")
        return None

    return Gemini(
        model=MODEL_NAME,
        http_options=types.HttpOptions(retry_options=HTTP_RETRY_OPTIONS)
    )


def create_orchestrator_agent(model):
    """Create the master orchestrator agent."""
    if not ADK_AVAILABLE or model is None:
        return None

    # Create sub-agents
    research_agent = create_parallel_research_agent(model)
    analysis_agent = create_sequential_analysis_agent(model)
    evaluation_agent = create_loop_evaluation_agent(model)

    if None in [research_agent, analysis_agent, evaluation_agent]:
        return None

    return Agent(
        name="oil_price_orchestrator",
        model=model,
        description="Master orchestrator for oil price prediction system",
        instruction="""You are the master orchestrator for oil price
prediction.

You coordinate three phases:
1. RESEARCH PHASE: Gather market intelligence from parallel research agents
2. ANALYSIS PHASE: Process research through sequential analysis pipeline
3. EVALUATION PHASE: Assess predictions and adjust weights

Execute all three phases and provide a comprehensive prediction summary.""",
        sub_agents=[research_agent, analysis_agent, evaluation_agent]
    )


async def run_with_adk(model, wti: float, brent: float):
    """Run the prediction using Google ADK agents."""
    print("🚀 Running with Google ADK agents...")

    # Set current prices
    set_current_prices(wti, brent)

    # Initialize ADK services
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    # Create session
    session = session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID
    )

    # Create orchestrator
    orchestrator = create_orchestrator_agent(model)
    if orchestrator is None:
        print("❌ Failed to create orchestrator agent")
        return None

    # Create runner
    runner = InMemoryRunner(
        agent=orchestrator,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # Run the agent
    user_message = f"""
    Run a complete oil price prediction cycle.

    Current prices:
    - WTI: ${wti:.2f}
    - Brent: ${brent:.2f}

    Execute:
    1. Research phase - gather current market intelligence
    2. Analysis phase - process data and generate predictions
    3. Evaluation phase - assess and learn

    Provide a comprehensive prediction summary.
    """

    content = types.Content(
        role="user",
        parts=[types.Part(text=user_message)]
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    final_response = part.text

    return final_response


def run_mock_mode(wti: float, brent: float):
    """Run in mock mode without ADK."""
    print("🔄 Running in mock mode...")

    # Get mock research data
    mock_research = get_all_mock_research()

    # Print mock results
    print("\n📊 Mock Research Results:")
    for agent, data in mock_research.items():
        print(f"  {agent}: {len(data['findings'])} findings")

    # Create mock prediction
    change_pct = 1.5  # Mock 1.5% increase
    wti_pred = wti * (1 + change_pct / 100)
    brent_pred = brent * (1 + change_pct / 100)

    report = f"""
🛢️ OIL PRICE PREDICTION (Mock Mode)
═══════════════════════════════════════════════════════════

📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 PREDICTIONS (24-Hour Outlook)
───────────────────────────────────────────────────────────
WTI Crude Oil:
  Current Price:    ${wti:.2f}
  Predicted Price:  ${wti_pred:.2f} (+{change_pct:.1f}%)
  Confidence:       75%

Brent Crude Oil:
  Current Price:    ${brent:.2f}
  Predicted Price:  ${brent_pred:.2f} (+{change_pct:.1f}%)
  Confidence:       72%

🔑 KEY FACTORS
───────────────────────────────────────────────────────────
BULLISH:
• OPEC+ production cut extension (HIGH impact)
• Strong Asian demand (MEDIUM impact)

BEARISH:
• USD strengthening (MEDIUM impact)

📊 RECOMMENDATION: HOLD
───────────────────────────────────────────────────────────
Market conditions are moderately bullish but near-term uncertainty
warrants a cautious approach.

⚖️ Factor Weights: {get_learned_weights()}
"""
    return report


async def run_prediction_pipeline(
    current_wti: float = 75.50,
    current_brent: float = 80.25
):
    """
    Run the full oil price prediction pipeline.

    Args:
        current_wti: Current WTI price
        current_brent: Current Brent price

    Returns:
        Prediction response string
    """
    print("\n" + "=" * 60)
    print("🛢️ OIL PRICE PREDICTION AGENT")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"💰 Current Prices: WTI ${current_wti:.2f}, "
        f"Brent ${current_brent:.2f}"
    )
    print("=" * 60)

    if ADK_AVAILABLE and AGENTS_ADK_AVAILABLE:
        model = create_model()
        if model:
            response = await run_with_adk(model, current_wti, current_brent)
            if response:
                print("\n" + response)
                return response

    # Fallback to mock mode
    report = run_mock_mode(current_wti, current_brent)
    print(report)
    return report


def main():
    """Main entry point"""
    print("\n" + "=" * 60)
    print("🚀 Starting Oil Price Prediction Agent")
    print("=" * 60)

    # Run prediction pipeline
    result = asyncio.run(run_prediction_pipeline())

    print("\n✅ Pipeline completed!")

    # Print performance summary
    print("\n📊 Performance Summary:")
    print(get_performance_summary())

    return result


if __name__ == "__main__":
    main()
