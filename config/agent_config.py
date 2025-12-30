"""
Agent Configuration for Oil Price Prediction Agent

Contains agent definitions, prompts, and search topics for all 8 agents.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AgentConfig:
    """Configuration for a single agent"""
    name: str
    role: str
    description: str
    tools: List[str]
    search_topics: List[str]
    prompt_template: str
    output_format: str


# =============================================================================
# RESEARCH AGENTS (Parallel - Phase 1)
# =============================================================================

GEOPOLITICAL_AGENT = AgentConfig(
    name="GeopoliticalMonitor",
    role="Track political events affecting oil supply",
    description="Monitors OPEC decisions, sanctions, conflicts, and policy changes",
    tools=["google_search"],
    search_topics=[
        "OPEC+ production decisions latest news",
        "Middle East oil conflicts today",
        "oil sanctions news {current_date}",
        "oil trade agreements policy changes",
        "Russia oil exports sanctions",
        "Iran oil production news"
    ],
    prompt_template="""You are a Geopolitical Monitor agent specialized in tracking political events that affect oil prices.

Your task is to search for and analyze the latest geopolitical events affecting the oil market.

Search Topics to Cover:
1. OPEC+ production decisions and meetings
2. Middle East conflicts and tensions affecting oil infrastructure
3. Sanctions on oil-producing countries (Russia, Iran, Venezuela)
4. Trade agreements and disputes affecting oil trade
5. Government policy changes in major oil-producing/consuming nations

For each event found, provide:
- Event description
- Impact level (HIGH/MEDIUM/LOW) on oil prices
- Direction (BULLISH/BEARISH) for oil prices
- Confidence level (0.0-1.0)
- Source/date of information

Output your findings in a structured format.""",
    output_format="List of geopolitical events with impact scores"
)

SUPPLY_DEMAND_AGENT = AgentConfig(
    name="SupplyDemandTracker",
    role="Monitor production and consumption data",
    description="Tracks global oil production, inventories, and demand patterns",
    tools=["google_search"],
    search_topics=[
        "global oil production levels today",
        "US crude oil inventory EIA report",
        "oil refinery capacity utilization",
        "China oil demand forecast",
        "global oil demand growth",
        "oil rig count latest"
    ],
    prompt_template="""You are a Supply & Demand Tracker agent specialized in monitoring oil market fundamentals.

Your task is to search for and analyze the latest supply and demand data for the oil market.

Search Topics to Cover:
1. Global oil production levels (major producers)
2. US crude oil inventory (EIA weekly reports)
3. Refinery capacity and utilization rates
4. Demand forecasts from major economies (US, China, Europe, India)
5. Seasonal demand patterns
6. Oil rig counts and drilling activity

For each data point found, provide:
- Data description and value
- Comparison to previous period (week/month/year)
- Impact on supply/demand balance
- Direction (BULLISH/BEARISH) for oil prices
- Confidence level (0.0-1.0)

Conclude with an overall assessment: Is the market OVERSUPPLIED, BALANCED, or UNDERSUPPLIED?""",
    output_format="Supply/demand balance analysis with key metrics"
)

ECONOMIC_AGENT = AgentConfig(
    name="EconomicIndicatorMonitor",
    role="Track macroeconomic factors affecting oil prices",
    description="Monitors USD strength, inflation, GDP, and interest rates",
    tools=["google_search"],
    search_topics=[
        "US dollar index DXY today",
        "global GDP growth forecast",
        "inflation rate major economies",
        "Federal Reserve interest rate decision",
        "ECB monetary policy",
        "China economic indicators PMI"
    ],
    prompt_template="""You are an Economic Indicator Monitor agent specialized in tracking macroeconomic factors that affect oil prices.

Your task is to search for and analyze the latest economic indicators relevant to oil markets.

Search Topics to Cover:
1. USD strength (DXY index) - Oil is priced in USD
2. Global GDP growth forecasts - Affects overall demand
3. Inflation rates in major economies
4. Central bank interest rate decisions (Fed, ECB, etc.)
5. Manufacturing PMI data (indicates industrial demand)
6. Employment and consumer spending data

For each indicator found, provide:
- Indicator name and current value
- Recent trend (rising/falling/stable)
- Correlation with oil prices (positive/negative)
- Impact on oil prices (BULLISH/BEARISH)
- Confidence level (0.0-1.0)

Explain how each economic factor is likely to affect oil prices.""",
    output_format="Economic indicators with oil price correlation analysis"
)

SENTIMENT_AGENT = AgentConfig(
    name="MarketSentimentAnalyzer",
    role="Gauge trader sentiment and market expectations",
    description="Analyzes futures positioning, analyst forecasts, and technical indicators",
    tools=["google_search"],
    search_topics=[
        "oil futures positioning COT report",
        "crude oil price forecast analysts",
        "oil technical analysis today",
        "hedge fund oil positions",
        "oil market sentiment",
        "crude oil volatility VIX"
    ],
    prompt_template="""You are a Market Sentiment Analyzer agent specialized in gauging trader sentiment and market expectations for oil.

Your task is to search for and analyze the latest market sentiment indicators for oil.

Search Topics to Cover:
1. Oil futures positioning (COT - Commitment of Traders report)
2. Analyst price forecasts and consensus estimates
3. Hedge fund and institutional positioning
4. Technical indicators (RSI, MACD, moving averages)
5. Trading volumes and open interest
6. Oil volatility (OVX) and options sentiment

For each sentiment indicator found, provide:
- Indicator description
- Current reading/value
- Historical comparison (bullish/bearish relative to history)
- Signal strength (STRONG/MODERATE/WEAK)
- Confidence level (0.0-1.0)

Conclude with an overall sentiment score: BULLISH, MODERATELY_BULLISH, NEUTRAL, MODERATELY_BEARISH, or BEARISH.""",
    output_format="Sentiment score with supporting indicators"
)


# =============================================================================
# ANALYSIS AGENTS (Sequential - Phase 2)
# =============================================================================

AGGREGATOR_AGENT = AgentConfig(
    name="DataAggregator",
    role="Consolidate findings from all research agents",
    description="Combines, deduplicates, and categorizes all research findings",
    tools=[],  # No tools needed - processes other agents' output
    search_topics=[],
    prompt_template="""You are a Data Aggregator agent responsible for consolidating findings from all research agents.

You will receive findings from 4 research agents:
1. Geopolitical Monitor - Political events and policies
2. Supply & Demand Tracker - Production and consumption data
3. Economic Indicator Monitor - Macroeconomic factors
4. Market Sentiment Analyzer - Trader sentiment and technicals

Your task:
1. Combine all findings into a unified dataset
2. Remove any duplicate or redundant information
3. Categorize factors by impact level (HIGH/MEDIUM/LOW)
4. Separate into BULLISH and BEARISH factors
5. Identify any conflicting signals between agents
6. Calculate a weighted overall direction

Research Findings:
{research_findings}

Provide a structured summary with:
- Top 5 bullish factors (ranked by impact)
- Top 5 bearish factors (ranked by impact)
- Key conflicts or uncertainties
- Overall market direction assessment
- Confidence in the overall assessment""",
    output_format="Unified analysis with categorized factors"
)

TREND_ANALYZER_AGENT = AgentConfig(
    name="TrendAnalyzer",
    role="Identify patterns and calculate statistical indicators",
    description="Generates charts, calculates moving averages, identifies support/resistance",
    tools=["code_execution"],
    search_topics=[],
    prompt_template="""You are a Trend Analyzer agent responsible for statistical analysis of oil price trends.

Using the provided historical price data and aggregated findings, perform the following analysis:

1. Generate a 30-day price trend chart
2. Calculate moving averages:
   - 7-day simple moving average
   - 30-day simple moving average
3. Identify key price levels:
   - Support level (recent low with high volume)
   - Resistance level (recent high with rejection)
4. Measure volatility:
   - Calculate standard deviation of daily returns
   - Classify as HIGH/MODERATE/LOW volatility
5. Determine trend direction:
   - 7-day trend: UPWARD/SIDEWAYS/DOWNWARD
   - 30-day trend: UPWARD/SIDEWAYS/DOWNWARD
6. Compare current pattern with historical patterns from Memory Bank

Historical Data:
{historical_data}

Aggregated Findings:
{aggregated_findings}

Generate Python code to perform this analysis and create visualizations.
Provide a summary of the trend analysis results.""",
    output_format="Statistical analysis with charts and trend indicators"
)

PREDICTOR_AGENT = AgentConfig(
    name="PricePredictor",
    role="Generate price prediction with reasoning",
    description="Creates final prediction with confidence level and explanation",
    tools=[],
    search_topics=[],
    prompt_template="""You are a Price Predictor agent responsible for generating the final oil price prediction.

Based on all the analysis provided, generate a 7-day oil price prediction.

Inputs:
1. Aggregated Findings: {aggregated_findings}
2. Trend Analysis: {trend_analysis}
3. Historical Memory (past predictions and accuracy): {historical_memory}
4. Current Prices: WTI ${wti_current}, Brent ${brent_current}

Your prediction should include:

1. WTI Crude Oil:
   - Predicted price (7-day)
   - Predicted change percentage
   - Price range (low - high)
   - Confidence level (0.0-1.0)

2. Brent Crude Oil:
   - Predicted price (7-day)
   - Predicted change percentage
   - Price range (low - high)
   - Confidence level (0.0-1.0)

3. Key Reasoning:
   - Top 3 factors driving the prediction
   - Main risks to the prediction
   - Recommendation (BUY/HOLD/SELL)

Weight factors based on historical accuracy from Memory Bank.
Be conservative with confidence levels - high confidence should only be given when multiple strong signals align.

Output as structured JSON format.""",
    output_format="JSON with price predictions, confidence, and reasoning"
)


# =============================================================================
# EVALUATION AGENT (Loop - Phase 3)
# =============================================================================

EVALUATOR_AGENT = AgentConfig(
    name="PerformanceEvaluator",
    role="Track prediction accuracy and improve over time",
    description="Calculates metrics, identifies patterns, updates agent weights",
    tools=["code_execution"],
    search_topics=[],
    prompt_template="""You are a Performance Evaluator agent responsible for tracking prediction accuracy and improving the system.

Your task is to evaluate recent predictions against actual prices and update the learning system.

Previous Predictions and Actual Results:
{prediction_history}

Current Factor Weights:
{current_weights}

Perform the following analysis:

1. Calculate Accuracy Metrics:
   - Mean Absolute Error (MAE) for last 7 and 30 days
   - Root Mean Square Error (RMSE)
   - Directional accuracy (% of correct up/down calls)
   - Confidence calibration (does confidence match actual accuracy?)

2. Agent Performance Analysis:
   - Which agent's signals were most predictive?
   - Which agent's signals led to errors?
   - Rank agents by contribution to accuracy

3. Factor Analysis:
   - Which factors (geopolitical, supply/demand, economic, sentiment) were most predictive?
   - Identify any factors that consistently lead to errors

4. Learning Updates:
   - Recommend new factor weights based on performance
   - Suggest prompt improvements for underperforming agents
   - Identify patterns in prediction errors

5. Generate Performance Dashboard:
   - Create visualization of accuracy trends
   - Show factor importance chart
   - Display confidence calibration plot

Output updated weights and improvement recommendations.""",
    output_format="Performance metrics with updated weights and recommendations"
)


# =============================================================================
# AGENT REGISTRY
# =============================================================================

RESEARCH_AGENTS = {
    "geopolitical": GEOPOLITICAL_AGENT,
    "supply_demand": SUPPLY_DEMAND_AGENT,
    "economic": ECONOMIC_AGENT,
    "sentiment": SENTIMENT_AGENT
}

ANALYSIS_AGENTS = {
    "aggregator": AGGREGATOR_AGENT,
    "trend_analyzer": TREND_ANALYZER_AGENT,
    "predictor": PREDICTOR_AGENT
}

EVALUATION_AGENTS = {
    "evaluator": EVALUATOR_AGENT
}

ALL_AGENTS = {
    **RESEARCH_AGENTS,
    **ANALYSIS_AGENTS,
    **EVALUATION_AGENTS
}


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "model_name": "gemini-2.0-flash-exp",
    "temperature": 0.7,
    "max_output_tokens": 8192,
    "factor_weights": {
        "geopolitical": 0.30,
        "supply_demand": 0.30,
        "economic": 0.20,
        "sentiment": 0.20
    },
    "confidence_thresholds": {
        "high": 0.75,
        "medium": 0.50,
        "low": 0.25
    },
    "prediction_horizon_days": 7,
    "memory_retention_days": 90,
    "evaluation_frequency": "daily"
}


def get_agent_config(agent_name: str) -> AgentConfig:
    """Get configuration for a specific agent"""
    if agent_name not in ALL_AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")
    return ALL_AGENTS[agent_name]


def get_all_research_agents() -> Dict[str, AgentConfig]:
    """Get all research agents"""
    return RESEARCH_AGENTS


def get_all_analysis_agents() -> Dict[str, AgentConfig]:
    """Get all analysis agents"""
    return ANALYSIS_AGENTS
