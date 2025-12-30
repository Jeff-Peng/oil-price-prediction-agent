"""
Data Models for Oil Price Prediction Agent

Contains data classes for predictions, analysis reports, historical records,
and performance metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class Impact(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Sentiment(Enum):
    BULLISH = "BULLISH"
    MODERATELY_BULLISH = "MODERATELY_BULLISH"
    NEUTRAL = "NEUTRAL"
    MODERATELY_BEARISH = "MODERATELY_BEARISH"
    BEARISH = "BEARISH"


class Recommendation(Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class TrendDirection(Enum):
    UPWARD = "UPWARD"
    SIDEWAYS = "SIDEWAYS"
    DOWNWARD = "DOWNWARD"


@dataclass
class CrudePricePrediction:
    """Prediction for a single crude oil type"""
    current_price: float
    predicted_price: float
    predicted_change_pct: float
    predicted_range: tuple  # (low, high)
    confidence: float  # 0.0 to 1.0


@dataclass
class PricePrediction:
    """Complete price prediction for WTI and Brent crude"""
    timestamp: datetime
    prediction_id: str
    wti_crude: CrudePricePrediction
    brent_crude: CrudePricePrediction
    time_horizon: str = "7_days"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "prediction_id": self.prediction_id,
            "wti_crude": {
                "current_price": self.wti_crude.current_price,
                "predicted_price": self.wti_crude.predicted_price,
                "predicted_change_pct": self.wti_crude.predicted_change_pct,
                "predicted_range": list(self.wti_crude.predicted_range),
                "confidence": self.wti_crude.confidence
            },
            "brent_crude": {
                "current_price": self.brent_crude.current_price,
                "predicted_price": self.brent_crude.predicted_price,
                "predicted_change_pct": self.brent_crude.predicted_change_pct,
                "predicted_range": list(self.brent_crude.predicted_range),
                "confidence": self.brent_crude.confidence
            },
            "time_horizon": self.time_horizon
        }


@dataclass
class Factor:
    """A single factor affecting oil prices"""
    factor: str
    impact: Impact
    confidence: float
    source: str  # Which agent provided this
    details: str


@dataclass
class TrendAnalysis:
    """Technical trend analysis results"""
    trend_7day: TrendDirection
    trend_30day: TrendDirection
    volatility: str  # HIGH, MODERATE, LOW
    support_level: float
    resistance_level: float
    moving_avg_7day: Optional[float] = None
    moving_avg_30day: Optional[float] = None


@dataclass
class AnalysisReport:
    """Complete analysis report from all agents"""
    prediction_id: str
    timestamp: datetime
    bullish_factors: List[Factor]
    bearish_factors: List[Factor]
    trend_analysis: TrendAnalysis
    market_sentiment: Sentiment
    recommendation: Recommendation
    reasoning: str

    def to_dict(self) -> Dict:
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "bullish_factors": [
                {
                    "factor": f.factor,
                    "impact": f.impact.value,
                    "confidence": f.confidence,
                    "source": f.source,
                    "details": f.details
                }
                for f in self.bullish_factors
            ],
            "bearish_factors": [
                {
                    "factor": f.factor,
                    "impact": f.impact.value,
                    "confidence": f.confidence,
                    "source": f.source,
                    "details": f.details
                }
                for f in self.bearish_factors
            ],
            "trend_analysis": {
                "7day_trend": self.trend_analysis.trend_7day.value,
                "30day_trend": self.trend_analysis.trend_30day.value,
                "volatility": self.trend_analysis.volatility,
                "support_level": self.trend_analysis.support_level,
                "resistance_level": self.trend_analysis.resistance_level
            },
            "market_sentiment": self.market_sentiment.value,
            "recommendation": self.recommendation.value,
            "reasoning": self.reasoning
        }


@dataclass
class HistoricalRecord:
    """Record of a past prediction for evaluation"""
    date: datetime
    prediction_id: str
    wti_predicted: float
    wti_actual: Optional[float]
    brent_predicted: float
    brent_actual: Optional[float]
    factors_used: List[Dict]  # factor name -> weight
    direction_correct: Optional[bool] = None
    error: Optional[float] = None

    def calculate_error(self):
        """Calculate prediction error if actual price is available"""
        if self.wti_actual is not None:
            self.error = abs(self.wti_predicted - self.wti_actual)
            self.direction_correct = (
                (self.wti_predicted > self.wti_actual and self.wti_actual > 0) or
                (self.wti_predicted < self.wti_actual and self.wti_actual < 0)
            )


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent"""
    agent_name: str
    impact_score: float  # How much this agent's signals affected predictions
    accuracy: float  # How accurate this agent's signals were
    total_signals: int
    correct_signals: int


@dataclass
class PerformanceMetrics:
    """Overall performance metrics"""
    evaluation_date: datetime
    period: str  # e.g., "last_30_days"
    predictions_made: int
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    directional_accuracy: float  # % of correct up/down predictions
    confidence_calibration: float  # How well confidence matches actual accuracy
    agent_performance: List[AgentPerformance]
    best_factors: List[str]
    improvement_trend: str  # IMPROVING, STABLE, DECLINING

    def to_dict(self) -> Dict:
        return {
            "evaluation_date": self.evaluation_date.isoformat(),
            "period": self.period,
            "predictions_made": self.predictions_made,
            "overall_accuracy": {
                "mae": self.mae,
                "rmse": self.rmse,
                "directional_accuracy": self.directional_accuracy,
                "confidence_calibration": self.confidence_calibration
            },
            "agent_performance": {
                ap.agent_name: {
                    "impact_score": ap.impact_score,
                    "accuracy": ap.accuracy
                }
                for ap in self.agent_performance
            },
            "best_factors": self.best_factors,
            "improvement_trend": self.improvement_trend
        }


@dataclass
class ResearchResult:
    """Result from a research agent"""
    agent_name: str
    timestamp: datetime
    search_queries: List[str]
    findings: List[Dict]
    summary: str
    confidence: float
    raw_response: Optional[str] = None


@dataclass
class SessionContext:
    """Context for an analysis session"""
    session_id: str
    started_at: datetime
    research_results: Dict[str, ResearchResult] = field(default_factory=dict)
    aggregated_data: Optional[Dict] = None
    trend_analysis: Optional[TrendAnalysis] = None
    final_prediction: Optional[PricePrediction] = None
    analysis_report: Optional[AnalysisReport] = None
    completed_at: Optional[datetime] = None

    def is_complete(self) -> bool:
        return self.final_prediction is not None
