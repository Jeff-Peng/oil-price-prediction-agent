"""
Memory Bank for Oil Price Prediction Agent

Long-term memory storage for predictions, performance metrics, and learned weights.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import HistoricalRecord, PerformanceMetrics


class MemoryBank:
    """
    Long-term memory storage for the Oil Price Prediction Agent.

    Stores:
    - Historical predictions and actual outcomes
    - Factor weights (learned over time)
    - Agent performance metrics
    - Major events and their impact
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize Memory Bank"""
        self.storage_path = storage_path or "memory_bank.json"

        # Prediction history
        self.predictions_history: List[HistoricalRecord] = []

        # Learned factor weights (start with equal weights)
        self.factor_weights = {
            "geopolitical": 0.25,
            "supply_demand": 0.25,
            "economic": 0.25,
            "sentiment": 0.25
        }

        # Agent performance tracking
        self.agent_performance: Dict[str, Dict] = {
            "geopolitical": {"total_signals": 0, "correct_signals": 0, "accuracy": 0.5},
            "supply_demand": {"total_signals": 0, "correct_signals": 0, "accuracy": 0.5},
            "economic": {"total_signals": 0, "correct_signals": 0, "accuracy": 0.5},
            "sentiment": {"total_signals": 0, "correct_signals": 0, "accuracy": 0.5}
        }

        # Performance metrics history
        self.performance_history: List[PerformanceMetrics] = []

        # Major events database
        self.major_events: List[Dict] = []

        # Load existing data if available
        self._load()

    def _load(self):
        """Load memory from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.factor_weights = data.get('factor_weights', self.factor_weights)
                    self.agent_performance = data.get('agent_performance', self.agent_performance)
                    self.major_events = data.get('major_events', [])

                    # Reconstruct predictions history
                    for pred in data.get('predictions_history', []):
                        record = HistoricalRecord(
                            date=datetime.fromisoformat(pred['date']),
                            prediction_id=pred['prediction_id'],
                            wti_predicted=pred['wti_predicted'],
                            wti_actual=pred.get('wti_actual'),
                            brent_predicted=pred['brent_predicted'],
                            brent_actual=pred.get('brent_actual'),
                            factors_used=pred['factors_used'],
                            direction_correct=pred.get('direction_correct'),
                            error=pred.get('error')
                        )
                        self.predictions_history.append(record)
            except Exception as e:
                print(f"Warning: Could not load memory bank: {e}")

    def save(self):
        """Save memory to storage"""
        data = {
            'factor_weights': self.factor_weights,
            'agent_performance': self.agent_performance,
            'major_events': self.major_events,
            'predictions_history': [
                {
                    'date': record.date.isoformat(),
                    'prediction_id': record.prediction_id,
                    'wti_predicted': record.wti_predicted,
                    'wti_actual': record.wti_actual,
                    'brent_predicted': record.brent_predicted,
                    'brent_actual': record.brent_actual,
                    'factors_used': record.factors_used,
                    'direction_correct': record.direction_correct,
                    'error': record.error
                }
                for record in self.predictions_history
            ]
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def store_prediction(self, prediction_data: Dict):
        """
        Store a new prediction for future evaluation.

        Args:
            prediction_data: Dictionary containing prediction details
        """
        record = HistoricalRecord(
            date=datetime.now(),
            prediction_id=prediction_data.get('prediction_id', f"pred_{datetime.now().timestamp()}"),
            wti_predicted=prediction_data['wti_predicted'],
            wti_actual=prediction_data.get('wti_actual'),
            brent_predicted=prediction_data['brent_predicted'],
            brent_actual=prediction_data.get('brent_actual'),
            factors_used=prediction_data.get('factors_used', [])
        )

        self.predictions_history.append(record)
        self.save()

    def update_actual_price(self, prediction_id: str, wti_actual: float, brent_actual: float):
        """
        Update a prediction with actual prices for evaluation.

        Args:
            prediction_id: ID of the prediction to update
            wti_actual: Actual WTI price
            brent_actual: Actual Brent price
        """
        for record in self.predictions_history:
            if record.prediction_id == prediction_id:
                record.wti_actual = wti_actual
                record.brent_actual = brent_actual
                record.calculate_error()
                self.save()
                return

        print(f"Warning: Prediction {prediction_id} not found")

    def update_weights(self, evaluation_results: Dict):
        """
        Update factor weights based on evaluation results.

        Uses exponential moving average to smooth weight updates.

        Args:
            evaluation_results: Dictionary with agent accuracy data
        """
        alpha = 0.1  # Learning rate

        # Calculate new weights based on accuracy
        total_accuracy = sum(
            evaluation_results.get(agent, {}).get('accuracy', 0.5)
            for agent in self.factor_weights.keys()
        )

        if total_accuracy > 0:
            for agent in self.factor_weights.keys():
                accuracy = evaluation_results.get(agent, {}).get('accuracy', 0.5)
                new_weight = accuracy / total_accuracy

                # Exponential moving average update
                self.factor_weights[agent] = (
                    (1 - alpha) * self.factor_weights[agent] +
                    alpha * new_weight
                )

            # Normalize weights to sum to 1
            total_weight = sum(self.factor_weights.values())
            self.factor_weights = {
                k: v / total_weight for k, v in self.factor_weights.items()
            }

        self.save()

    def get_historical_pattern(self, current_factors: Optional[Dict] = None) -> Dict:
        """
        Retrieve similar historical scenarios for pattern matching.

        Args:
            current_factors: Current market factors to match against

        Returns:
            Dictionary with similar historical patterns
        """
        if not self.predictions_history:
            return {"patterns": [], "message": "No historical data available"}

        # Get last 30 days of predictions with results
        recent_predictions = [
            p for p in self.predictions_history[-30:]
            if p.wti_actual is not None
        ]

        if not recent_predictions:
            return {"patterns": [], "message": "No evaluated predictions available"}

        # Calculate statistics
        errors = [p.error for p in recent_predictions if p.error is not None]
        directions = [p.direction_correct for p in recent_predictions if p.direction_correct is not None]

        return {
            "recent_predictions": len(recent_predictions),
            "avg_error": np.mean(errors) if errors else None,
            "directional_accuracy": np.mean(directions) if directions else None,
            "current_weights": self.factor_weights,
            "patterns": [
                {
                    "date": p.date.isoformat(),
                    "predicted": p.wti_predicted,
                    "actual": p.wti_actual,
                    "error": p.error,
                    "direction_correct": p.direction_correct
                }
                for p in recent_predictions[-5:]  # Last 5 predictions
            ]
        }

    def get_performance_trends(self, days: int = 30) -> Dict:
        """
        Get accuracy trends over specified time period.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with performance trends
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent = [
            p for p in self.predictions_history
            if p.date >= cutoff_date and p.wti_actual is not None
        ]

        if not recent:
            return {"message": "No data for specified period"}

        errors = [p.error for p in recent if p.error is not None]
        directions = [p.direction_correct for p in recent if p.direction_correct is not None]

        # Calculate improvement trend (compare first half vs second half)
        mid = len(recent) // 2
        if mid > 0:
            first_half_errors = [p.error for p in recent[:mid] if p.error is not None]
            second_half_errors = [p.error for p in recent[mid:] if p.error is not None]

            if first_half_errors and second_half_errors:
                improvement = np.mean(first_half_errors) - np.mean(second_half_errors)
                trend = "IMPROVING" if improvement > 0 else "DECLINING" if improvement < 0 else "STABLE"
            else:
                trend = "UNKNOWN"
        else:
            trend = "INSUFFICIENT_DATA"

        return {
            "period_days": days,
            "predictions_count": len(recent),
            "mae": np.mean(errors) if errors else None,
            "rmse": np.sqrt(np.mean([e**2 for e in errors])) if errors else None,
            "directional_accuracy": np.mean(directions) if directions else None,
            "trend": trend,
            "current_weights": self.factor_weights
        }

    def add_major_event(self, event: Dict):
        """
        Record a major market event for future reference.

        Args:
            event: Dictionary with event details
        """
        event['recorded_at'] = datetime.now().isoformat()
        self.major_events.append(event)
        self.save()

    def get_relevant_events(self, lookback_days: int = 30) -> List[Dict]:
        """
        Get major events from the specified lookback period.

        Args:
            lookback_days: Number of days to look back

        Returns:
            List of relevant events
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)

        return [
            e for e in self.major_events
            if datetime.fromisoformat(e['recorded_at']) >= cutoff
        ]

    def clear_old_data(self, retention_days: int = 90):
        """
        Remove data older than retention period.

        Args:
            retention_days: Days to retain data
        """
        cutoff = datetime.now() - timedelta(days=retention_days)

        self.predictions_history = [
            p for p in self.predictions_history
            if p.date >= cutoff
        ]

        self.major_events = [
            e for e in self.major_events
            if datetime.fromisoformat(e['recorded_at']) >= cutoff
        ]

        self.save()
