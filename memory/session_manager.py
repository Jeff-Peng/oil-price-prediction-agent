"""
Session Manager for Oil Price Prediction Agent

Manages analysis sessions and tracks context across agent interactions.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_models import (
    AnalysisReport,
    PricePrediction,
    ResearchResult,
    SessionContext,
    TrendAnalysis,
)


class SessionManager:
    """
    Manages analysis sessions for the Oil Price Prediction Agent.

    Provides:
    - Session creation and tracking
    - Context management across agent calls
    - State persistence within a session
    - Session history for debugging
    """

    def __init__(self):
        """Initialize Session Manager"""
        self.active_sessions: Dict[str, SessionContext] = {}
        self.completed_sessions: List[SessionContext] = []
        self.current_session_id: Optional[str] = None

    def create_session(self) -> str:
        """
        Create a new analysis session.

        Returns:
            Session ID
        """
        session_id = f"session_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"

        session = SessionContext(
            session_id=session_id,
            started_at=datetime.now()
        )

        self.active_sessions[session_id] = session
        self.current_session_id = session_id

        print(f"📋 Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: Optional[str] = None) -> Optional[SessionContext]:
        """
        Get a session by ID or return current session.

        Args:
            session_id: Optional session ID. Uses current if not provided.

        Returns:
            SessionContext or None
        """
        sid = session_id or self.current_session_id

        if sid is None:
            return None

        return self.active_sessions.get(sid)

    def get_current_session(self) -> Optional[SessionContext]:
        """Get the current active session"""
        return self.get_session(self.current_session_id)

    def add_research_result(
        self,
        agent_name: str,
        result: ResearchResult,
        session_id: Optional[str] = None
    ):
        """
        Add a research agent's result to the session.

        Args:
            agent_name: Name of the research agent
            result: ResearchResult from the agent
            session_id: Optional session ID
        """
        session = self.get_session(session_id)

        if session is None:
            raise ValueError("No active session. Create a session first.")

        session.research_results[agent_name] = result
        print(f"  ✅ Added result from {agent_name}")

    def set_aggregated_data(
        self,
        data: Dict,
        session_id: Optional[str] = None
    ):
        """
        Set the aggregated data from the Data Aggregator agent.

        Args:
            data: Aggregated findings
            session_id: Optional session ID
        """
        session = self.get_session(session_id)

        if session is None:
            raise ValueError("No active session. Create a session first.")

        session.aggregated_data = data
        print("  ✅ Set aggregated data")

    def set_trend_analysis(
        self,
        analysis: TrendAnalysis,
        session_id: Optional[str] = None
    ):
        """
        Set the trend analysis from the Trend Analyzer agent.

        Args:
            analysis: TrendAnalysis results
            session_id: Optional session ID
        """
        session = self.get_session(session_id)

        if session is None:
            raise ValueError("No active session. Create a session first.")

        session.trend_analysis = analysis
        print("  ✅ Set trend analysis")

    def set_prediction(
        self,
        prediction: PricePrediction,
        session_id: Optional[str] = None
    ):
        """
        Set the final prediction.

        Args:
            prediction: PricePrediction from the Predictor agent
            session_id: Optional session ID
        """
        session = self.get_session(session_id)

        if session is None:
            raise ValueError("No active session. Create a session first.")

        session.final_prediction = prediction
        print("  ✅ Set final prediction")

    def set_analysis_report(
        self,
        report: AnalysisReport,
        session_id: Optional[str] = None
    ):
        """
        Set the complete analysis report.

        Args:
            report: AnalysisReport combining all findings
            session_id: Optional session ID
        """
        session = self.get_session(session_id)

        if session is None:
            raise ValueError("No active session. Create a session first.")

        session.analysis_report = report
        print("  ✅ Set analysis report")

    def get_context_for_agent(
        self,
        agent_name: str,
        session_id: Optional[str] = None
    ) -> Dict:
        """
        Get the relevant context for a specific agent.

        Different agents need different context:
        - Research agents: Just the session metadata
        - Aggregator: All research results
        - Trend Analyzer: Aggregated data
        - Predictor: Aggregated data + trend analysis
        - Evaluator: Final prediction + historical data

        Args:
            agent_name: Name of the agent needing context
            session_id: Optional session ID

        Returns:
            Dictionary with relevant context
        """
        session = self.get_session(session_id)

        if session is None:
            return {"error": "No active session"}

        base_context = {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        }

        # Research agents just need basic context
        if agent_name in ["geopolitical", "supply_demand", "economic", "sentiment"]:
            return base_context

        # Aggregator needs all research results
        if agent_name == "aggregator":
            return {
                **base_context,
                "research_results": {
                    name: {
                        "summary": result.summary,
                        "findings": result.findings,
                        "confidence": result.confidence
                    }
                    for name, result in session.research_results.items()
                }
            }

        # Trend Analyzer needs aggregated data
        if agent_name == "trend_analyzer":
            return {
                **base_context,
                "aggregated_data": session.aggregated_data
            }

        # Predictor needs everything so far
        if agent_name == "predictor":
            return {
                **base_context,
                "aggregated_data": session.aggregated_data,
                "trend_analysis": (
                    {
                        "trend_7day": session.trend_analysis.trend_7day.value,
                        "trend_30day": session.trend_analysis.trend_30day.value,
                        "volatility": session.trend_analysis.volatility,
                        "support_level": session.trend_analysis.support_level,
                        "resistance_level": session.trend_analysis.resistance_level
                    }
                    if session.trend_analysis else None
                )
            }

        # Evaluator needs the full session
        if agent_name == "evaluator":
            return {
                **base_context,
                "prediction": (
                    session.final_prediction.to_dict()
                    if session.final_prediction else None
                ),
                "analysis_report": (
                    session.analysis_report.to_dict()
                    if session.analysis_report else None
                )
            }

        return base_context

    def complete_session(self, session_id: Optional[str] = None):
        """
        Mark a session as complete and archive it.

        Args:
            session_id: Optional session ID
        """
        sid = session_id or self.current_session_id

        if sid is None or sid not in self.active_sessions:
            print("Warning: No session to complete")
            return

        session = self.active_sessions[sid]
        session.completed_at = datetime.now()

        self.completed_sessions.append(session)
        del self.active_sessions[sid]

        if self.current_session_id == sid:
            self.current_session_id = None

        duration = (session.completed_at - session.started_at).total_seconds()
        print(f"✅ Session {sid} completed in {duration:.1f}s")

    def get_session_summary(self, session_id: Optional[str] = None) -> Dict:
        """
        Get a summary of a session.

        Args:
            session_id: Optional session ID

        Returns:
            Dictionary with session summary
        """
        session = self.get_session(session_id)

        if session is None:
            # Check completed sessions
            for s in self.completed_sessions:
                if s.session_id == session_id:
                    session = s
                    break

        if session is None:
            return {"error": "Session not found"}

        return {
            "session_id": session.session_id,
            "started_at": session.started_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "research_agents_completed": list(session.research_results.keys()),
            "has_aggregated_data": session.aggregated_data is not None,
            "has_trend_analysis": session.trend_analysis is not None,
            "has_prediction": session.final_prediction is not None,
            "is_complete": session.is_complete()
        }

    def list_sessions(self) -> Dict:
        """
        List all active and recent completed sessions.

        Returns:
            Dictionary with session lists
        """
        return {
            "active_sessions": [
                {
                    "session_id": s.session_id,
                    "started_at": s.started_at.isoformat(),
                    "is_complete": s.is_complete()
                }
                for s in self.active_sessions.values()
            ],
            "completed_sessions": [
                {
                    "session_id": s.session_id,
                    "started_at": s.started_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None
                }
                for s in self.completed_sessions[-10:]  # Last 10
            ],
            "current_session": self.current_session_id
        }


# Singleton instance for easy access
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
