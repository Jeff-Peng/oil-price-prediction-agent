"""
Visualization Utilities for Oil Price Prediction Agent

Provides chart generation and dashboard visualization.
"""

import base64
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa: F401
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed")

try:
    import numpy as np  # noqa: F401
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import os
import sys

from models.data_models import (
    AnalysisReport,
    PerformanceMetrics,
    PricePrediction,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_plotting_style():
    """Set up matplotlib style for consistent plots"""
    if MATPLOTLIB_AVAILABLE:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10


def create_price_trend_chart(
    prices: List[Dict],
    prediction: Optional[PricePrediction] = None,
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Create a price trend chart with optional prediction overlay.

    Args:
        prices: List of price dictionaries with 'date' and 'close' keys
        prediction: Optional prediction to overlay
        save_path: Optional path to save the chart

    Returns:
        Base64 encoded image string or None
    """
    if not MATPLOTLIB_AVAILABLE or not prices:
        print("Cannot create chart: matplotlib not available or no data")
        return None

    setup_plotting_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Parse dates and prices
    dates = [
        datetime.strptime(p['date'], '%Y-%m-%d')
        if isinstance(p['date'], str)
        else p['date']
        for p in prices
    ]
    closes = [p['close'] for p in prices]

    # Plot historical prices
    ax.plot(
        dates, closes, 'b-', linewidth=2,
        label='WTI Crude Oil', marker='o', markersize=4
    )

    # Add moving averages if enough data
    if len(closes) >= 7:
        ma7 = (
            pd.Series(closes).rolling(7).mean().values
            if PANDAS_AVAILABLE
            else None
        )
        if ma7 is not None:
            ax.plot(
                dates, ma7, 'g--', linewidth=1.5,
                label='7-Day MA', alpha=0.7
            )

    if len(closes) >= 30:
        ma30 = (
            pd.Series(closes).rolling(30).mean().values
            if PANDAS_AVAILABLE
            else None
        )
        if ma30 is not None:
            ax.plot(
                dates, ma30, 'r--', linewidth=1.5,
                label='30-Day MA', alpha=0.7
            )

    # Add prediction if provided
    if prediction:
        pred_date = prediction.timestamp + timedelta(days=7)
        pred_price = prediction.wti_crude.predicted_price
        pred_range = prediction.wti_crude.predicted_range

        # Prediction point
        ax.scatter([pred_date], [pred_price], color='purple', s=100, zorder=5,
                   label=f'Prediction: ${pred_price:.2f}')

        # Prediction range
        ax.fill_between(
            [dates[-1], pred_date],
            [closes[-1], pred_range[0]],
            [closes[-1], pred_range[1]],
            alpha=0.2,
            color='purple',
            label=f'Range: ${pred_range[0]:.2f}-${pred_range[1]:.2f}'
        )

        # Connecting line
        ax.plot([dates[-1], pred_date], [closes[-1], pred_price],
                'purple', linestyle=':', linewidth=2)

    # Formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.set_title('WTI Crude Oil Price Trend', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or return as base64
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


def create_factor_chart(
    report: AnalysisReport,
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Create a chart showing bullish vs bearish factors.

    Args:
        report: Analysis report with factors
        save_path: Optional path to save the chart

    Returns:
        Base64 encoded image string or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_plotting_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Impact scores
    impact_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

    # Bullish factors
    bullish_names = [f.factor[:30] + '...' if len(f.factor) > 30 else f.factor
                     for f in report.bullish_factors]
    bullish_scores = [impact_scores.get(f.impact.value, 1) * f.confidence
                      for f in report.bullish_factors]

    if bullish_names:
        colors = ['#2ecc71' if s > 2 else '#27ae60' if s > 1 else '#1abc9c'
                  for s in bullish_scores]
        ax1.barh(bullish_names, bullish_scores, color=colors)
        ax1.set_xlabel('Impact Score')
        ax1.set_title('🔴 Bullish Factors', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, max(bullish_scores) * 1.2 if bullish_scores else 3)
    else:
        ax1.text(0.5, 0.5, 'No bullish factors', ha='center', va='center')
        ax1.set_title('Bullish Factors')

    # Bearish factors
    bearish_names = [f.factor[:30] + '...' if len(f.factor) > 30 else f.factor
                     for f in report.bearish_factors]
    bearish_scores = [impact_scores.get(f.impact.value, 1) * f.confidence
                      for f in report.bearish_factors]

    if bearish_names:
        colors = ['#e74c3c' if s > 2 else '#c0392b' if s > 1 else '#e67e22'
                  for s in bearish_scores]
        ax2.barh(bearish_names, bearish_scores, color=colors)
        ax2.set_xlabel('Impact Score')
        ax2.set_title('🟢 Bearish Factors', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, max(bearish_scores) * 1.2 if bearish_scores else 3)
    else:
        ax2.text(0.5, 0.5, 'No bearish factors', ha='center', va='center')
        ax2.set_title('Bearish Factors')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


def create_performance_dashboard(
    metrics: PerformanceMetrics,
    prediction_history: Optional[List[Dict]] = None,
    save_path: Optional[str] = None
) -> Optional[str]:
    """
    Create a performance dashboard with multiple charts.

    Args:
        metrics: Performance metrics
        prediction_history: Optional prediction history for trend charts
        save_path: Optional path to save the chart

    Returns:
        Base64 encoded image string or None
    """
    if not MATPLOTLIB_AVAILABLE:
        return None

    setup_plotting_style()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Accuracy Metrics
    ax1 = axes[0, 0]
    metric_names = [
        'MAE ($)',
        'RMSE ($)',
        'Directional\nAccuracy',
        'Confidence\nCalibration',
    ]
    metric_values = [
        metrics.mae,
        metrics.rmse,
        metrics.directional_accuracy * 100,
        metrics.confidence_calibration * 100,
    ]
    colors = ['#3498db', '#2980b9', '#27ae60', '#16a085']

    bars = ax1.bar(metric_names, metric_values, color=colors)
    ax1.set_title('📊 Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value')

    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}' if val > 10 else f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=10)

    # Chart 2: Agent Performance
    ax2 = axes[0, 1]
    if metrics.agent_performance:
        agents = [ap.agent_name for ap in metrics.agent_performance]
        accuracies = [ap.accuracy * 100 for ap in metrics.agent_performance]
        impacts = [ap.impact_score * 100 for ap in metrics.agent_performance]

        x = range(len(agents))
        width = 0.35

        ax2.bar(
            [i - width / 2 for i in x],
            accuracies,
            width,
            label='Accuracy %',
            color='#3498db',
        )
        ax2.bar(
            [i + width / 2 for i in x],
            impacts,
            width,
            label='Impact %',
            color='#e74c3c',
        )

        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Percentage')
        ax2.set_title('🤖 Agent Performance', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([a.replace('_', '\n') for a in agents], fontsize=9)
        ax2.legend()

    # Chart 3: Factor Weights (Pie Chart)
    ax3 = axes[1, 0]
    if metrics.agent_performance:
        labels = [
            ap.agent_name.replace('_', '\n')
            for ap in metrics.agent_performance
        ]
        sizes = [ap.impact_score for ap in metrics.agent_performance]
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        ax3.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors_pie,
            startangle=90,
        )
        ax3.set_title(
            '📊 Factor Weight Distribution',
            fontsize=12,
            fontweight='bold',
        )

    # Chart 4: Summary Statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    📈 PERFORMANCE SUMMARY
    ═══════════════════════════════════

    Period: {metrics.period}
    Predictions Made: {metrics.predictions_made}

    Mean Absolute Error: ${metrics.mae:.2f}
    Root Mean Square Error: ${metrics.rmse:.2f}
    Directional Accuracy: {metrics.directional_accuracy:.1%}

    Best Factors:
    {chr(10).join(f'  • {f}' for f in metrics.best_factors[:3])}

    Trend: {metrics.improvement_trend}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Oil Price Prediction Agent - Performance Dashboard',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


def print_prediction_report(
    prediction: PricePrediction,
    report: AnalysisReport
):
    """
    Print a formatted prediction report to console.

    Args:
        prediction: Price prediction
        report: Analysis report
    """
    print("\n" + "═" * 60)
    print("🛢️ OIL PRICE PREDICTION REPORT")
    print("═" * 60)

    print(f"\n📅 Date: {prediction.timestamp.strftime('%B %d, %Y')}")
    print(f"📋 Prediction ID: {prediction.prediction_id}")

    print("\n📈 PREDICTIONS (7-Day Horizon)")
    print("─" * 60)

    # WTI
    wti = prediction.wti_crude
    print("\nWTI Crude Oil:")
    print(f"  Current Price:    ${wti.current_price:.2f}")
    print(
        f"  Predicted Price:  ${wti.predicted_price:.2f} "
        f"({wti.predicted_change_pct:+.2f}%)"
    )
    print(
        f"  Price Range:      ${wti.predicted_range[0]:.2f} - "
        f"${wti.predicted_range[1]:.2f}"
    )
    print(f"  Confidence:       {wti.confidence:.0%}")

    # Brent
    brent = prediction.brent_crude
    print("\nBrent Crude Oil:")
    print(f"  Current Price:    ${brent.current_price:.2f}")
    print(
        f"  Predicted Price:  ${brent.predicted_price:.2f} "
        f"({brent.predicted_change_pct:+.2f}%)"
    )
    print(
        f"  Price Range:      ${brent.predicted_range[0]:.2f} - "
        f"${brent.predicted_range[1]:.2f}"
    )
    print(f"  Confidence:       {brent.confidence:.0%}")

    # Bullish Factors
    print("\n🔴 BULLISH FACTORS")
    print("─" * 60)
    for i, f in enumerate(report.bullish_factors, 1):
        print(f"{i}. [{f.impact.value}] {f.factor}")
        if f.details:
            print(f"   {f.details}")

    # Bearish Factors
    print("\n🟢 BEARISH FACTORS")
    print("─" * 60)
    for i, f in enumerate(report.bearish_factors, 1):
        print(f"{i}. [{f.impact.value}] {f.factor}")
        if f.details:
            print(f"   {f.details}")

    # Trend Analysis
    print("\n📊 TREND ANALYSIS")
    print("─" * 60)
    ta = report.trend_analysis
    print(f"  7-Day Trend:  {ta.trend_7day.value}")
    print(f"  30-Day Trend: {ta.trend_30day.value}")
    print(f"  Volatility:   {ta.volatility}")
    print(f"  Support:      ${ta.support_level:.2f}")
    print(f"  Resistance:   ${ta.resistance_level:.2f}")

    # Recommendation
    print(f"\n📊 RECOMMENDATION: {report.recommendation.value}")
    print("─" * 60)
    print(report.reasoning)

    print("\n" + "═" * 60)


def print_performance_summary(metrics: PerformanceMetrics):
    """
    Print a formatted performance summary to console.

    Args:
        metrics: Performance metrics
    """
    print("\n" + "═" * 60)
    print("📊 PERFORMANCE METRICS")
    print("═" * 60)

    print(f"\nPeriod: {metrics.period}")
    print(f"Predictions Made: {metrics.predictions_made}")

    print("\n📈 Accuracy Metrics")
    print("─" * 40)
    print(f"  Mean Absolute Error:    ${metrics.mae:.2f}")
    print(f"  Root Mean Square Error: ${metrics.rmse:.2f}")
    print(f"  Directional Accuracy:   {metrics.directional_accuracy:.1%}")
    print(f"  Confidence Calibration: {metrics.confidence_calibration:.1%}")

    print("\n🤖 Agent Performance")
    print("─" * 40)
    for ap in metrics.agent_performance:
        print(f"  {ap.agent_name}:")
        print(
            f"    Accuracy: {ap.accuracy:.1%}, "
            f"Impact: {ap.impact_score:.1%}"
        )

    print("\n🏆 Best Factors")
    print("─" * 40)
    for i, f in enumerate(metrics.best_factors, 1):
        print(f"  {i}. {f}")

    print(f"\n📈 Trend: {metrics.improvement_trend}")
    print("═" * 60)
