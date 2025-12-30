"""Utils package for Oil Price Prediction Agent"""

from .observability import (
    MetricsCollector,
    Tracer,
    log_agent_complete,
    log_agent_error,
    log_agent_start,
    log_prediction,
    logger,
    metrics,
    print_observability_summary,
    setup_logger,
    timed,
    traced,
    tracer,
)
from .visualization import (
    create_factor_chart,
    create_performance_dashboard,
    create_price_trend_chart,
    print_performance_summary,
    print_prediction_report,
    setup_plotting_style,
)

__all__ = [
    # Observability
    'setup_logger',
    'logger',
    'Tracer',
    'tracer',
    'traced',
    'MetricsCollector',
    'metrics',
    'timed',
    'print_observability_summary',
    'log_agent_start',
    'log_agent_complete',
    'log_agent_error',
    'log_prediction',

    # Visualization
    'setup_plotting_style',
    'create_price_trend_chart',
    'create_factor_chart',
    'create_performance_dashboard',
    'print_prediction_report',
    'print_performance_summary'
]
