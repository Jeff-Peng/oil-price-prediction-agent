"""
Observability Utilities for Oil Price Prediction Agent

Provides logging, tracing, and metrics collection.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List, Optional

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(
    name: str = "OilPriceAgent",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logger()


# =============================================================================
# TRACING
# =============================================================================

@dataclass
class TraceSpan:
    """A single span in a trace"""
    span_id: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "IN_PROGRESS"
    metadata: Dict = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    children: List['TraceSpan'] = field(default_factory=list)

    def complete(self, status: str = "SUCCESS"):
        """Mark span as complete"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status

    def to_dict(self) -> Dict:
        return {
            "span_id": self.span_id,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class Trace:
    """A complete trace of an operation"""
    trace_id: str
    name: str
    start_time: datetime
    root_span: Optional[TraceSpan] = None
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None

    def complete(self):
        """Mark trace as complete"""
        self.end_time = datetime.now()
        self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
            "root_span": self.root_span.to_dict() if self.root_span else None
        }


class Tracer:
    """Tracing system for tracking operation flow"""

    def __init__(self):
        self.traces: List[Trace] = []
        self.active_trace: Optional[Trace] = None
        self.span_stack: List[TraceSpan] = []
        self._span_counter = 0

    def start_trace(self, name: str) -> Trace:
        """Start a new trace"""
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.traces)}"

        trace = Trace(
            trace_id=trace_id,
            name=name,
            start_time=datetime.now()
        )

        self.active_trace = trace
        self.traces.append(trace)

        logger.info(f"🔍 Started trace: {name} [{trace_id}]")
        return trace

    def start_span(self, operation: str, metadata: Optional[Dict] = None) -> TraceSpan:
        """Start a new span within the current trace"""
        self._span_counter += 1
        span_id = f"span_{self._span_counter}"

        parent_span_id = self.span_stack[-1].span_id if self.span_stack else None

        span = TraceSpan(
            span_id=span_id,
            operation=operation,
            start_time=datetime.now(),
            metadata=metadata or {},
            parent_span_id=parent_span_id
        )

        # Add to parent if exists
        if self.span_stack:
            self.span_stack[-1].children.append(span)
        elif self.active_trace:
            self.active_trace.root_span = span

        self.span_stack.append(span)

        logger.debug(f"  → Started span: {operation}")
        return span

    def end_span(self, status: str = "SUCCESS"):
        """End the current span"""
        if self.span_stack:
            span = self.span_stack.pop()
            span.complete(status)
            logger.debug(f"  ← Ended span: {span.operation} ({span.duration_ms:.1f}ms)")

    def end_trace(self):
        """End the current trace"""
        if self.active_trace:
            self.active_trace.complete()
            logger.info(
                f"✅ Completed trace: {self.active_trace.name} "
                f"[{self.active_trace.total_duration_ms:.1f}ms]"
            )
            self.active_trace = None
            self.span_stack = []

    def get_trace_summary(self) -> List[Dict]:
        """Get summary of all traces"""
        return [t.to_dict() for t in self.traces]


# Global tracer instance
tracer = Tracer()


def traced(operation_name: Optional[str] = None):
    """
    Decorator to automatically trace a function.

    Args:
        operation_name: Optional name for the operation
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            span = tracer.start_span(op_name)

            try:
                result = func(*args, **kwargs)
                tracer.end_span("SUCCESS")
                return result
            except Exception as e:
                tracer.end_span("ERROR")
                span.metadata["error"] = str(e)
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            span = tracer.start_span(op_name)

            try:
                result = await func(*args, **kwargs)
                tracer.end_span("SUCCESS")
                return result
            except Exception as e:
                tracer.end_span("ERROR")
                span.metadata["error"] = str(e)
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class Metric:
    """A single metric measurement"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = {}

    def record(self, name: str, value: float, tags: Optional[Dict] = None, unit: str = ""):
        """Record a metric value"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {},
            unit=unit
        )
        self.metrics.append(metric)

    def increment(self, name: str, value: int = 1):
        """Increment a counter"""
        self.counters[name] = self.counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float):
        """Set a gauge value"""
        self.gauges[name] = value

    def record_time(self, name: str, duration_ms: float):
        """Record a timing measurement"""
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(duration_ms)

    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        timer_stats = {}
        for name, values in self.timers.items():
            if values:
                timer_stats[name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values)
                }

        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "timers": timer_stats,
            "total_metrics": len(self.metrics)
        }

    def reset(self):
        """Reset all metrics"""
        self.metrics = []
        self.counters = {}
        self.gauges = {}
        self.timers = {}


# Global metrics collector
metrics = MetricsCollector()


def timed(metric_name: Optional[str] = None):
    """
    Decorator to time a function and record the duration.

    Args:
        metric_name: Optional name for the metric
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__name__}_duration"
            start = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.record_time(name, duration_ms)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__name__}_duration"
            start = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.record_time(name, duration_ms)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# OBSERVABILITY DASHBOARD
# =============================================================================

def print_observability_summary():
    """Print a summary of all observability data"""
    print("\n" + "=" * 60)
    print("📊 OBSERVABILITY SUMMARY")
    print("=" * 60)

    # Traces
    print("\n🔍 TRACES")
    print("-" * 40)
    for trace in tracer.traces[-5:]:  # Last 5 traces
        status = "✅" if trace.end_time else "🔄"
        duration = f"{trace.total_duration_ms:.1f}ms" if trace.total_duration_ms else "in progress"
        print(f"  {status} {trace.name}: {duration}")

    # Metrics
    print("\n📈 METRICS")
    print("-" * 40)
    summary = metrics.get_summary()

    print("  Counters:")
    for name, value in summary['counters'].items():
        print(f"    {name}: {value}")

    print("  Gauges:")
    for name, value in summary['gauges'].items():
        print(f"    {name}: {value}")

    print("  Timers:")
    for name, stats in summary['timers'].items():
        print(f"    {name}: avg={stats['avg_ms']:.1f}ms, "
              f"min={stats['min_ms']:.1f}ms, max={stats['max_ms']:.1f}ms")

    print("\n" + "=" * 60)


# =============================================================================
# LOG AGENT ACTIVITY
# =============================================================================

def log_agent_start(agent_name: str):
    """Log when an agent starts"""
    logger.info(f"🤖 Starting agent: {agent_name}")
    metrics.increment("agent_starts")
    tracer.start_span(f"agent_{agent_name}")


def log_agent_complete(agent_name: str, findings_count: int = 0):
    """Log when an agent completes"""
    logger.info(f"✅ Agent completed: {agent_name} ({findings_count} findings)")
    metrics.increment("agent_completions")
    tracer.end_span("SUCCESS")


def log_agent_error(agent_name: str, error: str):
    """Log when an agent encounters an error"""
    logger.error(f"❌ Agent error: {agent_name} - {error}")
    metrics.increment("agent_errors")
    tracer.end_span("ERROR")


def log_prediction(prediction_id: str, wti_price: float, confidence: float):
    """Log a prediction"""
    logger.info(f"🎯 Prediction {prediction_id}: WTI=${wti_price:.2f} (confidence={confidence:.0%})")
    metrics.increment("predictions_made")
    metrics.set_gauge("last_wti_prediction", wti_price)
    metrics.set_gauge("last_confidence", confidence)
