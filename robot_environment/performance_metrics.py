"""
Performance monitoring and metrics for robot_environment package.

Provides comprehensive tracking of system performance including:
- Frame capture rates
- Object detection latency
- Robot operation durations
- Memory update times
- Redis communication latency
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from time import perf_counter
from collections import deque
import numpy as np
import threading
import json
from datetime import datetime
import logging


@dataclass
class TimingStats:
    """Statistical summary of timing measurements."""

    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = float("inf")
    max: float = 0.0
    p50: float = 0.0  # Median
    p95: float = 0.0
    p99: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "TimingStats":
        """Calculate statistics from a list of samples."""
        if not samples:
            return cls()

        arr = np.array(samples)
        return cls(
            count=len(samples),
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, metrics: "PerformanceMetrics", metric_name: str):
        self.metrics = metrics
        self.metric_name = metric_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        self.metrics.record_timing(self.metric_name, duration_ms)
        return False

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.end_time is None:
            return (perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


class PerformanceMetrics:
    """
    Centralized performance monitoring for robot_environment.

    Tracks timing and throughput metrics for all major operations:
    - Camera frame capture
    - Object detection
    - Robot movements (pick, place, push)
    - Memory updates
    - Redis communication

    Example:
        metrics = PerformanceMetrics(history_size=100)

        # Using context manager
        with metrics.timer('frame_capture'):
            frame = capture_frame()

        # Manual recording
        metrics.record_timing('detection', 45.2)

        # Get statistics
        stats = metrics.get_stats()
        print(f"Average FPS: {stats['camera']['fps']:.1f}")
    """

    def __init__(self, history_size: int = 100, verbose: bool = False):
        """
        Initialize performance metrics tracker.

        Args:
            history_size: Number of samples to keep for each metric
            verbose: Enable verbose logging of metrics
        """
        self._history_size = history_size
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)

        # Thread safety
        self._lock = threading.Lock()

        # Timing metrics (stored as deques for efficient rolling windows)
        self._timings: Dict[str, deque] = {
            # Camera operations
            "frame_capture": deque(maxlen=history_size),
            "frame_publish": deque(maxlen=history_size),
            # Vision operations
            "object_detection": deque(maxlen=history_size),
            "object_fetch_redis": deque(maxlen=history_size),
            # Memory operations
            "memory_update": deque(maxlen=history_size),
            "memory_get": deque(maxlen=history_size),
            "memory_clear": deque(maxlen=history_size),
            # Robot operations
            "robot_pick": deque(maxlen=history_size),
            "robot_place": deque(maxlen=history_size),
            "robot_push": deque(maxlen=history_size),
            "robot_move_observation": deque(maxlen=history_size),
            "robot_get_pose": deque(maxlen=history_size),
            # High-level operations
            "pick_place_total": deque(maxlen=history_size),
            "camera_loop_iteration": deque(maxlen=history_size),
            # Communication
            "redis_publish": deque(maxlen=history_size),
            "redis_fetch": deque(maxlen=history_size),
        }

        # Counter metrics
        self._counters: Dict[str, int] = {
            "frames_captured": 0,
            "objects_detected": 0,
            "pick_operations": 0,
            "place_operations": 0,
            "push_operations": 0,
            "pick_successes": 0,
            "place_successes": 0,
            "pick_failures": 0,
            "place_failures": 0,
            "memory_updates": 0,
            "memory_clears": 0,
        }

        # Start time for uptime tracking
        self._start_time = datetime.now()

        # Last values for rate calculations
        self._last_frame_time = None
        self._last_stats_time = perf_counter()

        if verbose:
            self._logger.setLevel(logging.DEBUG)

    def timer(self, metric_name: str) -> PerformanceTimer:
        """
        Create a context manager timer for an operation.

        Args:
            metric_name: Name of the metric to record

        Returns:
            PerformanceTimer context manager

        Example:
            with metrics.timer('frame_capture'):
                frame = camera.get_frame()
        """
        return PerformanceTimer(self, metric_name)

    def record_timing(self, metric_name: str, duration_ms: float) -> None:
        """
        Record a timing measurement.

        Args:
            metric_name: Name of the metric
            duration_ms: Duration in milliseconds
        """
        with self._lock:
            if metric_name in self._timings:
                self._timings[metric_name].append(duration_ms)

                if self._verbose and len(self._timings[metric_name]) % 10 == 0:
                    stats = TimingStats.from_samples(list(self._timings[metric_name]))
                    self._logger.debug(
                        f"{metric_name}: {duration_ms:.1f}ms " f"(avg: {stats.mean:.1f}ms, p95: {stats.p95:.1f}ms)"
                    )
            else:
                # FIX: Create the metric on-the-fly if it doesn't exist
                from collections import deque

                self._timings[metric_name] = deque([duration_ms], maxlen=self._history_size)
                self._logger.debug(f"Created new timing metric: {metric_name}")

    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """
        Increment a counter metric.

        Args:
            counter_name: Name of the counter
            amount: Amount to increment by (default: 1)
        """
        with self._lock:
            if counter_name in self._counters:
                self._counters[counter_name] += amount
            else:
                self._logger.warning(f"Unknown counter: {counter_name}")

    def record_frame_captured(self, duration_ms: float) -> None:
        """Record a frame capture event."""
        self.record_timing("frame_capture", duration_ms)
        self.increment_counter("frames_captured")
        self._last_frame_time = perf_counter()

    def record_objects_detected(self, count: int, detection_time_ms: float) -> None:
        """Record object detection results."""
        self.record_timing("object_detection", detection_time_ms)
        self.increment_counter("objects_detected", count)

    def record_pick_operation(self, duration_s: float, success: bool) -> None:
        """Record a pick operation."""
        self.record_timing("robot_pick", duration_s * 1000)
        self.increment_counter("pick_operations")
        if success:
            self.increment_counter("pick_successes")
        else:
            self.increment_counter("pick_failures")

    def record_place_operation(self, duration_s: float, success: bool) -> None:
        """Record a place operation."""
        self.record_timing("robot_place", duration_s * 1000)
        self.increment_counter("place_operations")
        if success:
            self.increment_counter("place_successes")
        else:
            self.increment_counter("place_failures")

    def record_memory_update(self, duration_ms: float, objects_added: int, objects_updated: int) -> None:
        """Record a memory update operation."""
        self.record_timing("memory_update", duration_ms)
        self.increment_counter("memory_updates")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics.

        Returns:
            Dictionary containing all performance metrics and statistics
        """
        with self._lock:
            # current_time = perf_counter()
            uptime_seconds = (datetime.now() - self._start_time).total_seconds()

            stats = {
                "uptime_seconds": uptime_seconds,
                "timestamp": datetime.now().isoformat(),
                # Camera metrics
                "camera": {
                    "frames_captured": self._counters["frames_captured"],
                    "fps": self._calculate_fps(),
                    "frame_capture": self._get_timing_stats("frame_capture"),
                    "frame_publish": self._get_timing_stats("frame_publish"),
                    "loop_iteration": self._get_timing_stats("camera_loop_iteration"),
                },
                # Vision metrics
                "vision": {
                    "objects_detected": self._counters["objects_detected"],
                    "detection_time": self._get_timing_stats("object_detection"),
                    "redis_fetch_time": self._get_timing_stats("object_fetch_redis"),
                },
                # Memory metrics
                "memory": {
                    "updates": self._counters["memory_updates"],
                    "clears": self._counters["memory_clears"],
                    "update_time": self._get_timing_stats("memory_update"),
                    "get_time": self._get_timing_stats("memory_get"),
                },
                # Robot operation metrics
                "robot": {
                    "operations": {
                        "pick": {
                            "count": self._counters["pick_operations"],
                            "successes": self._counters["pick_successes"],
                            "failures": self._counters["pick_failures"],
                            "success_rate": self._calculate_success_rate("pick"),
                            "duration": self._get_timing_stats("robot_pick"),
                        },
                        "place": {
                            "count": self._counters["place_operations"],
                            "successes": self._counters["place_successes"],
                            "failures": self._counters["place_failures"],
                            "success_rate": self._calculate_success_rate("place"),
                            "duration": self._get_timing_stats("robot_place"),
                        },
                        "push": {
                            "count": self._counters["push_operations"],
                            "duration": self._get_timing_stats("robot_push"),
                        },
                    },
                    "movement": {
                        "observation_pose": self._get_timing_stats("robot_move_observation"),
                        "get_pose": self._get_timing_stats("robot_get_pose"),
                    },
                    "pick_place_total": self._get_timing_stats("pick_place_total"),
                },
                # Communication metrics
                "communication": {
                    "redis_publish": self._get_timing_stats("redis_publish"),
                    "redis_fetch": self._get_timing_stats("redis_fetch"),
                },
            }

            return stats

    def get_summary(self) -> str:
        """
        Get a human-readable summary of performance metrics.

        Returns:
            Formatted string with key performance indicators
        """
        stats = self.get_stats()

        lines = [
            "=" * 70,
            "PERFORMANCE METRICS SUMMARY",
            "=" * 70,
            f"Uptime: {stats['uptime_seconds']:.1f}s",
            "",
            "CAMERA:",
            f"  Frames captured: {stats['camera']['frames_captured']}",
            f"  Current FPS: {stats['camera']['fps']:.1f}",
            f"  Frame capture time: {stats['camera']['frame_capture']['mean']:.1f}ms "
            f"(p95: {stats['camera']['frame_capture']['p95']:.1f}ms)",
            "",
            "VISION:",
            f"  Objects detected: {stats['vision']['objects_detected']}",
            f"  Detection time: {stats['vision']['detection_time']['mean']:.1f}ms "
            f"(p95: {stats['vision']['detection_time']['p95']:.1f}ms)",
            "",
            "ROBOT OPERATIONS:",
            f"  Pick operations: {stats['robot']['operations']['pick']['count']} "
            f"(success rate: {stats['robot']['operations']['pick']['success_rate']:.1f}%)",
            f"  Pick duration: {stats['robot']['operations']['pick']['duration']['mean']:.0f}ms",
            f"  Place operations: {stats['robot']['operations']['place']['count']} "
            f"(success rate: {stats['robot']['operations']['place']['success_rate']:.1f}%)",
            f"  Place duration: {stats['robot']['operations']['place']['duration']['mean']:.0f}ms",
            "",
            "MEMORY:",
            f"  Updates: {stats['memory']['updates']}",
            f"  Update time: {stats['memory']['update_time']['mean']:.1f}ms",
            "=" * 70,
        ]

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for timing_deque in self._timings.values():
                timing_deque.clear()

            for counter_name in self._counters:
                self._counters[counter_name] = 0

            self._start_time = datetime.now()
            self._last_frame_time = None

            self._logger.info("Performance metrics reset")

    def export_json(self, filepath: str) -> None:
        """
        Export metrics to JSON file.

        Args:
            filepath: Path to output file
        """
        stats = self.get_stats()
        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)

        self._logger.info(f"Metrics exported to {filepath}")

    # Private helper methods

    def _get_timing_stats(self, metric_name: str) -> Dict[str, float]:
        """Get timing statistics for a metric."""
        if metric_name not in self._timings:
            return TimingStats().to_dict()

        samples = list(self._timings[metric_name])
        return TimingStats.from_samples(samples).to_dict()

    def _calculate_fps(self) -> float:
        """Calculate current frames per second."""
        frame_times = list(self._timings["frame_capture"])
        if not frame_times:
            return 0.0

        # Use recent samples for more accurate instantaneous FPS
        recent_samples = frame_times[-10:] if len(frame_times) >= 10 else frame_times
        avg_time_ms = np.mean(recent_samples)

        if avg_time_ms > 0:
            return 1000.0 / avg_time_ms
        return 0.0

    def _calculate_success_rate(self, operation: str) -> float:
        """Calculate success rate for an operation."""
        total = self._counters.get(f"{operation}_operations", 0)
        if total == 0:
            return 0.0

        successes = self._counters.get(f"{operation}_successes", 0)
        return (successes / total) * 100.0


class PerformanceMonitor:
    """
    Background monitor that periodically logs performance metrics.

    Example:
        metrics = PerformanceMetrics()
        monitor = PerformanceMonitor(metrics, interval_seconds=30)
        monitor.start()

        # ... do work ...

        monitor.stop()
    """

    def __init__(self, metrics: PerformanceMetrics, interval_seconds: float = 60.0, verbose: bool = True):
        """
        Initialize performance monitor.

        Args:
            metrics: PerformanceMetrics instance to monitor
            interval_seconds: Logging interval in seconds
            verbose: Enable verbose logging
        """
        self.metrics = metrics
        self.interval_seconds = interval_seconds
        self.verbose = verbose
        self._logger = logging.getLogger(__name__)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            self._logger.warning("Monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        self._logger.info(f"Performance monitor started (interval: {self.interval_seconds}s)")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)

        self._logger.info("Performance monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.wait(self.interval_seconds):
            try:
                if self.verbose:
                    summary = self.metrics.get_summary()
                    self._logger.info(f"\n{summary}")
                else:
                    stats = self.metrics.get_stats()
                    self._logger.info(
                        f"Performance: FPS={stats['camera']['fps']:.1f}, "
                        f"Pick rate={stats['robot']['operations']['pick']['success_rate']:.0f}%, "
                        f"Objects={stats['vision']['objects_detected']}"
                    )
            except Exception as e:
                self._logger.error(f"Error in monitor loop: {e}", exc_info=True)


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create metrics tracker
    metrics = PerformanceMetrics(history_size=100, verbose=True)

    # Simulate some operations
    print("Simulating operations...")

    # Frame captures
    for i in range(50):
        with metrics.timer("frame_capture"):
            import time

            time.sleep(0.015)  # Simulate 15ms frame capture

        if i % 5 == 0:
            metrics.record_objects_detected(count=3, detection_time_ms=25.0)

    # Pick/place operations
    metrics.record_pick_operation(duration_s=2.5, success=True)
    metrics.record_place_operation(duration_s=2.0, success=True)
    metrics.record_pick_operation(duration_s=2.3, success=False)

    # Memory updates
    metrics.record_memory_update(duration_ms=5.2, objects_added=2, objects_updated=1)

    # Print summary
    print("\n" + metrics.get_summary())

    # Export to JSON
    metrics.export_json("performance_metrics.json")
    print("\nMetrics exported to performance_metrics.json")

    # Test monitor
    print("\nTesting performance monitor (10 seconds)...")
    monitor = PerformanceMonitor(metrics, interval_seconds=3, verbose=True)
    monitor.start()

    import time

    time.sleep(10)

    monitor.stop()
    print("Monitor test complete")
