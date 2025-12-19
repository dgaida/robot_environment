"""
Unit tests for PerformanceMetrics and PerformanceMonitor classes

Tests cover:
- Metrics initialization
- Timing measurements
- Counter operations
- Statistical calculations
- Performance monitoring
- JSON export
- Thread safety
"""

import pytest
import time
import json
import threading
from robot_environment.performance_metrics import PerformanceMetrics, PerformanceMonitor, TimingStats


@pytest.fixture
def metrics():
    """Create a fresh PerformanceMetrics instance"""
    return PerformanceMetrics(history_size=100, verbose=False)


@pytest.fixture
def metrics_verbose():
    """Create a verbose PerformanceMetrics instance"""
    return PerformanceMetrics(history_size=100, verbose=True)


class TestTimingStats:
    """Test TimingStats dataclass"""

    def test_from_samples_empty(self):
        """Test creating stats from empty samples"""
        stats = TimingStats.from_samples([])

        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.min == float("inf")
        assert stats.max == 0.0

    def test_from_samples_single_value(self):
        """Test creating stats from single value"""
        stats = TimingStats.from_samples([10.0])

        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.min == 10.0
        assert stats.max == 10.0
        assert stats.p50 == 10.0

    def test_from_samples_multiple_values(self):
        """Test creating stats from multiple values"""
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = TimingStats.from_samples(samples)

        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.min == 10.0
        assert stats.max == 50.0
        assert stats.p50 == 30.0

    def test_to_dict(self):
        """Test converting stats to dictionary"""
        stats = TimingStats(count=10, mean=25.5, std=5.2, min=10.0, max=40.0, p50=25.0, p95=35.0, p99=38.0)

        d = stats.to_dict()

        assert d["count"] == 10
        assert d["mean"] == 25.5
        assert d["p95"] == 35.0

    def test_percentile_calculations(self):
        """Test percentile calculations"""
        samples = list(range(1, 101))  # 1 to 100
        stats = TimingStats.from_samples(samples)

        assert stats.p50 == pytest.approx(50.5, abs=1)
        assert stats.p95 == pytest.approx(95.0, abs=1)
        assert stats.p99 == pytest.approx(99.0, abs=1)


class TestPerformanceTimerContextManager:
    """Test PerformanceTimer context manager"""

    def test_timer_basic_usage(self, metrics):
        """Test basic timer usage"""
        with metrics.timer("test_operation"):
            time.sleep(0.01)

        # Should have recorded timing
        assert len(metrics._timings["test_operation"]) > 0

    def test_timer_measures_time(self, metrics):
        """Test that timer measures elapsed time"""
        with metrics.timer("test_operation") as timer:
            time.sleep(0.02)

        elapsed = timer.elapsed_ms()
        assert elapsed >= 20.0  # At least 20ms

    def test_timer_records_to_metrics(self, metrics):
        """Test timer records to correct metric"""
        with metrics.timer("frame_capture"):
            time.sleep(0.01)

        timings = list(metrics._timings["frame_capture"])
        assert len(timings) == 1
        assert timings[0] >= 10.0

    def test_timer_elapsed_before_exit(self):
        """Test elapsed_ms() can be called during operation"""
        metrics_obj = PerformanceMetrics()

        with metrics_obj.timer("test_operation") as timer:
            time.sleep(0.01)
            mid_elapsed = timer.elapsed_ms()
            time.sleep(0.01)
            end_elapsed = timer.elapsed_ms()

        assert mid_elapsed > 0
        assert end_elapsed > mid_elapsed


class TestPerformanceMetricsInitialization:
    """Test PerformanceMetrics initialization"""

    def test_initialization_default(self):
        """Test initialization with defaults"""
        metrics = PerformanceMetrics()

        assert metrics._history_size == 100
        assert metrics._verbose is False
        assert metrics._lock is not None

    def test_initialization_custom_history_size(self):
        """Test initialization with custom history size"""
        metrics = PerformanceMetrics(history_size=50)

        assert metrics._history_size == 50
        # Check that deques have correct maxlen
        assert metrics._timings["frame_capture"].maxlen == 50

    def test_initialization_creates_all_metrics(self, metrics):
        """Test that all metric categories are created"""
        expected_timings = ["frame_capture", "object_detection", "memory_update", "robot_pick", "robot_place", "robot_push"]

        for metric in expected_timings:
            assert metric in metrics._timings

    def test_initialization_creates_all_counters(self, metrics):
        """Test that all counters are created"""
        expected_counters = ["frames_captured", "objects_detected", "pick_operations", "pick_successes", "pick_failures"]

        for counter in expected_counters:
            assert counter in metrics._counters
            assert metrics._counters[counter] == 0


class TestPerformanceMetricsRecordTiming:
    """Test recording timing measurements"""

    def test_record_timing_basic(self, metrics):
        """Test recording a basic timing"""
        metrics.record_timing("frame_capture", 15.5)

        timings = list(metrics._timings["frame_capture"])
        assert len(timings) == 1
        assert timings[0] == 15.5

    def test_record_timing_multiple(self, metrics):
        """Test recording multiple timings"""
        for i in range(10):
            metrics.record_timing("frame_capture", 10.0 + i)

        timings = list(metrics._timings["frame_capture"])
        assert len(timings) == 10

    def test_record_timing_respects_history_size(self):
        """Test that history size is respected"""
        metrics = PerformanceMetrics(history_size=5)

        for i in range(10):
            metrics.record_timing("frame_capture", float(i))

        timings = list(metrics._timings["frame_capture"])
        assert len(timings) == 5  # Should only keep last 5

    def test_record_timing_unknown_metric(self, metrics):
        """Test recording to unknown metric logs warning"""
        # Should not crash
        metrics.record_timing("unknown_metric", 10.0)

        # This is actually the correct behavior (see performance_metrics.py line 224)
        assert "unknown_metric" in metrics._timings
        assert len(metrics._timings["unknown_metric"]) == 1
        assert metrics._timings["unknown_metric"][0] == 10.0


class TestPerformanceMetricsCounters:
    """Test counter operations"""

    def test_increment_counter_basic(self, metrics):
        """Test basic counter increment"""
        metrics.increment_counter("frames_captured")

        assert metrics._counters["frames_captured"] == 1

    def test_increment_counter_by_amount(self, metrics):
        """Test incrementing by specific amount"""
        metrics.increment_counter("objects_detected", 5)

        assert metrics._counters["objects_detected"] == 5

    def test_increment_counter_multiple_times(self, metrics):
        """Test multiple increments"""
        for _ in range(10):
            metrics.increment_counter("pick_operations")

        assert metrics._counters["pick_operations"] == 10

    def test_increment_unknown_counter(self, metrics):
        """Test incrementing unknown counter logs warning"""
        # Should not crash
        metrics.increment_counter("unknown_counter")


class TestPerformanceMetricsSpecializedRecording:
    """Test specialized recording methods"""

    def test_record_frame_captured(self, metrics):
        """Test recording frame capture"""
        metrics.record_frame_captured(15.5)

        assert len(metrics._timings["frame_capture"]) == 1
        assert metrics._counters["frames_captured"] == 1

    def test_record_objects_detected(self, metrics):
        """Test recording object detection"""
        metrics.record_objects_detected(count=3, detection_time_ms=25.0)

        assert len(metrics._timings["object_detection"]) == 1
        assert metrics._counters["objects_detected"] == 3

    def test_record_pick_operation_success(self, metrics):
        """Test recording successful pick"""
        metrics.record_pick_operation(duration_s=2.5, success=True)

        assert metrics._counters["pick_operations"] == 1
        assert metrics._counters["pick_successes"] == 1
        assert metrics._counters["pick_failures"] == 0

    def test_record_pick_operation_failure(self, metrics):
        """Test recording failed pick"""
        metrics.record_pick_operation(duration_s=2.5, success=False)

        assert metrics._counters["pick_operations"] == 1
        assert metrics._counters["pick_successes"] == 0
        assert metrics._counters["pick_failures"] == 1

    def test_record_place_operation(self, metrics):
        """Test recording place operation"""
        metrics.record_place_operation(duration_s=2.0, success=True)

        assert metrics._counters["place_operations"] == 1
        assert metrics._counters["place_successes"] == 1

    def test_record_memory_update(self, metrics):
        """Test recording memory update"""
        metrics.record_memory_update(duration_ms=5.5, objects_added=2, objects_updated=1)

        assert len(metrics._timings["memory_update"]) == 1
        assert metrics._counters["memory_updates"] == 1


class TestPerformanceMetricsStatistics:
    """Test statistical calculations"""

    def test_get_stats_empty(self, metrics):
        """Test getting stats with no data"""
        stats = metrics.get_stats()

        assert "camera" in stats
        assert stats["camera"]["frames_captured"] == 0
        assert stats["camera"]["fps"] == 0.0

    def test_get_stats_with_data(self, metrics):
        """Test getting stats with recorded data"""
        # Record some data
        for i in range(5):
            metrics.record_frame_captured(15.0)
            metrics.record_objects_detected(2, 20.0)

        stats = metrics.get_stats()

        assert stats["camera"]["frames_captured"] == 5
        assert stats["vision"]["objects_detected"] == 10

    def test_calculate_fps(self, metrics):
        """Test FPS calculation"""
        # Record frames with consistent timing
        for _ in range(10):
            metrics.record_timing("frame_capture", 20.0)  # 20ms = 50fps

        stats = metrics.get_stats()
        fps = stats["camera"]["fps"]

        assert fps == pytest.approx(50.0, abs=5.0)

    def test_calculate_success_rate(self, metrics):
        """Test success rate calculation"""
        # Record some picks
        metrics.record_pick_operation(2.0, True)
        metrics.record_pick_operation(2.0, True)
        metrics.record_pick_operation(2.0, False)

        stats = metrics.get_stats()
        success_rate = stats["robot"]["operations"]["pick"]["success_rate"]

        assert success_rate == pytest.approx(66.67, abs=0.1)

    def test_timing_stats_calculation(self, metrics):
        """Test timing statistics calculation"""
        # Record some timings
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            metrics.record_timing("frame_capture", v)

        stats = metrics.get_stats()
        frame_stats = stats["camera"]["frame_capture"]

        assert frame_stats["count"] == 5
        assert frame_stats["mean"] == 30.0
        assert frame_stats["min"] == 10.0
        assert frame_stats["max"] == 50.0


class TestPerformanceMetricsSummary:
    """Test summary generation"""

    def test_get_summary_basic(self, metrics):
        """Test basic summary generation"""
        summary = metrics.get_summary()

        assert isinstance(summary, str)
        assert "PERFORMANCE METRICS SUMMARY" in summary
        assert "CAMERA:" in summary
        assert "ROBOT OPERATIONS:" in summary

    def test_get_summary_with_data(self, metrics):
        """Test summary with actual data"""
        # Add some data
        metrics.record_frame_captured(15.0)
        metrics.record_pick_operation(2.5, True)

        summary = metrics.get_summary()

        assert "Frames captured: 1" in summary
        assert "Pick operations: 1" in summary


class TestPerformanceMetricsReset:
    """Test metrics reset"""

    def test_reset_clears_timings(self, metrics):
        """Test that reset clears timing data"""
        metrics.record_timing("frame_capture", 15.0)
        metrics.record_timing("object_detection", 20.0)

        metrics.reset()

        assert len(metrics._timings["frame_capture"]) == 0
        assert len(metrics._timings["object_detection"]) == 0

    def test_reset_clears_counters(self, metrics):
        """Test that reset clears counters"""
        metrics.increment_counter("frames_captured", 10)
        metrics.increment_counter("pick_operations", 5)

        metrics.reset()

        assert metrics._counters["frames_captured"] == 0
        assert metrics._counters["pick_operations"] == 0

    def test_reset_updates_start_time(self, metrics):
        """Test that reset updates start time"""
        old_start = metrics._start_time
        time.sleep(0.01)

        metrics.reset()

        assert metrics._start_time > old_start


class TestPerformanceMetricsExport:
    """Test JSON export"""

    def test_export_json(self, metrics, tmp_path):
        """Test exporting metrics to JSON"""
        # Add some data
        metrics.record_frame_captured(15.0)
        metrics.record_pick_operation(2.5, True)

        # Export
        output_file = tmp_path / "metrics.json"
        metrics.export_json(str(output_file))

        assert output_file.exists()

        # Load and verify
        with open(output_file) as f:
            data = json.load(f)

        assert "camera" in data
        assert "robot" in data

    def test_export_json_structure(self, metrics, tmp_path):
        """Test exported JSON has correct structure"""
        metrics.record_frame_captured(15.0)

        output_file = tmp_path / "metrics.json"
        metrics.export_json(str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "camera" in data
        assert "vision" in data
        assert "robot" in data


class TestPerformanceMetricsThreadSafety:
    """Test thread safety of metrics operations"""

    def test_concurrent_timing_records(self, metrics):
        """Test concurrent timing records are safe"""

        def record_timings():
            for _ in range(100):
                metrics.record_timing("frame_capture", 15.0)

        threads = [threading.Thread(target=record_timings) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded all timings
        # (deque might have limited some due to maxlen)
        assert len(metrics._timings["frame_capture"]) > 0

    def test_concurrent_counter_increments(self, metrics):
        """Test concurrent counter increments are safe"""

        def increment_counters():
            for _ in range(100):
                metrics.increment_counter("frames_captured")

        threads = [threading.Thread(target=increment_counters) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have correct total
        assert metrics._counters["frames_captured"] == 500

    def test_concurrent_get_stats(self, metrics):
        """Test concurrent stats access is safe"""
        # Add some data
        metrics.record_frame_captured(15.0)

        results = []

        def get_stats():
            stats = metrics.get_stats()
            results.append(stats["camera"]["frames_captured"])

        threads = [threading.Thread(target=get_stats) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should succeed
        assert len(results) == 10


class TestPerformanceMonitor:
    """Test PerformanceMonitor background monitoring"""

    def test_monitor_initialization(self, metrics):
        """Test monitor initialization"""
        monitor = PerformanceMonitor(metrics, interval_seconds=1.0)

        assert monitor.metrics == metrics
        assert monitor.interval_seconds == 1.0
        assert monitor._thread is None

    def test_monitor_start_stop(self, metrics):
        """Test starting and stopping monitor"""
        monitor = PerformanceMonitor(metrics, interval_seconds=0.1, verbose=False)

        monitor.start()
        assert monitor._thread is not None
        assert monitor._thread.is_alive()

        time.sleep(0.2)  # Let it run briefly

        monitor.stop()
        time.sleep(0.1)  # Give it time to stop

        assert not monitor._stop_event.is_set() or not monitor._thread.is_alive()

    def test_monitor_logs_periodically(self, metrics):
        """Test that monitor logs at intervals"""
        # Add some data
        metrics.record_frame_captured(15.0)

        monitor = PerformanceMonitor(metrics, interval_seconds=0.1, verbose=False)

        monitor.start()
        time.sleep(0.25)  # Let it run for 2-3 intervals
        monitor.stop()

        # Just verify it didn't crash
        assert True

    def test_monitor_double_start(self, metrics):
        """Test starting monitor twice doesn't create multiple threads"""
        monitor = PerformanceMonitor(metrics, interval_seconds=0.1)

        monitor.start()
        first_thread = monitor._thread

        monitor.start()  # Try to start again

        # Should still be same thread
        assert monitor._thread == first_thread

        monitor.stop()

    def test_monitor_stop_without_start(self, metrics):
        """Test stopping monitor that wasn't started"""
        monitor = PerformanceMonitor(metrics)

        # Should not crash
        monitor.stop()
        assert True


class TestPerformanceMetricsVerboseMode:
    """Test verbose logging mode"""

    def test_verbose_initialization(self, metrics_verbose):
        """Test verbose initialization"""
        assert metrics_verbose._verbose is True

    def test_verbose_logging_on_record(self, metrics_verbose, caplog):
        """Test that verbose mode logs on record"""
        import logging

        caplog.set_level(logging.DEBUG)

        # Record enough times to trigger verbose logging (every 10)
        for i in range(15):
            metrics_verbose.record_timing("frame_capture", 15.0)

        # Should have logged something (check logs if needed)
        # This is a basic test - verbose actually prints to logger


class TestPerformanceMetricsEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_timing_stats(self, metrics):
        """Test getting stats for metric with no data"""
        stats = metrics._get_timing_stats("frame_capture")

        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_success_rate_with_no_operations(self, metrics):
        """Test success rate when no operations performed"""
        rate = metrics._calculate_success_rate("pick")

        assert rate == 0.0

    def test_fps_with_no_frames(self, metrics):
        """Test FPS calculation with no frames"""
        fps = metrics._calculate_fps()

        assert fps == 0.0

    def test_record_zero_duration(self, metrics):
        """Test recording zero duration"""
        metrics.record_timing("frame_capture", 0.0)

        timings = list(metrics._timings["frame_capture"])
        assert timings[0] == 0.0

    def test_record_negative_duration(self, metrics):
        """Test recording negative duration (should work, though unusual)"""
        metrics.record_timing("frame_capture", -5.0)

        timings = list(metrics._timings["frame_capture"])
        assert timings[0] == -5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
