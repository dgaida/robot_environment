"""
Example: Using Performance Metrics with robot_environment

This example demonstrates how to use the performance monitoring system
to track and analyze robot system performance.
"""

from robot_environment.environment import Environment
from robot_workspace import Location
import time

# ============================================================================
# Example 1: Basic Performance Monitoring
# ============================================================================


def example_basic_monitoring():
    """Basic example with automatic performance monitoring."""

    print("=" * 70)
    print("Example 1: Basic Performance Monitoring")
    print("=" * 70)

    # Create environment with performance monitoring enabled (default)
    env = Environment(
        el_api_key="your_key",
        use_simulation=True,
        robot_id="niryo",
        verbose=False,
        start_camera_thread=True,
        enable_performance_monitoring=True,  # Enabled by default
        performance_log_interval=30.0,  # Log every 30 seconds
    )

    # Perform some operations
    print("\nPerforming pick-and-place operations...")

    robot = env.robot()

    # Move to observation pose
    env.robot_move2observation_pose(env.get_workspace_home_id())
    time.sleep(2)

    # Do some pick-and-place operations
    for i in range(3):
        success = robot.pick_place_object(
            object_name="cube",
            pick_coordinate=[0.2, 0.05 + i * 0.05],
            place_coordinate=[0.25, -0.05 - i * 0.05],
            location=Location.RIGHT_NEXT_TO,
        )
        print(f"Operation {i+1}: {'Success' if success else 'Failed'}")
        time.sleep(1)

    # Get performance statistics
    print("\n" + "=" * 70)
    print("PERFORMANCE STATISTICS")
    print("=" * 70)

    stats = env.get_performance_stats()

    if stats:
        # Camera performance
        print("\nCamera:")
        print(f"  Frames captured: {stats['camera']['frames_captured']}")
        print(f"  Current FPS: {stats['camera']['fps']:.1f}")
        print(f"  Avg frame time: {stats['camera']['frame_capture']['mean']:.1f}ms")
        print(f"  P95 frame time: {stats['camera']['frame_capture']['p95']:.1f}ms")

        # Vision performance
        print("\nVision:")
        print(f"  Objects detected: {stats['vision']['objects_detected']}")
        print(f"  Avg detection time: {stats['vision']['detection_time']['mean']:.1f}ms")

        # Robot operations
        pick_stats = stats["robot"]["operations"]["pick"]
        print("\nRobot Pick Operations:")
        print(f"  Total: {pick_stats['count']}")
        print(f"  Success rate: {pick_stats['success_rate']:.1f}%")
        print(f"  Avg duration: {pick_stats['duration']['mean']:.0f}ms")
        print(f"  P95 duration: {pick_stats['duration']['p95']:.0f}ms")

        place_stats = stats["robot"]["operations"]["place"]
        print("\nRobot Place Operations:")
        print(f"  Total: {place_stats['count']}")
        print(f"  Success rate: {place_stats['success_rate']:.1f}%")
        print(f"  Avg duration: {place_stats['duration']['mean']:.0f}ms")

    # Print full summary
    print("\n" + "=" * 70)
    print("FULL PERFORMANCE SUMMARY")
    print("=" * 70)
    env.print_performance_summary()

    # Export to file
    env.export_performance_metrics("performance_report.json")
    print("\nDetailed metrics exported to performance_report.json")

    # Cleanup
    env.cleanup()


# ============================================================================
# Example 2: Custom Performance Tracking
# ============================================================================


def example_custom_tracking():
    """Example with custom performance tracking."""

    print("\n" + "=" * 70)
    print("Example 2: Custom Performance Tracking")
    print("=" * 70)

    env = Environment(
        el_api_key="your_key",
        use_simulation=False,
        robot_id="niryo",
        verbose=False,
        start_camera_thread=True,
        enable_performance_monitoring=True,
    )

    # Get metrics object for custom tracking
    metrics = env.get_performance_metrics()

    if metrics:
        # Track custom operation
        print("\nTracking custom operation...")
        with metrics.timer("custom_operation"):
            # Simulate some work
            env.robot_move2observation_pose(env.get_workspace_home_id())
            time.sleep(2)
            objects = env.get_detected_objects_from_memory()
            print(f"Detected {len(objects)} objects")

        # Manually record metrics
        metrics.record_timing("my_custom_metric", 123.45)
        metrics.increment_counter("custom_counter", 5)

        print("\nCustom metrics recorded")

    # Get stats after custom operations
    stats = env.get_performance_stats()
    if stats:
        print(f"\nSystem uptime: {stats['uptime_seconds']:.1f}s")
        print(f"Frames captured: {stats['camera']['frames_captured']}")
        print(f"Objects detected: {stats['vision']['objects_detected']}")

    env.cleanup()


# ============================================================================
# Example 3: Performance Comparison
# ============================================================================


def example_performance_comparison():
    """Compare performance between different configurations."""

    print("\n" + "=" * 70)
    print("Example 3: Performance Comparison")
    print("=" * 70)

    env = Environment(el_api_key="your_key", use_simulation=False, robot_id="niryo", verbose=False, start_camera_thread=True)

    metrics = env.get_performance_metrics()

    if not metrics:
        print("Metrics not available")
        return

    # Test 1: Fast operations
    print("\nTest 1: Performing 10 fast pick-place operations...")
    metrics.reset()

    robot = env.robot()
    env.robot_move2observation_pose(env.get_workspace_home_id())
    time.sleep(2)

    for i in range(10):
        robot.pick_place_object(
            object_name="cube", pick_coordinate=[0.2, 0.0], place_coordinate=[0.25, 0.0], location=Location.NONE
        )

    stats1 = env.get_performance_stats()

    # Test 2: Complex operations with location
    print("\nTest 2: Performing 10 complex pick-place operations...")
    metrics.reset()

    for i in range(10):
        robot.pick_place_object(
            object_name="cube", pick_coordinate=[0.2, 0.0], place_coordinate=[0.25, 0.05], location=Location.RIGHT_NEXT_TO
        )

    stats2 = env.get_performance_stats()

    # Compare results
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    if stats1 and stats2:
        print("\nTest 1 (Simple placement):")
        print(f"  Avg pick time: {stats1['robot']['operations']['pick']['duration']['mean']:.0f}ms")
        print(f"  Avg place time: {stats1['robot']['operations']['place']['duration']['mean']:.0f}ms")
        print(f"  Success rate: {stats1['robot']['operations']['pick']['success_rate']:.0f}%")

        print("\nTest 2 (Location-based placement):")
        print(f"  Avg pick time: {stats2['robot']['operations']['pick']['duration']['mean']:.0f}ms")
        print(f"  Avg place time: {stats2['robot']['operations']['place']['duration']['mean']:.0f}ms")
        print(f"  Success rate: {stats2['robot']['operations']['pick']['success_rate']:.0f}%")

        # Calculate differences
        pick_diff = (
            stats2["robot"]["operations"]["pick"]["duration"]["mean"]
            - stats1["robot"]["operations"]["pick"]["duration"]["mean"]
        )
        place_diff = (
            stats2["robot"]["operations"]["place"]["duration"]["mean"]
            - stats1["robot"]["operations"]["place"]["duration"]["mean"]
        )

        print("\nDifference:")
        print(f"  Pick time: {pick_diff:+.0f}ms")
        print(f"  Place time: {place_diff:+.0f}ms")

    env.cleanup()


# ============================================================================
# Example 4: Real-time Monitoring Dashboard
# ============================================================================


def example_realtime_dashboard():
    """Example of real-time performance monitoring."""

    print("\n" + "=" * 70)
    print("Example 4: Real-time Monitoring Dashboard")
    print("=" * 70)

    env = Environment(
        el_api_key="your_key",
        use_simulation=False,
        robot_id="niryo",
        verbose=False,
        start_camera_thread=True,
        performance_log_interval=10.0,  # Log every 10 seconds
    )

    metrics = env.get_performance_metrics()

    if not metrics:
        print("Metrics not available")
        return

    print("\nMonitoring system performance for 60 seconds...")
    print("Performance summary will be logged every 10 seconds")
    print("Press Ctrl+C to stop early\n")

    try:
        # Let system run and collect metrics
        env.robot()
        env.robot_move2observation_pose(env.get_workspace_home_id())

        # Simulate continuous operation
        for i in range(60):
            time.sleep(1)

            # Print live stats every 5 seconds
            if i % 5 == 0 and i > 0:
                stats = env.get_performance_stats()
                if stats:
                    print(
                        f"[{i}s] FPS: {stats['camera']['fps']:.1f}, "
                        f"Objects: {stats['vision']['objects_detected']}, "
                        f"Pick ops: {stats['robot']['operations']['pick']['count']}"
                    )

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL PERFORMANCE SUMMARY")
    print("=" * 70)
    env.print_performance_summary()

    env.cleanup()


# ============================================================================
# Example 5: Performance Optimization
# ============================================================================


def example_performance_optimization():
    """Example showing how to identify and optimize bottlenecks."""

    print("\n" + "=" * 70)
    print("Example 5: Performance Optimization")
    print("=" * 70)

    env = Environment(el_api_key="your_key", use_simulation=False, robot_id="niryo", verbose=True, start_camera_thread=True)

    metrics = env.get_performance_metrics()

    if not metrics:
        print("Metrics not available")
        return

    # Run system for a while to collect data
    print("\nCollecting performance data...")

    robot = env.robot()
    env.robot_move2observation_pose(env.get_workspace_home_id())
    time.sleep(5)

    # Perform various operations
    for i in range(5):
        robot.pick_place_object(
            object_name="cube", pick_coordinate=[0.2, 0.0], place_coordinate=[0.25, 0.0], location=Location.RIGHT_NEXT_TO
        )

    # Analyze performance
    stats = env.get_performance_stats()

    if stats:
        print("\n" + "=" * 70)
        print("PERFORMANCE ANALYSIS")
        print("=" * 70)

        # Identify bottlenecks
        bottlenecks = []

        # Check frame rate
        if stats["camera"]["fps"] < 20:
            bottlenecks.append(f"Low frame rate: {stats['camera']['fps']:.1f} FPS " f"(target: 20+ FPS)")

        # Check detection time
        if stats["vision"]["detection_time"]["mean"] > 100:
            bottlenecks.append(
                f"Slow object detection: {stats['vision']['detection_time']['mean']:.0f}ms " f"(target: <100ms)"
            )

        # Check pick/place times
        if stats["robot"]["operations"]["pick"]["duration"]["mean"] > 5000:
            bottlenecks.append(
                f"Slow pick operations: {stats['robot']['operations']['pick']['duration']['mean']:.0f}ms " f"(target: <5000ms)"
            )

        # Check success rates
        if stats["robot"]["operations"]["pick"]["success_rate"] < 90:
            bottlenecks.append(
                f"Low pick success rate: {stats['robot']['operations']['pick']['success_rate']:.0f}% " f"(target: >90%)"
            )

        if bottlenecks:
            print("\n⚠️  Performance Bottlenecks Identified:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"  {i}. {bottleneck}")

            print("\n💡 Optimization Suggestions:")
            print("  - Reduce detection model complexity")
            print("  - Increase camera update interval")
            print("  - Optimize workspace configuration")
            print("  - Check robot calibration")
        else:
            print("\n✅ System performance is within optimal ranges!")

        # Show detailed timing breakdown
        print("\n" + "=" * 70)
        print("DETAILED TIMING BREAKDOWN")
        print("=" * 70)

        print("\nCamera Loop (per iteration):")
        print(f"  Mean: {stats['camera']['loop_iteration']['mean']:.1f}ms")
        print(f"  P95: {stats['camera']['loop_iteration']['p95']:.1f}ms")

        print("\nObject Detection Pipeline:")
        print(f"  Detection: {stats['vision']['detection_time']['mean']:.1f}ms")
        print(f"  Redis fetch: {stats['vision']['redis_fetch_time']['mean']:.1f}ms")
        print(f"  Memory update: {stats['memory']['update_time']['mean']:.1f}ms")

        print("\nRobot Operations:")
        print(f"  Pick: {stats['robot']['operations']['pick']['duration']['mean']:.0f}ms")
        print(f"  Place: {stats['robot']['operations']['place']['duration']['mean']:.0f}ms")
        print(f"  Move to observation: {stats['robot']['movement']['observation_pose']['mean']:.0f}ms")

    # Export detailed report
    env.export_performance_metrics("optimization_report.json")
    print("\n📊 Detailed report exported to optimization_report.json")

    env.cleanup()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\nRobot Environment - Performance Metrics Examples")
    print("=" * 70)

    # Run examples
    try:
        example_basic_monitoring()
        # example_custom_tracking()
        # example_performance_comparison()
        # example_realtime_dashboard()
        # example_performance_optimization()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
