#!/usr/bin/env python3
"""
Real-time Performance Monitoring Script

Continuously monitors robot_environment performance and prints statistics
to the console every minute. Includes visual indicators and color coding.

Usage:
    python performance_monitor.py [--interval SECONDS] [--verbose] [--compact]

Options:
    --interval SECONDS  Stats print interval (default: 60)
    --verbose          Show detailed statistics
    --compact          Compact output format
    --export PATH      Export stats to JSON file every interval
"""

import sys
import time
import argparse
from datetime import datetime
from typing import Optional

try:
    # Try to import colorama for colored output (optional)
    from colorama import init, Fore, Style

    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False

    # Define dummy color codes if colorama not available
    class Fore:
        RED = GREEN = YELLOW = CYAN = WHITE = MAGENTA = BLUE = ""

    class Style:
        BRIGHT = RESET_ALL = ""


from robot_environment.environment import Environment


class PerformanceMonitor:
    """Real-time performance monitoring with console output."""

    def __init__(
        self,
        env: Environment,
        interval: int = 60,
        verbose: bool = False,
        compact: bool = False,
        export_path: Optional[str] = None,
    ):
        """
        Initialize performance monitor.

        Args:
            env: Environment instance to monitor
            interval: Print interval in seconds
            verbose: Show detailed statistics
            compact: Use compact output format
            export_path: Optional path to export JSON stats
        """
        self.env = env
        self.interval = interval
        self.verbose = verbose
        self.compact = compact
        self.export_path = export_path

        self.metrics = env.get_performance_metrics()
        if not self.metrics:
            print(f"{Fore.RED}⚠️  Performance metrics not enabled in Environment!{Style.RESET_ALL}")
            sys.exit(1)

        self.start_time = time.time()
        self.iteration = 0

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def format_ms(self, ms: float, threshold_warning: float = 100, threshold_error: float = 200) -> str:
        """Format milliseconds with color coding."""
        if not COLORS_AVAILABLE:
            return f"{ms:.1f}ms"

        if ms >= threshold_error:
            return f"{Fore.RED}{ms:.1f}ms{Style.RESET_ALL}"
        elif ms >= threshold_warning:
            return f"{Fore.YELLOW}{ms:.1f}ms{Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}{ms:.1f}ms{Style.RESET_ALL}"

    def format_fps(self, fps: float) -> str:
        """Format FPS with color coding."""
        if not COLORS_AVAILABLE:
            return f"{fps:.1f}"

        if fps < 10:
            return f"{Fore.RED}{fps:.1f}{Style.RESET_ALL}"
        elif fps < 20:
            return f"{Fore.YELLOW}{fps:.1f}{Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}{fps:.1f}{Style.RESET_ALL}"

    def format_percentage(self, pct: float, inverse: bool = False) -> str:
        """Format percentage with color coding."""
        if not COLORS_AVAILABLE:
            return f"{pct:.1f}%"

        # For success rates, higher is better
        # For failure rates, lower is better (inverse=True)
        if inverse:
            thresholds = [(10, Fore.GREEN), (20, Fore.YELLOW), (100, Fore.RED)]
        else:
            thresholds = [(50, Fore.RED), (80, Fore.YELLOW), (100, Fore.GREEN)]

        color = Fore.WHITE
        for threshold, col in thresholds:
            if pct <= threshold:
                color = col
                break

        return f"{color}{pct:.1f}%{Style.RESET_ALL}"

    def print_compact_stats(self, stats: dict) -> None:
        """Print compact one-line statistics."""
        cam = stats["camera"]
        vis = stats["vision"]
        pick = stats["robot"]["operations"]["pick"]
        place = stats["robot"]["operations"]["place"]

        elapsed = time.time() - self.start_time

        line = (
            f"[{self.format_duration(elapsed)}] "
            f"FPS:{self.format_fps(cam['fps'])} "
            f"Frame:{self.format_ms(cam['frame_capture']['mean'], 50, 100)} "
            f"Detect:{self.format_ms(vis['detection_time']['mean'])} "
            f"Objs:{vis['objects_detected']} "
            f"Pick:{pick['count']}({self.format_percentage(pick['success_rate'])}) "
            f"Place:{place['count']}({self.format_percentage(place['success_rate'])})"
        )

        print(line)

    def print_detailed_stats(self, stats: dict) -> None:
        """Print detailed statistics with sections."""
        self.iteration += 1
        elapsed = time.time() - self.start_time

        # Header
        print("\n" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}PERFORMANCE STATISTICS - Iteration #{self.iteration}{Style.RESET_ALL}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | " f"Uptime: {self.format_duration(elapsed)}")
        print("=" * 80)

        # Camera Section
        cam = stats["camera"]
        print(f"\n{Fore.MAGENTA}📷 CAMERA{Style.RESET_ALL}")
        print(f"  Frames captured: {Fore.WHITE}{cam['frames_captured']}{Style.RESET_ALL}")
        print(f"  Current FPS: {self.format_fps(cam['fps'])}")
        print(
            f"  Frame capture time: {self.format_ms(cam['frame_capture']['mean'], 50, 100)} "
            f"(p95: {self.format_ms(cam['frame_capture']['p95'], 50, 100)})"
        )
        if cam["loop_iteration"]["count"] > 0:
            print(
                f"  Loop iteration: {self.format_ms(cam['loop_iteration']['mean'], 100, 200)} "
                f"(p95: {self.format_ms(cam['loop_iteration']['p95'], 100, 200)})"
            )

        # Vision Section
        vis = stats["vision"]
        print(f"\n{Fore.BLUE}👁️  VISION{Style.RESET_ALL}")
        print(f"  Objects detected: {Fore.WHITE}{vis['objects_detected']}{Style.RESET_ALL}")
        if vis["detection_time"]["count"] > 0:
            print(
                f"  Detection time: {self.format_ms(vis['detection_time']['mean'])} "
                f"(p95: {self.format_ms(vis['detection_time']['p95'])})"
            )
        if vis["redis_fetch_time"]["count"] > 0:
            print(f"  Redis fetch time: {self.format_ms(vis['redis_fetch_time']['mean'], 20, 50)}")

        # Memory Section
        mem = stats["memory"]
        print(f"\n{Fore.YELLOW}💾 MEMORY{Style.RESET_ALL}")
        print(
            f"  Updates: {Fore.WHITE}{mem['updates']}{Style.RESET_ALL} | "
            f"Clears: {Fore.WHITE}{mem['clears']}{Style.RESET_ALL}"
        )
        if mem["update_time"]["count"] > 0:
            print(
                f"  Update time: {self.format_ms(mem['update_time']['mean'], 20, 50)} "
                f"(p95: {self.format_ms(mem['update_time']['p95'], 20, 50)})"
            )

        # Robot Operations Section
        robot = stats["robot"]["operations"]
        print(f"\n{Fore.GREEN}🤖 ROBOT OPERATIONS{Style.RESET_ALL}")

        # Pick operations
        pick = robot["pick"]
        if pick["count"] > 0:
            print(
                f"  Pick operations: {Fore.WHITE}{pick['count']}{Style.RESET_ALL} | "
                f"Success rate: {self.format_percentage(pick['success_rate'])}"
            )
            print(
                f"    Duration: {self.format_ms(pick['duration']['mean'], 3000, 5000)} "
                f"(p95: {self.format_ms(pick['duration']['p95'], 3000, 5000)})"
            )
            print(
                f"    Successes: {Fore.GREEN}{pick['successes']}{Style.RESET_ALL} | "
                f"Failures: {Fore.RED}{pick['failures']}{Style.RESET_ALL}"
            )
        else:
            print(f"  Pick operations: {Fore.WHITE}0{Style.RESET_ALL}")

        # Place operations
        place = robot["place"]
        if place["count"] > 0:
            print(
                f"  Place operations: {Fore.WHITE}{place['count']}{Style.RESET_ALL} | "
                f"Success rate: {self.format_percentage(place['success_rate'])}"
            )
            print(
                f"    Duration: {self.format_ms(place['duration']['mean'], 2000, 4000)} "
                f"(p95: {self.format_ms(place['duration']['p95'], 2000, 4000)})"
            )
            print(
                f"    Successes: {Fore.GREEN}{place['successes']}{Style.RESET_ALL} | "
                f"Failures: {Fore.RED}{place['failures']}{Style.RESET_ALL}"
            )
        else:
            print(f"  Place operations: {Fore.WHITE}0{Style.RESET_ALL}")

        # Push operations
        push = robot["push"]
        if push["count"] > 0:
            print(f"  Push operations: {Fore.WHITE}{push['count']}{Style.RESET_ALL}")
            print(f"    Duration: {self.format_ms(push['duration']['mean'], 2000, 4000)}")

        # Movement operations
        movement = stats["robot"]["movement"]
        if movement["observation_pose"]["count"] > 0:
            print(
                f"\n  Movement to observation pose: "
                f"{self.format_ms(movement['observation_pose']['mean'], 3000, 5000)} "
                f"(count: {movement['observation_pose']['count']})"
            )

        # Verbose section
        if self.verbose:
            print(f"\n{Fore.CYAN}📊 DETAILED STATISTICS{Style.RESET_ALL}")

            # Show percentiles
            print("  Frame capture percentiles:")
            print(
                f"    p50: {cam['frame_capture']['p50']:.1f}ms | "
                f"p95: {cam['frame_capture']['p95']:.1f}ms | "
                f"p99: {cam['frame_capture']['p99']:.1f}ms"
            )

            if pick["count"] > 0:
                print("  Pick duration percentiles:")
                print(
                    f"    p50: {pick['duration']['p50']:.0f}ms | "
                    f"p95: {pick['duration']['p95']:.0f}ms | "
                    f"p99: {pick['duration']['p99']:.0f}ms"
                )

            if place["count"] > 0:
                print("  Place duration percentiles:")
                print(
                    f"    p50: {place['duration']['p50']:.0f}ms | "
                    f"p95: {place['duration']['p95']:.0f}ms | "
                    f"p99: {place['duration']['p99']:.0f}ms"
                )

        print("=" * 80)

    def print_stats(self) -> None:
        """Print statistics based on mode."""
        stats = self.env.get_performance_stats()

        if not stats:
            print(f"{Fore.RED}⚠️  No statistics available{Style.RESET_ALL}")
            return

        if self.compact:
            self.print_compact_stats(stats)
        else:
            self.print_detailed_stats(stats)

        # Export if path specified
        if self.export_path:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.export_path.replace(".json", f"_{timestamp}.json")
                self.env.export_performance_metrics(filename)
                print(f"{Fore.CYAN}📁 Exported to: {filename}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Export failed: {e}{Style.RESET_ALL}")

    def run(self) -> None:
        """Main monitoring loop."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Starting Performance Monitor{Style.RESET_ALL}")
        print(f"Interval: {self.interval}s | Mode: {'Compact' if self.compact else 'Detailed'}")
        print("Press Ctrl+C to stop\n")

        if not COLORS_AVAILABLE:
            print("💡 Tip: Install 'colorama' for colored output (pip install colorama)\n")

        try:
            while True:
                self.print_stats()
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Monitoring stopped by user{Style.RESET_ALL}")

            # Print final summary
            print(f"\n{Fore.CYAN}{Style.BRIGHT}FINAL SUMMARY{Style.RESET_ALL}")
            self.print_stats()

        except Exception as e:
            print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time performance monitoring for robot_environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor with default 60 second interval
  python performance_monitor.py

  # Monitor every 30 seconds with compact output
  python performance_monitor.py --interval 30 --compact

  # Detailed monitoring with verbose output
  python performance_monitor.py --verbose

  # Export stats to JSON every minute
  python performance_monitor.py --export stats.json
        """,
    )

    parser.add_argument("--interval", type=int, default=60, help="Statistics print interval in seconds (default: 60)")

    parser.add_argument("--verbose", action="store_true", help="Show detailed statistics including percentiles")

    parser.add_argument("--compact", action="store_true", help="Use compact one-line output format")

    parser.add_argument("--export", type=str, metavar="PATH", help="Export statistics to JSON file at each interval")

    parser.add_argument("--robot-id", type=str, default="niryo", choices=["niryo", "widowx"], help="Robot ID (default: niryo)")

    parser.add_argument("--simulation", action="store_true", help="Use simulation mode")

    parser.add_argument("--no-camera-thread", action="store_true", help="Do not start camera update thread")

    args = parser.parse_args()

    # Validate arguments
    if args.interval < 1:
        print(f"{Fore.RED}Error: Interval must be at least 1 second{Style.RESET_ALL}")
        sys.exit(1)

    # Initialize environment
    print(f"{Fore.CYAN}Initializing robot environment...{Style.RESET_ALL}")

    try:
        env = Environment(
            el_api_key="",  # Empty for monitoring only
            use_simulation=args.simulation,
            robot_id=args.robot_id,
            verbose=False,
            start_camera_thread=not args.no_camera_thread,
            enable_performance_monitoring=True,
            performance_log_interval=args.interval,
        )

        print(f"{Fore.GREEN}✓ Environment initialized{Style.RESET_ALL}")

        # Create and run monitor
        monitor = PerformanceMonitor(
            env=env, interval=args.interval, verbose=args.verbose, compact=args.compact, export_path=args.export
        )

        monitor.run()

    except Exception as e:
        print(f"\n{Fore.RED}❌ Failed to initialize environment: {e}{Style.RESET_ALL}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if "env" in locals():
            print(f"\n{Fore.CYAN}Cleaning up...{Style.RESET_ALL}")
            env.cleanup()
            print(f"{Fore.GREEN}✓ Cleanup complete{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
