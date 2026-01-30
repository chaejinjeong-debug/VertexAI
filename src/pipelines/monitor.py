"""
Pipeline Monitor

파이프라인 실행 상태를 실시간으로 모니터링합니다.

Usage:
    python -m src.pipelines.monitor <job_name>
    python -m src.pipelines.monitor customer-churn-training-pipeline-20260130085524
    python -m src.pipelines.monitor --latest
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import PipelineState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "env.yaml"

# State colors (ANSI)
COLORS = {
    "PENDING": "\033[33m",      # Yellow
    "RUNNING": "\033[36m",      # Cyan
    "SUCCEEDED": "\033[32m",    # Green
    "FAILED": "\033[31m",       # Red
    "CANCELLED": "\033[35m",    # Magenta
    "RESET": "\033[0m",
}

# State mapping
STATE_NAMES = {
    0: "UNSPECIFIED",
    1: "QUEUED",
    2: "PENDING",
    3: "RUNNING",
    4: "SUCCEEDED",
    5: "FAILED",
    6: "CANCELLING",
    7: "CANCELLED",
    8: "PAUSED",
    9: "NOT_STARTED",
}


def load_config() -> dict:
    """Load configuration from env.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Monitor Vertex AI Pipeline execution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "job_name",
        type=str,
        nargs="?",
        help="Pipeline job name or resource name"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Monitor the latest pipeline job"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=config["gcp"]["project_id"],
        help="GCP project ID"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=config["gcp"]["region"],
        help="GCP region"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Polling interval in seconds"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    return parser.parse_args()


def get_state_color(state: int, no_color: bool = False) -> tuple[str, str]:
    """Get color codes for state."""
    if no_color:
        return "", ""

    state_name = STATE_NAMES.get(state, "UNKNOWN")
    if state_name in ["SUCCEEDED"]:
        return COLORS["SUCCEEDED"], COLORS["RESET"]
    elif state_name in ["FAILED", "CANCELLED"]:
        return COLORS["FAILED"], COLORS["RESET"]
    elif state_name in ["RUNNING"]:
        return COLORS["RUNNING"], COLORS["RESET"]
    elif state_name in ["PENDING", "QUEUED"]:
        return COLORS["PENDING"], COLORS["RESET"]
    return "", ""


def get_latest_job(project_id: str, region: str) -> str:
    """Get the latest pipeline job name."""
    jobs = aiplatform.PipelineJob.list(
        order_by="create_time desc",
    )

    if not jobs:
        raise ValueError("No pipeline jobs found")

    return jobs[0].resource_name


def format_duration(start_time, end_time=None) -> str:
    """Format duration between two times."""
    if not start_time:
        return "-"

    if end_time:
        duration = end_time - start_time
    else:
        duration = datetime.now(start_time.tzinfo) - start_time

    total_seconds = int(duration.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def print_status(job, no_color: bool = False) -> None:
    """Print pipeline status."""
    resource = job._gca_resource
    state = resource.state
    state_name = STATE_NAMES.get(state, "UNKNOWN")

    start_color, end_color = get_state_color(state, no_color)

    # Clear screen and print header
    print("\033[2J\033[H", end="")  # Clear screen
    print("=" * 70)
    print(f"Pipeline Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Pipeline info
    print(f"Job Name:    {job.display_name}")
    print(f"State:       {start_color}{state_name}{end_color}")

    # Duration
    start_time = resource.start_time if hasattr(resource, 'start_time') else None
    end_time = resource.end_time if hasattr(resource, 'end_time') else None
    print(f"Duration:    {format_duration(start_time, end_time)}")
    print()

    # Task details
    print("-" * 70)
    print(f"{'Task':<30} {'State':<15} {'Duration':<15}")
    print("-" * 70)

    if hasattr(resource, 'job_detail') and resource.job_detail.task_details:
        for task in resource.job_detail.task_details:
            task_state = STATE_NAMES.get(task.state, "UNKNOWN")
            task_start = task.start_time if hasattr(task, 'start_time') else None
            task_end = task.end_time if hasattr(task, 'end_time') else None
            task_duration = format_duration(task_start, task_end) if task_start else "-"

            task_color, task_end_color = get_state_color(task.state, no_color)

            # Truncate long task names
            task_name = task.task_name[:28] + ".." if len(task.task_name) > 30 else task.task_name

            print(f"{task_name:<30} {task_color}{task_state:<15}{task_end_color} {task_duration:<15}")

            # Show error if failed
            if task.error and task.error.message and task.state == 7:  # CANCELLED/FAILED
                error_msg = task.error.message[:60] + "..." if len(task.error.message) > 60 else task.error.message
                print(f"  └─ Error: {COLORS['FAILED'] if not no_color else ''}{error_msg}{COLORS['RESET'] if not no_color else ''}")

    print("-" * 70)
    print()

    # Console URL
    job_id = job.name.split("/")[-1]
    console_url = f"https://console.cloud.google.com/vertex-ai/pipelines/runs/{job_id}?project={job.project}"
    print(f"Console: {console_url}")
    print()
    print("Press Ctrl+C to exit")


def monitor_pipeline(args: argparse.Namespace) -> int:
    """Monitor pipeline execution."""
    # Initialize Vertex AI
    aiplatform.init(
        project=args.project_id,
        location=args.region,
    )

    # Get job resource name
    if args.latest:
        resource_name = get_latest_job(args.project_id, args.region)
        logger.info(f"Monitoring latest job: {resource_name}")
    elif args.job_name:
        if args.job_name.startswith("projects/"):
            resource_name = args.job_name
        else:
            resource_name = f"projects/{args.project_id}/locations/{args.region}/pipelineJobs/{args.job_name}"
    else:
        print("Error: Either job_name or --latest is required")
        return 1

    # Monitor loop
    try:
        while True:
            job = aiplatform.PipelineJob.get(resource_name)
            print_status(job, args.no_color)

            state = job._gca_resource.state
            state_name = STATE_NAMES.get(state, "UNKNOWN")

            # Exit if completed
            if state_name in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                if state_name == "SUCCEEDED":
                    print(f"\n{COLORS['SUCCEEDED']}Pipeline completed successfully!{COLORS['RESET']}")
                    return 0
                else:
                    print(f"\n{COLORS['FAILED']}Pipeline {state_name.lower()}{COLORS['RESET']}")
                    return 1

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        return 0


def main() -> None:
    """Main entry point."""
    args = parse_args()
    sys.exit(monitor_pipeline(args))


if __name__ == "__main__":
    main()
