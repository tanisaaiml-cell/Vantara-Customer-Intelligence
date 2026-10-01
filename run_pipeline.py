"""Rebuild the complete project without manual notebook execution."""

import argparse
import subprocess
import sys


def main() -> None:
    """Run preparation, training, explanations and documentation in order."""
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use Python 3.11: py -3.11 run_pipeline.py")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    stages = [] if args.skip_prepare else ["src.data.prepare"]
    for module in stages + ["src.models.train", "scripts.build_assets", "scripts.build_report"]:
        subprocess.run([sys.executable, "-m", module], check=True)
    print(
        "Pipeline complete. Stop the API and run python -m scripts.refresh_database to replace a previously seeded snapshot."
    )


if __name__ == "__main__":
    main()
