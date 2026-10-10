"""Run the available pipeline stages in order."""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke = ["--smoke"] if args.smoke else []
    commands = [
        [sys.executable, "-m", "src.preprocessing", "--dataset", "all", "--force", *smoke],
        [sys.executable, "-m", "src.train", "--dataset", "all", "--model", "all", "--force", *smoke],
        [sys.executable, "-m", "src.evaluate", "--experiments", "A", "B", "C", "D", "--force", *smoke],
        [sys.executable, "-m", "src.explain", "--experiments", "A", "B", "C", "D", "--models", "mlp_a", "mlp_b", "mlp_c", "rf", "dt", "--methods", "shap", "lime", "--force", *smoke],
        [sys.executable, "-m", "src.compare_xai", *smoke],
    ]
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()