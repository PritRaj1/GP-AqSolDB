import argparse
import glob
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gp-kan",
        description="GP-KAN solubility prediction",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("gp", help="Train standard GP model")
    sub.add_parser("kan", help="Train GP-KAN hybrid model")
    sub.add_parser("stats", help="Dataset statistics")
    sub.add_parser("clean", help="Remove generated artifacts")

    args = parser.parse_args()

    if args.command == "gp":
        from src.train_gp import main as run

        run()
    elif args.command == "kan":
        from src.train_kan import main as run

        run()
    elif args.command == "stats":
        from src.stats import main as run

        run()
    elif args.command == "clean":
        _clean()
    else:
        parser.print_help()
        sys.exit(1)


def _clean() -> None:
    patterns = {
        "figures": ["figures/*.png", "figures/*.gif", "tests/figures/*.png"],
        "logs": ["logs/"],
        "config": ["config/*.ini", "config/*.pkl"],
        "build": ["htmlcov/", ".pytest_cache/", "*.egg-info/", "build/", "dist/"],
    }

    removed = 0
    for _category, globs in patterns.items():
        for g in globs:
            full = os.path.join(_ROOT, g)
            for path in glob.glob(full):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
                print(f"  rm {os.path.relpath(path, _ROOT)}")

    if removed == 0:
        print("Nothing to clean.")
    else:
        print(f"Removed {removed} items.")


if __name__ == "__main__":
    main()
