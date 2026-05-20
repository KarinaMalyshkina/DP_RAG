"""Clone the original DP-RAG baseline repositories into baselines/."""

import subprocess
import sys
from pathlib import Path

BASELINES_DIR = Path(__file__).parent / "baselines"

REPOS = {
    "koga": "https://github.com/tacchan7412/DPRAG.git",
    "grislain": "https://github.com/sarus-tech/dp-rag.git",
}


def clone_repo(name: str, url: str) -> None:
    dest = BASELINES_DIR / name
    if dest.exists():
        print(f"[skip] {name}: already cloned at {dest}")
        return
    print(f"[clone] {name}: {url} -> {dest}")
    subprocess.run(["git", "clone", url, str(dest)], check=True)
    print(f"[done] {name}")


def main() -> None:
    BASELINES_DIR.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        clone_repo(name, url)

    print()
    print("Baselines cloned successfully.")
    print()
    print("Next steps:")
    print("  1. Install this project in dev mode:")
    print('     pip install -e ".[dev]"')
    print()
    print("  2. Run tests:")
    print("     pytest tests/ -v")


if __name__ == "__main__":
    main()
