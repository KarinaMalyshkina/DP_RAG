"""Clone the original DP-RAG baseline repositories into baselines/ and apply our patches.

The baselines/ directory is gitignored (it holds clones of external repos), so our
local edits to those clones are not tracked by this project's git. To keep them from
being lost on a fresh clone, the edits are stored as patch files under patches/ and
re-applied here after cloning.

Patches are applied idempotently: re-running this script will not fail if a patch is
already applied.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES_DIR = ROOT / "baselines"
PATCHES_DIR = ROOT / "patches"

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


def apply_patch(name: str) -> None:
    """Apply patches/{name}.patch inside baselines/{name}, idempotently.

    Uses `git apply` from within the clone. If the patch is already applied
    (detected via a reverse-check), it is skipped instead of erroring.
    """
    patch = PATCHES_DIR / f"{name}.patch"
    repo = BASELINES_DIR / name
    if not patch.exists():
        print(f"[patch] {name}: no patch file at {patch}, skipping")
        return
    if not repo.exists():
        print(f"[patch] {name}: clone missing at {repo}, skipping")
        return

    def git_apply(*flags: str) -> bool:
        result = subprocess.run(
            ["git", "apply", *flags, str(patch)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    # Already applied? (the patch reverses cleanly => its changes are present)
    if git_apply("--reverse", "--check"):
        print(f"[patch] {name}: already applied, skipping")
        return
    # Applies cleanly to the pristine clone?
    if git_apply("--check"):
        subprocess.run(["git", "apply", str(patch)], cwd=repo, check=True)
        print(f"[patch] {name}: applied {patch.name}")
        return
    print(
        f"[patch] {name}: WARNING — patch did not apply cleanly. "
        f"The upstream repo may have changed. Re-create it with:\n"
        f"    (cd {repo} && git diff) > {patch}"
    )


def main() -> None:
    BASELINES_DIR.mkdir(exist_ok=True)
    for name, url in REPOS.items():
        clone_repo(name, url)

    print()
    for name in REPOS:
        apply_patch(name)

    print()
    print("Baselines cloned and patched successfully.")
    print()
    print("Next steps:")
    print("  1. Install this project in dev mode:")
    print('     pip install -e ".[dev]"')
    print()
    print("  2. Run the demo:")
    print("     python demo.py")


if __name__ == "__main__":
    main()
