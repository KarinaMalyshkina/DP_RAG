"""Clone the original DP-RAG baseline repositories into baselines/ and apply our patches.

The baselines/ directory is gitignored (it holds clones of external repos), so our
local edits to those clones are not tracked by this project's git. To keep them from
being lost on a fresh clone, the edits are stored as patch files under patches/ and
re-applied here.

Two things this script is strict about, because both fail silently otherwise:

* **Upstream commits are pinned.** Cloning whatever ``main`` points at today means a
  rerun months from now gets different baseline code — and the patches, which are
  diffs against a specific tree, stop applying. Neither the privacy mechanisms nor
  the results would be reproducible.
* **A patch that does not apply is a fatal error.** The patches are not cosmetic:
  they disable the reasoning-mode block that would consume the whole DP token
  budget, chunk the corpus encoder so a duplicated corpus does not exhaust memory,
  and make the device selection work off-GPU. Continuing without them produces
  plausible numbers computed by the wrong code.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINES_DIR = ROOT / "baselines"
PATCHES_DIR = ROOT / "patches"

# Pinned upstream commits. Update deliberately, and regenerate the matching patch
# with `(cd baselines/<name> && git diff) > patches/<name>.patch` when you do.
REPOS = {
    "koga": (
        "https://github.com/tacchan7412/DPRAG.git",
        "28c79e5c6c0b01f9fb5f4ad1091824ea316bf315",
    ),
    "grislain": (
        "https://github.com/sarus-tech/dp-rag.git",
        "b9d1dea2c7799b9b432ab871b8982f0298533ea1",
    ),
}


def run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def clone_repo(name: str, url: str, commit: str) -> None:
    """Clone if absent, then make sure the checkout sits on the pinned commit."""
    dest = BASELINES_DIR / name
    if not dest.exists():
        print(f"[clone] {name}: {url}")
        result = run(["git", "clone", url, str(dest)])
        if result.returncode != 0:
            fail(f"could not clone {url}\n{result.stderr}")

    head = run(["git", "rev-parse", "HEAD"], cwd=dest).stdout.strip()
    if head == commit:
        print(f"[pin]   {name}: already at {commit[:12]}")
        return

    # Refuse to move a checkout that has uncommitted work in it — that work is
    # almost certainly a patch someone is in the middle of authoring.
    dirty = run(["git", "status", "--porcelain"], cwd=dest).stdout.strip()
    if dirty:
        fail(
            f"{name} is at {head[:12]} but pinned to {commit[:12]}, and has local "
            f"changes that checking out would disturb.\n"
            f"Save them first:  (cd {dest} && git diff) > {PATCHES_DIR / f'{name}.patch'}\n"
            f"then re-run this script."
        )

    print(f"[pin]   {name}: {head[:12]} -> {commit[:12]}")
    if run(["git", "fetch", "--all"], cwd=dest).returncode != 0:
        print(f"[pin]   {name}: fetch failed, trying the local objects anyway")
    result = run(["git", "checkout", "--detach", commit], cwd=dest)
    if result.returncode != 0:
        fail(f"could not check out pinned commit {commit} in {name}\n{result.stderr}")


def apply_patch(name: str) -> None:
    """Apply patches/{name}.patch inside baselines/{name}, idempotently and strictly."""
    patch = PATCHES_DIR / f"{name}.patch"
    repo = BASELINES_DIR / name
    if not patch.exists():
        print(f"[patch] {name}: no patch file at {patch}, nothing to apply")
        return
    if not repo.exists():
        fail(f"{name}: clone missing at {repo}")

    def git_apply(*flags: str) -> bool:
        return run(["git", "apply", *flags, str(patch)], cwd=repo).returncode == 0

    if git_apply("--reverse", "--check"):
        print(f"[patch] {name}: already applied")
        return
    if git_apply("--check"):
        result = run(["git", "apply", str(patch)], cwd=repo)
        if result.returncode != 0:
            fail(f"{name}: patch passed --check but failed to apply\n{result.stderr}")
        print(f"[patch] {name}: applied {patch.name}")
        return

    fail(
        f"{name}: {patch.name} does not apply to the pinned checkout.\n"
        f"The framework depends on these edits — running without them silently "
        f"changes what the pipelines compute.\n"
        f"Either the pin in REPOS was changed without regenerating the patch, or the "
        f"clone is in an unexpected state. To rebuild from scratch:\n"
        f"    rm -rf {repo} && python setup_baselines.py"
    )


def main() -> None:
    BASELINES_DIR.mkdir(exist_ok=True)
    for name, (url, commit) in REPOS.items():
        clone_repo(name, url, commit)

    print()
    for name in REPOS:
        apply_patch(name)

    print("\nBaselines cloned at their pinned commits and patched.")
    print("\nNext: uv sync --extra dev   (or see docs/RUNBOOK_GPU.md)")


if __name__ == "__main__":
    main()
