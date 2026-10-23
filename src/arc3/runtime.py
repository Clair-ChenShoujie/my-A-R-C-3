"""Install competition runtime and register the mounted source bundle."""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

def install(ctx):
    BUNDLE_DIR = ctx.BUNDLE_DIR
    WORKING_DIR = ctx.WORKING_DIR

    # Install the ARC runtime from the bundled competition wheels.
    # Quiet: stdout is discarded; stderr (and a non-zero exit) still surface real failures.
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-index",
            "--no-warn-conflicts",
            "--disable-pip-version-check",
            "--find-links",
            "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
            "arc-agi",
        ],
        stdout=subprocess.DEVNULL,
    )


    

    # Each bundled repo exposes its importable tree at <repo>/src or <repo>.
    def _source_path_entries(bundle_dir: Path) -> list:
        entries = []
        for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
            for candidate in (repo / "src", repo):
                if candidate.is_dir():
                    entries.append(candidate)
        return entries


    # Environment handed to each setup command (paths + any keys it has persisted).
    def _command_env() -> dict:
        env = os.environ.copy()
        # "$PYTHON" in a command resolves to this notebook's interpreter.
        env["PYTHON"] = sys.executable
        # Absolute path to the mounted source bundle.
        env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
        # The writable /kaggle/working directory.
        env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
        # A command writes a JSON object here to persist env keys to later commands + the run.
        return env


    # Make the bundled repos importable here (sys.path) and in child processes (.pth).
    source_entries = _source_path_entries(BUNDLE_DIR)
    for entry in source_entries:
        sys.path.insert(0, str(entry))
    pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
    pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
    print(f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)")
