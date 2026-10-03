"""Daily pipeline: refresh data, rerun every model walk-forward, stack them, write web/data/picks.json.

    python backend/pipeline.py              # full run (fetch + models + ensemble + picks)
    python backend/pipeline.py --no-fetch   # reuse research/cache as is
    python backend/pipeline.py --quick      # fetch only the current season where a cache exists

Each model family lives in research/<family>/run.py and writes research/preds/<model>.csv
(walk-forward, out-of-sample predictions for every game since 2006 plus the upcoming slate).
"""
from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "research"

# Order matters: efficiency writes the shared team feature table other families may read.
FAMILIES = ["efficiency", "ratings", "personnel", "situational", "roster", "closegames", "meta"]


def run(cmd: list[str]) -> None:
    t = time.time()
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    print(f"  done in {time.time() - t:.0f}s (peak memory so far {peak:.1f} GB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not a.no_fetch:
        run([sys.executable, "backend/fetch_cache.py"] + (["--quick"] if a.quick else []))
    for fam in FAMILIES:
        script = RESEARCH / fam / "run.py"
        if script.exists():
            run([sys.executable, str(script.relative_to(ROOT))])
    run([sys.executable, "backend/build_picks.py"])


if __name__ == "__main__":
    main()
