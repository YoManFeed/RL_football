from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_rl.training.curriculum import CurriculumRunner, make_default_curriculum


def main() -> None:
    runner = CurriculumRunner(make_default_curriculum(), workdir="./artifacts")
    checkpoints = runner.run_stub(seed=5, steps_per_stage=48)
    for ckpt in checkpoints:
        print("saved checkpoint:", ckpt)


if __name__ == "__main__":
    main()
