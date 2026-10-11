"""Portable driver for the exact archived training and evaluation scripts.

Keeps the original executed scripts byte-for-byte unchanged, including their
embedded source hashes. Overrides only research-data directory paths at runtime.
Use --stage audit to independently verify the evidence without downloading data.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = {
    "download": "download_observations.py",
    "prepare": "prepare_observations.py",
    "encoder": "train_encoder.py",
    "diffusion": "run_diffusion.py",
    "report": "make_report.py",
    "readme": "write_readme.py",
    "audit": "audit_evidence.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=STAGES,
        help="Individual reproducible experiment stage",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("CRYOCAST_X0_EVIDENCE_DIR", HERE)),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[477, 478, 479]
    )
    parser.add_argument(
        "--timesteps", nargs="+", type=int, default=[16, 32, 64]
    )
    args = parser.parse_args()
    base = args.data_dir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    source = HERE / STAGES[args.stage]
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"x0evidence_{args.stage}", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # I/O location changes only, not training science. Source script hash stays
    # the same as the exact script used for the archived results.
    if hasattr(module, "BASE"):
        module.BASE = base
    if hasattr(module, "DEST"):
        module.DEST = base
    if hasattr(module, "EVIDENCE"):
        module.EVIDENCE = HERE / "reference_previous_pilot"

    if args.stage == "audit":
        module.audit(base)
        return
    if args.stage == "diffusion":
        sys.argv = [str(source), "--seeds", *map(str, args.seeds),
                    "--timesteps", *map(str, args.timesteps)]
    elif args.stage == "encoder":
        sys.argv = [
            str(source), "--max-steps", "20000", "--check-every", "200",
            "--min-steps", "3000", "--patience", "15",
        ]
    else:
        sys.argv = [str(source)]
    module.main()


if __name__ == "__main__":
    main()
