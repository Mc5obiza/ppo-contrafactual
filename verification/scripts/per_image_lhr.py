"""
Dump per-image LHR (hit, null, lift) for every arm's shards to one JSON, so
paired tests and bootstraps can be recomputed later without re-reading shards.

CPU only, no model is loaded: lift is pure arithmetic on |x_cf - x| and the
rasterised RSNA boxes. Reuses paired_arm_test.py's loader and per-image lift so
the numbers are by construction the same computation as the reported ones.

Run from ~/verification/code/49e6050 (see 01_paired_lhr.sbatch).
"""
import json
import sys

import config
import ingestion
import paired_arm_test as pat


def main(out_path: str, shard_dirs: list[str]) -> None:
    dataset = ingestion.build_dataset(img_size=config.IMG_SIZE)
    out = {}
    for d in shard_dirs:
        xp, xc, idx = pat._load_arm(d)
        lifts = pat._per_image_lift(xp, xc, idx, dataset)
        out[d] = {str(k): float(v) for k, v in lifts.items()}
        print(f"{d}: {len(lifts)} annotated images, mean lift "
              f"{sum(lifts.values()) / max(len(lifts), 1):.4f}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
