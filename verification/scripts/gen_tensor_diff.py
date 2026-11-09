"""
CPU-only, read-only: compare the n_to_p and p_to_n generator state dicts of one
arm tensor by tensor (a byte-level sha256 mismatch can in principle come from
serialisation metadata alone; this says whether the WEIGHTS differ).

    python gen_tensor_diff.py <arm_dir> [<arm_dir> ...]
"""
import json
import os
import sys

import torch


def compare(a_path, b_path):
    a = torch.load(a_path, map_location="cpu", weights_only=True, mmap=True)
    b = torch.load(b_path, map_location="cpu", weights_only=True, mmap=True)
    out = {"keys_only_in_np": sorted(set(a) - set(b)),
           "keys_only_in_pn": sorted(set(b) - set(a)),
           "n_keys": len(a), "n_differing": 0, "n_differing_lora": 0,
           "max_abs_diff": 0.0, "max_abs_diff_lora": 0.0,
           "max_abs_diff_nonlora": 0.0, "examples": []}
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape:
            out["examples"].append([k, "shape", list(ta.shape), list(tb.shape)])
            out["n_differing"] += 1
            continue
        if torch.equal(ta, tb):
            continue
        d = float((ta.float() - tb.float()).abs().max())
        lora = "lora" in k.lower()
        out["n_differing"] += 1
        out["n_differing_lora"] += int(lora)
        out["max_abs_diff"] = max(out["max_abs_diff"], d)
        key = "max_abs_diff_lora" if lora else "max_abs_diff_nonlora"
        out[key] = max(out[key], d)
        if len(out["examples"]) < 5:
            out["examples"].append([k, d])
    return out


def main(dirs):
    res = {}
    for d in dirs:
        r = {}
        for name in ("unet_finetuned.pt", "cond_proj.pt"):
            r[name] = compare(os.path.join(d, "n_to_p", name),
                              os.path.join(d, "p_to_n", name))
        res[d] = r
        print(d, json.dumps({n: {k: v[k] for k in ("n_keys", "n_differing", "n_differing_lora",
                                                   "max_abs_diff_lora", "max_abs_diff_nonlora")}
                             for n, v in r.items()}), flush=True)
    return res


if __name__ == "__main__":
    with open(os.path.expanduser("~/verification/generator_tensor_diff.json"), "w") as f:
        json.dump(main(sys.argv[1:]), f, indent=1)
