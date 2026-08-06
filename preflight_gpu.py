"""
preflight_gpu.py
================
Fail fast if the venv's torch has no compiled kernels for the GPU this job
landed on. Run from run_ppo.slurm and run_eval_ppo.slurm before any real work.

WHY THIS EXISTS
---------------
Job 17662452_0 was scheduled onto hpc-n983 — an "NVIDIA RTX PRO 6000 Blackwell
Server Edition" (sm_120) — and died ~5 minutes in, inside peft's
get_peft_model(), with:

    RuntimeError: CUDA error: no kernel image is available for execution on
    the device

The venv is torch 2.5.1+cu121, whose kernels stop at sm_90. The traceback
points into peft/diffusers and reads like a software bug; it is really "wrong
node". Five minutes of dataset scanning and model loading happen before the
first CUDA op that trips it.

The SLURM --constraint in those scripts is the primary defence, but feature
strings do NOT encode compute capability — hpc-n983 advertises the same
`gpurtx6000` tag as the Turing Quadro RTX 6000 nodes it must be told apart
from. A node added to an existing feature tag would slip straight through, so
this check is the backstop that does not depend on the tags being accurate.

Exits 0 and prints the GPU + arch list on success, so every job log records
which hardware it actually ran on.
"""

import sys

import torch


def main() -> int:
    if not torch.cuda.is_available():
        print("[preflight] FATAL: torch.cuda.is_available() is False — this job "
              "has no usable GPU. Check --gres=gpu:1 and the cuda module load.",
              file=sys.stderr)
        return 1

    major, minor = torch.cuda.get_device_capability()
    sm        = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()

    print(f"[preflight] {torch.cuda.get_device_name(0)} -> {sm}")
    print(f"[preflight] torch {torch.__version__} kernels: {arch_list}")

    # Accept the whole major family, not an exact string match. CUDA cubins run
    # on any device of the SAME major version with an equal-or-higher minor, so
    # an sm_86 binary is valid on an sm_89 Ada card (hpc-n[967-968], L40S) even
    # though 'sm_89' never appears in arch_list. Requiring an exact match would
    # reject those nodes for no reason.
    supported = any(
        a.startswith(f"sm_{major}") and int(a.split("_")[1][1:]) <= minor
        for a in arch_list
    )
    if not supported:
        print(
            f"[preflight] FATAL: this torch has no kernels for {sm}.\n"
            f"[preflight] Every CUDA op would fail with 'no kernel image is "
            f"available for execution on the device', most likely several "
            f"minutes from now and with a traceback pointing somewhere "
            f"misleading.\n"
            f"[preflight] Fix: add this node to --exclude in the .slurm script, "
            f"or rebuild the venv with a torch built for {sm} "
            f"(Blackwell/sm_120 needs CUDA 12.8+).",
            file=sys.stderr,
        )
        return 1

    print("[preflight] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
