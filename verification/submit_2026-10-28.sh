#!/bin/bash
# Approved 2026-10-28: groups A, B, C and J9 (seed 1, ens, gpuh100&gpu80g).
# Per-direction adapters for J4, J5, J5b, J6, J8 (n->p committed, p->n pn_lora).
set -euo pipefail
cd ~/verification/scripts
L=~/verification/JOBS_SUBMITTED.txt
E="ADAPTER_EXPECT_NP_SHA=658cdd7956a858c16a6b13c136ec01e4a61668b738826c74b51cdbf5ddc86f99,ADAPTER_EXPECT_PN_SHA=6ae442301a803162fd66d020521c36d8aa800138acb209e27f20ec59766c9651"
s () { local name=$1; shift; local id; id=$(sbatch --parsable "$@"); echo "$name $id $(date -Iseconds) sbatch $*" | tee -a "$L"; echo "$id"; }
echo "# submitted $(date -Iseconds)" >> "$L"
# --- B: J4 first
J4=$(s J4_gen    --export=ALL,VIEW=committed,PN_VIEW=pn_lora,ARM=trained,TAG=J4_perdir,$E 04_eval_generate.sbatch | tail -1)
s J4_reduce      --dependency=afterok:$J4 --export=ALL,VIEW=committed,PN_VIEW=pn_lora,ARM=trained,TAG=J4_perdir,$E 05_eval_reduce.sbatch >/dev/null
# --- A: J1, J2 on existing shards now; re-runs on the J4 shards after J4 generate
s J1             01_paired_lhr.sbatch >/dev/null
s J2             02_transfer_bootstrap.sbatch >/dev/null
s J1_on_J4       --dependency=afterok:$J4 --export=ALL,S2=$HOME/verification/out/J4_perdir/shards,OUTNAME=paired_lhr_J4 01_paired_lhr.sbatch >/dev/null
s J2_on_J4       --dependency=afterok:$J4 --export=ALL,ENS2=$HOME/verification/out/J4_perdir/shards,ENS2_TAG=ens2_J4_perdir,ONLY_ENS2=1,OUTNAME=transfer_J4 02_transfer_bootstrap.sbatch >/dev/null
s J3             03_gradcam_diag.sbatch >/dev/null
# --- B: J5
J5=$(s J5_gen    --export=ALL,VIEW=committed,PN_VIEW=pn_lora,ARM=none,TAG=J5_none_perdir,$E 04_eval_generate.sbatch | tail -1)
s J5_reduce      --dependency=afterok:$J5 --export=ALL,VIEW=committed,PN_VIEW=pn_lora,ARM=none,TAG=J5_none_perdir,$E 05_eval_reduce.sbatch >/dev/null
# --- C: J6 -> J5b, J7, J8
J6=$(s J6        06_action_stats.sbatch | tail -1)
J5B=$(s J5b_gen  --dependency=afterok:$J6 10_fixed_schedule_generate.sbatch | tail -1)
s J5b_reduce     --dependency=afterok:$J5B 11_fixed_schedule_reduce.sbatch >/dev/null
s J7             07_runtime_memory.sbatch >/dev/null
s J8             08_mask_ablation.sbatch >/dev/null
# --- J9: retrain seed 1 (both tasks gpuh100&gpu80g), then per-direction eval with its own adapters
J9=$(s J9_train  --export=ALL,SEED=1,GUIDANCE=ens 09_retrain.sbatch | tail -1)
J9E=$(s J9_eval_gen --dependency=afterok:$J9 --export=ALL,VIEW=J9_seed1_np,PN_VIEW=J9_seed1_pn,ARM=trained,TAG=J9_seed1 04_eval_generate.sbatch | tail -1)
s J9_eval_reduce --dependency=afterok:$J9E --export=ALL,VIEW=J9_seed1_np,PN_VIEW=J9_seed1_pn,ARM=trained,TAG=J9_seed1 05_eval_reduce.sbatch >/dev/null
