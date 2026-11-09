"""J12 coarse-grid RE-SCORE on A100 (2026-11-03, user option "A100 + re-score coarse"). The local grid moved from L40S
to A100 for queue reasons; GPU type alone shifts per-image counterfactuals and hence returns, so the final pick must
compare configs scored on ONE GPU type. This re-runs the coarse grid's stage-2 configs (its top 6 per direction, read
from out/J12_grid/select/stage2_*.json) on the same 48 training-fold images and seeds, with the same code, env, adapters
and criterion as j12_grid_select.py (stage 1 = images 0-11, stage 2 = images 12-47, all 6 kept: TOPK = 6).
tau stays at the coarse value (0.5); j12_select.py restores the stop bonus exactly as for the L40S coarse rows."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j12_grid_select as G
SRC = os.path.expanduser('~/verification/out/J12_grid/select')

def configs(direction):
    out = []
    for f in sorted(glob.glob(f'{SRC}/stage2_{direction}_*.json')):
        r = json.load(open(f)); out.append(dict(name=r['name'], eligible=True, controls=r['controls'], raw=r['raw']))
    assert len(out) == 6, (direction, len(out))
    return out

if __name__ == '__main__':
    G.configs = configs; G.TOPK = 6
    G.main()
