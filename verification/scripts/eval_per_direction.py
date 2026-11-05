"""
eval_main.py (code 49e6050) with each direction generating with its own
adapter (per_direction.py). Same CLI as eval_main.py.

    ADAPTER_NP_VIEW=… ADAPTER_PN_VIEW=… python eval_per_direction.py --stage generate …
"""
import sys

import per_direction
import eval_main

per_direction.install_eval_main(eval_main)
eval_main.main(sys.argv[1:])
