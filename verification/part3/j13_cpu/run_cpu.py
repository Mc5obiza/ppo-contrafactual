"""J13 extra: the single-classifier transfer cell on CPU (Table 3's source job 17699844 had no GPU in its allocation)."""
import os, sys, runpy
os.environ['PPO_DOMAIN'] = 'rsna_pneumonia'
HEAD = os.path.expanduser('~/verification/code/HEAD'); sys.path.insert(0, HEAD); os.chdir(HEAD)
import config; config.DEVICE = 'cpu'
import torch; print('torch threads', torch.get_num_threads(), 'cuda_available', torch.cuda.is_available())
sys.argv = ['arm_transfer.py', os.environ['PPO_CF_REPO'] + '/checkpoints/eval_shards',
            '--bootstrap', '--json', os.path.expanduser('~/verification/part3/j13_cpu/single_cpu.json')]
runpy.run_path(HEAD + '/arm_transfer.py', run_name='__main__')
