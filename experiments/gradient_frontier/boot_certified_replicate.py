import os
import subprocess
import sys
from pathlib import Path

subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
os.environ.setdefault('WANDB_CONSOLE','off')
os.environ.setdefault('WANDB_SILENT','true')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR','/tmp/gradient-frontier-crossover-jit')
exe=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
cmd=[exe,'-u','-m','experiments.gradient_frontier.fifty_crossover']
subprocess.run([*cmd,'logging',*sys.argv[1:]],check=True)
subprocess.run([exe,'-u','-m','experiments.gradient_frontier.certified_replicate',*sys.argv[1:]],check=True)
os.execv(exe,[*cmd,'main',*sys.argv[1:]])
