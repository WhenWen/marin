"""Replay independently once the uninterrupted reference and durable checkpoint exist."""
import os,sys,subprocess
from pathlib import Path
os.environ.setdefault('WANDB_CONSOLE','off')
os.environ.setdefault('WANDB_SILENT','true')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR','/tmp/gradient-frontier-crossover-jit')
subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
exe=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
os.execv(exe,[exe,'-u','-m','experiments.gradient_frontier.fifty_crossover','verify',*sys.argv[1:]])
