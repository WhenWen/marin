"""Run online logging and interrupted continuation checks before the real crossover."""
import os
import subprocess
import sys
from pathlib import Path

subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
exe=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
cmd=[exe,'-u','-m','experiments.gradient_frontier.fifty_crossover']
for mode in ('logging','check','verify'):
    subprocess.run([*cmd,mode,*sys.argv[1:]],check=True)
os.execv(exe,[*cmd,'main',*sys.argv[1:]])
