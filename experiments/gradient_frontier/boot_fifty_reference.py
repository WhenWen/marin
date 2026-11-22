"""Extend validated native optimizer trajectories to fifty updates without modifying the original schedule."""
import os
import subprocess
import sys
from pathlib import Path
subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
exe=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
mod='experiments.gradient_frontier.'
method=sys.argv[1]
subprocess.run([exe,'-u','-m',mod+'trajectory','logging',method],check=True)
os.execv(exe,[exe,'-u','-m',mod+'trajectory','main50',method])
