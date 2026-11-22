"""Gate every scale trajectory on fresh-process deterministic checkpoint replay."""
import os
from pathlib import Path
import subprocess
import sys

subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
exe=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
width,depth=sys.argv[1:]
root='experiments.gradient_frontier.'
for mode in ('logging','check','verify'):
    subprocess.run([exe,'-u','-m',root+'trajectory',mode,'native',width,depth],check=True)
subprocess.run([exe,'-u','-m',root+'replay_comparison','native',width,depth],check=True)
os.execv(exe,[exe,'-u','-m',root+'trajectory','main','native',width,depth])
