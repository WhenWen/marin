import os,sys,subprocess
from pathlib import Path
subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
p=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
os.execv(p,[p,'-u','-m','experiments.gradient_frontier.oracle_vjp',*sys.argv[1:]])
