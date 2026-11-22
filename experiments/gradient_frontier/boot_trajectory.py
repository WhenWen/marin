import os,sys,subprocess
from pathlib import Path
subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
p=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
subprocess.run([p,'-u','-m','experiments.gradient_frontier.trajectory','logging'],check=True)
if sys.argv[1:] == ['verify']:
    subprocess.run([p,'-u','-m','experiments.gradient_frontier.trajectory','verify'],check=True)
    os.execv(p,[p,'-u','-m','experiments.gradient_frontier.replay_comparison'])
os.execv(p,[p,'-u','-m','experiments.gradient_frontier.trajectory',*sys.argv[1:]])
