import os,sys,subprocess
from pathlib import Path
subprocess.run(['uv','sync','--frozen','--package','marin-levanter','--extra','gpu','--no-dev'],check=True)
p=str(Path(os.environ.get('IRIS_VENV','.venv'))/'bin/python')
mode,method=sys.argv[1:3]
def execute(module,*args):
    subprocess.run([p,'-u','-m','experiments.gradient_frontier.'+module,*args],check=True)
execute('trajectory','logging',method)
if mode=='validated':
    execute('trajectory','check',method)
    execute('trajectory','verify',method)
    execute('replay_comparison',method)
    execute('trajectory','main',method)
else:
    execute('trajectory',mode,method)
    if mode=='verify':execute('replay_comparison',method)
