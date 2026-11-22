"""Keep completed diagnostic observations reachable by attempt as well as by configuration."""
import json
from .run import ROOT


def publish(fs,path,payload):
    data=json.dumps(payload,allow_nan=False).encode()
    identifier=payload['wandb'].rstrip('/').rsplit('/',1)[-1]
    suffix=path[len(ROOT)+1:]
    fs.pipe(f'{ROOT}/diagnostic-attempts/{suffix.removesuffix(".json")}/{identifier}.json',data)
    fs.pipe(path,data)
