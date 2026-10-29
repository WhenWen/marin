# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import gzip
import json
from collections import defaultdict

import fsspec
import ijson

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
a = parser.parse_args()
processes = {}
threads = {}
stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, {}]))
with fsspec.open(a.source, "rb") as raw, gzip.GzipFile(fileobj=raw) as stream:
    for e in ijson.items(stream, "traceEvents.item", use_float=True):
        pid = e.get("pid", 0)
        tid = e.get("tid", 0)
        name = e.get("name", "")
        if e.get("ph") == "M":
            if name == "process_name":
                processes[pid] = e.get("args", {})
            if name == "thread_name":
                threads[(pid, tid)] = e.get("args", {})
        if e.get("ph") == "X" and "dur" in e:
            row = stats[(pid, tid)][name]
            d = float(e["dur"])
            row[0] += d
            row[1] += 1
            row[2] = max(row[2], d)
            if not row[3]:
                row[3] = e.get("args", {})
result = []
for (pid, tid), s in stats.items():
    ops = sorted(s.items(), key=lambda x: -x[1][0])[:30]
    result.append(
        {
            "pid": pid,
            "tid": tid,
            "process": processes.get(pid),
            "thread": threads.get((pid, tid)),
            "total_duration_us": sum(r[0] for r in s.values()),
            "top_operations": [
                {"name": n, "total_us": r[0], "count": r[1], "max_us": r[2], "args": r[3]} for n, r in ops
            ],
        }
    )
result.sort(key=lambda x: -x["total_duration_us"])
with fsspec.open(a.output, "wt") as out:
    json.dump(result, out, default=str)
for row in result[:20]:
    print(json.dumps({k: v for k, v in row.items() if k != "top_operations"}))
    print(json.dumps(row["top_operations"][:3], default=str)[:2400])
