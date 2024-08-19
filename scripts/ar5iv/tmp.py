import fsspec
import zipfile

fs = fsspec.filesystem("gcs")

with fsspec.open("zip://no-problem/2008/2008.09020.html::gcs://marin-data/raw/arxiv/data.fau.de/ar5iv-04-2024-no-problem.zip", "rb") as f:
    print(f.read()[:1000])