"""
This script downloads a HuggingFace dataset and uploads it to Google Cloud Storage.

It can be run locally, but for large datasets it is recommended to run it on a VM on GCP for faster transfer speeds.

How to run this file?
1. Create a VM on GCP
2. Connect to the VM
3. Install Git: `sudo apt-get install git`
4. Clone the repository: https://github.com/stanford-crfm/marin
5. Install conda:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
6. Create a new conda env (use python 3.9 because 3.10 and 3.11 lead to issues when installing the requirements and running the script):
```
conda create --name marin python=3.9
conda activate marin
```
7. Install your requirements:
```
cd marin
pip install -r requirements.txt
```
6. Start a tmux session:
```
sudo apt-get install tmux
tmux
```
7. Run the file and specify the dataset to be downloaded:
```python3 save_hf_dataset_to_gcp.py tiiuae/falcon-refinedweb```
"""

import argparse
import requests
import subprocess
from urllib.parse import urlparse
import os
import tqdm
from pprint import pprint

# Create the parser
parser = argparse.ArgumentParser(description='Save a HuggingFace dataset to Google Cloud Storage')

# Example datasets: tiiuae/falcon-refinedweb, cerebras/SlimPajama-627B
parser.add_argument('dataset_path', type=str, help='The path to the dataset')
parser.add_argument('--bucket_name', type=str, help='The name of the GCS bucket', default='marin-data')

if __name__ == '__main__':
    # Parse the arguments
    args = parser.parse_args()

    DATASET_URL = f"https://datasets-server.huggingface.co/parquet?dataset={args.dataset_path}"
    overview = requests.get(DATASET_URL).json()
    pprint(overview)

    for file in tqdm.tqdm(overview['parquet_files']):
        split = file['split']
        url = file['url']
        filename = os.path.basename(urlparse(url).path)
        output_path = f"gs://{args.bucket_name}/{args.dataset_path}/{split}/{filename}"
        # Download the file to memory and stream it to GCP directly
        command = f"wget -qO- {url} | gsutil -h 'Content-Type:application/parquet' cp - {output_path}"
        print(f"Running command {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stderr)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        else:
            print(f"File {filename} uploaded to {output_path}")
