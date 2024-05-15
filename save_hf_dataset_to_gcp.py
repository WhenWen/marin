# How to run this file?
# python3 save_hf_dataset_to_gcp.py tiiuae/falcon-refinedweb


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
