#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Continue all three qualified prefixes concurrently with native full-state restore.
set -euo pipefail
client_python=$1
client_script=$2
log_dir=$3
mkdir -p "$log_dir"
: "${WANDB_API_KEY:?Online W&B credentials must be available on the submit host}"
iris=("$client_python" "$client_script" --cluster marin)
root=gs://marin-us-central2/kaiyuew/stack-kda-depth4-20261023

for condition in 0:6d4890e9ab5e4cfab29751310e8b4541 2:ba085c2bc208429491147cb5ab969bef 4:55ac35a0d5b04e7883ebeb924f7f8b17; do
    bias=${condition%%:*}
    parent=${condition#*:}
    "${iris[@]}" job run --no-wait --job-name "boundary-stack-bias${bias}-full-20261023-v1" \
        --tpu v4-32 --enable-extra-resources --cpu 4 --memory 48GB --disk 16GB \
        --region us-central2 --priority batch --sync-package marin-levanter --extra tpu \
        -e PYTHONPATH .:lib/marin/src:lib/iris/src:lib/fray/src:lib/rigging/src:lib/haliax/src:lib/levanter/src \
        -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX 32843 \
        -- python -m experiments.grug.moe_cplm.train_belay \
        --tokenizer tokenizer --output /tmp/boundary-bias-full-results \
        --checkpoint-path "$root/bias-full-checkpoints" --artifact-root "$root/bias-full-results" \
        --resume-checkpoint "$root/bias-checkpoints/$parent/step-1000" --resume-step 1000 \
        --run-name "boundary-stack-b${bias}-full-d512-s0-v1" \
        --wandb-entity marin-community --wandb-project auto-understanding-arch \
        --variant plus-boundary-stack-kda --width 512 --steps 10676 \
        --tpu-chips 16 --boundary-stay-bias "$bias" \
        > "$log_dir/bias${bias}-submit.log" 2>&1
done
