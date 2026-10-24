#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Run from the qualified Marin checkout on the remote submit host.
set -euo pipefail
client_python=$1
client_script=$2
mode=$3
log_dir=$4
mkdir -p "$log_dir"
iris=("$client_python" "$client_script" --cluster marin)
root=gs://marin-us-central2/kaiyuew/stack-kda-depth4-20261023

if [[ "$mode" == kernel ]]; then
    "${iris[@]}" job run --no-wait --job-name orthogonal-boundary-kernel-20261023-v1 \
        --tpu v4-8 --cpu 4 --memory 48GB --disk 16GB --region us-central2 \
        --priority batch --sync-package marin-levanter --extra tpu \
        -e PYTHONPATH .:lib/marin/src:lib/iris/src:lib/fray/src:lib/rigging/src:lib/haliax/src:lib/levanter/src \
        -e XLA_FLAGS --xla_allow_excess_precision=false \
        -- python -m experiments.grug.moe_cplm.orthogonal_boundary_validation \
        --artifact-root "$root/orthogonal-kernel-results" > "$log_dir/kernel-submit.log" 2>&1
elif [[ "$mode" == train ]]; then
    : "${WANDB_API_KEY:?Online W&B credentials must be available on the submit host}"
    "${iris[@]}" job run --no-wait --job-name orthogonal-boundary-lm-20261023-v1 \
        --tpu v4-32 --enable-extra-resources --cpu 4 --memory 48GB --disk 16GB \
        --region us-central2 --priority batch --sync-package marin-levanter --extra tpu \
        -e PYTHONPATH .:lib/marin/src:lib/iris/src:lib/fray/src:lib/rigging/src:lib/haliax/src:lib/levanter/src \
        -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX 32843 \
        -- python -m experiments.grug.moe_cplm.train_belay \
        --tokenizer tokenizer --output /tmp/orthogonal-boundary-results \
        --checkpoint-path "$root/orthogonal-checkpoints" \
        --artifact-root "$root/orthogonal-results" \
        --run-name orthogonal-boundary-d512-s0-v1 \
        --wandb-entity marin-community --wandb-project auto-understanding-arch \
        --variant plus-boundary-orthogonal-kda --width 512 --steps 10676 --tpu-chips 16 \
        > "$log_dir/train-submit.log" 2>&1
else
    echo 'Mode must be kernel or train' >&2
    exit 2
fi
