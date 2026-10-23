#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Run from the Marin checkout. Each condition owns its Iris completion callback.
set -euo pipefail
client_python=$1
client_script=$2
log_dir=$3
mkdir -p "$log_dir"
: "${WANDB_API_KEY:?Online W&B credentials must be available on the submit host}"
iris=("$client_python" "$client_script" --cluster marin)

for bias in 0 2 4; do
    job="boundary-stack-bias${bias}-20261023-v2"
    "${iris[@]}" job run --no-wait --job-name "$job" \
        --tpu v4-32 --enable-extra-resources --cpu 4 --memory 48GB --disk 16GB \
        --region us-central2 --priority batch --sync-package marin-levanter --extra tpu \
        -e PYTHONPATH .:lib/marin/src:lib/iris/src:lib/fray/src:lib/rigging/src:lib/haliax/src:lib/levanter/src \
        -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX 32843 \
        -- python -m experiments.grug.moe_cplm.train_belay \
        --tokenizer tokenizer --output /tmp/boundary-bias-results \
        --checkpoint-path gs://marin-us-central2/kaiyuew/stack-kda-depth4-20261023/bias-checkpoints \
        --artifact-root gs://marin-us-central2/kaiyuew/stack-kda-depth4-20261023/bias-results \
        --run-name "boundary-stack-b${bias}-d512-s0-v2" \
        --wandb-entity marin-community --wandb-project auto-understanding-arch \
        --variant plus-boundary-stack-kda --width 512 --steps 10676 \
        --stop-after-steps 1000 --tpu-chips 16 --boundary-stay-bias "$bias" \
        > "$log_dir/bias${bias}-submit.log" 2>&1
done

failed=0
for bias in 0 2 4; do
    job="boundary-stack-bias${bias}-20261023-v2"
    "${iris[@]}" job wait "/kaiyuew@stanford.edu/$job" > "$log_dir/bias${bias}-wait.log" 2>&1 || failed=1
done
exit "$failed"
