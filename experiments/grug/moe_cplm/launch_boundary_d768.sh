#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Full d768 bias-4 schedule, matching the existing native Plus reference.
set -euo pipefail
client_python=$1
client_script=$2
log_dir=$3
mkdir -p "$log_dir"
exec 9> "$log_dir/submit.lock"
flock -n 9
test ! -e "$log_dir/submission-started"
: "${WANDB_API_KEY:?Online W&B credentials must be available on the submit host}"
git diff --exit-code
git diff --cached --exit-code
git rev-parse HEAD > "$log_dir/source-commit.txt"
iris=("$client_python" "$client_script" --cluster marin)
root=gs://marin-us-central2/kaiyuew/stack-kda-d768-20261023
args=(
    --tokenizer tokenizer --output /tmp/boundary-stack-d768
    --checkpoint-path "$root/checkpoints" --artifact-root "$root/results"
    --run-name boundary-stack-b4-d768-s0-v3
    --wandb-entity marin-community --wandb-project auto-understanding-arch
    --variant plus-boundary-stack-kda --width 768 --steps 29343
    --batch-size 16 --tpu-chips 16 --boundary-stay-bias 4
)
JAX_PLATFORMS=cpu .venv/bin/python -m experiments.grug.moe_cplm.train_belay \
    "${args[@]}" --logging-check > "$log_dir/logging-check.log" 2>&1
date -u +%FT%TZ > "$log_dir/submission-started"
"${iris[@]}" job run --no-wait --job-name boundary-stack-b4-d768-20261023-v3 \
    --tpu v4-32 --enable-extra-resources --cpu 4 --memory 48GB --disk 16GB \
    --region us-central2 --priority batch --sync-package marin-levanter --extra tpu \
    -e PYTHONPATH .:lib/marin/src:lib/iris/src:lib/fray/src:lib/rigging/src:lib/haliax/src:lib/levanter/src \
    -e WANDB_API_KEY "$WANDB_API_KEY" -e IRIS_PORT_JAX 32843 \
    -- python -m experiments.grug.moe_cplm.train_belay "${args[@]}" \
    > "$log_dir/submit.log" 2>&1
