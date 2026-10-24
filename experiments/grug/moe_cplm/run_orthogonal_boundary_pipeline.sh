#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Continue the already submitted qualification, then train only if it passes.
set -euo pipefail
client_python=$1
client_script=$2
log_dir=$3
source_commit=$4
mkdir -p "$log_dir"
exec 9> "$log_dir/pipeline.lock"
flock -n 9
iris=("$client_python" "$client_script" --cluster marin)
kernel_job=/kaiyuew@stanford.edu/orthogonal-boundary-kernel-20261023-v1
train_job=/kaiyuew@stanford.edu/orthogonal-boundary-lm-20261023-v1

record_phase() {
    printf '{"phase":"%s","source_commit":"%s","updated_utc":"%s"}\n' \
        "$1" "$source_commit" "$(date -u +%FT%TZ)" > "$log_dir/pipeline-state.json"
}

trap 'record_phase failed' ERR
record_phase waiting_for_qualification
"${iris[@]}" job wait "$kernel_job" > "$log_dir/kernel-pipeline-wait.log" 2>&1
record_phase qualification_passed
[[ "$(git rev-parse HEAD)" == "$source_commit" ]]
git diff --exit-code
git diff --cached --exit-code
"${iris[@]}" job logs "$kernel_job" --tail --max-lines 40 > "$log_dir/kernel-final.log" 2>&1
bash experiments/grug/moe_cplm/launch_orthogonal_boundary.sh \
    "$client_python" "$client_script" train "$log_dir"
record_phase training_submitted
"${iris[@]}" job wait "$train_job" > "$log_dir/train-wait.log" 2>&1
record_phase training_finished
