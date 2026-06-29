---
name: launch-discipline
description: Discipline for launching experiment sweeps on Iris and babysitting them safely — avoid wedging the shared controller, never lose job ids, babysit to wandb. Use when launching grug/moe or speedrun sweeps, submitting many Iris jobs, or babysitting a launched run.
---

# Skill: Experiment Launch & Babysit Discipline

Hard-learned rules for launching (especially **sweeps**) on the shared Iris cluster and babysitting
them to a real result. The failure that motivated this: a ~26-job launch+mass-stop churn **wedged the
shared marin controller** (OOM / bloated DB → all RPCs hang for hours), blocking everyone. Most rules
here exist to prevent that and to not waste a launch.

For the mechanical monitor-and-recover loop of a *single* job, use **babysit-job**. This skill is the
*judgment* layer around launching and sweeping.

## 0. The shared-controller rule (most important)
- The Iris controller is **shared infrastructure** serving other people's jobs. **Never restart it,
  roll back its DB, or otherwise "fix" it to clear your own backlog** — that's the cluster owner/admin's
  call. Only ever stop/manage **your own jobs, by job id.**
- Corollary: don't create the mess in the first place (see §1).

## 1. Before launching a sweep — cap the blast radius
- **Launch a small batch first** (1–4 jobs), confirm they compile + step + log to wandb, *then* expand.
  Do **not** fire 10–20+ jobs at once. The controller backlog from a big launch (and the subsequent
  mass-stop) is what OOMs/wedges it.
- Decide the metric and the comparison up front: which wandb metric, at the **final** step (e.g.
  `eval/paloma/c4_en/bpb`), vs which baseline.
- Confirm region/data co-location: TPU region, data, validation set, `MARIN_PREFIX` all in the same
  region (e.g. us-east5). Pin it.
- Keep `--preemptible` unless the user explicitly authorizes otherwise.

## 2. Launching — never lose the job id
- `iris job run ... -- python -m <launcher>` submits a **CPU launcher** job that submits the **TPU
  training** job. wandb runs are created by the *training* (child) job, not the launcher.
- **ALWAYS tee the launch output to a file**: `... 2>&1 | tee /tmp/launch_<tag>.log`. Piping
  `iris job run` straight into `grep "submitted"` inside a background task **loses the job id to shell
  buffering** — you end up with a running job you can't address. Read the id back from the file.
- Each `iris` CLI call **re-establishes the SSH/IAP tunnel (~84 s)** before talking to the controller.
  Use timeouts **≥180 s** on every call. Short timeouts (60–90 s) die *mid-tunnel* and masquerade as
  controller failures. For many back-to-back ops, hold one tunnel and use `--controller-url`.

## 3. Babysitting — to wandb, not to "submitted"
- "Job submitted" is **not** done. Babysit until **`train/loss` is actually logging in wandb** for the
  child run. (A monitor that greps `iris job logs` for the tqdm `postfix:loss` proves *training* is
  happening, but the user watches **wandb** — confirm both.)
- Read the **CHILD** training task for progress/errors, not the parent launcher log. Cold starts can be
  long (tens of minutes on some regions); don't call a run hung prematurely.
- Poll cadence: ~2 min during deps→compile→step-1; back off once it's stepping.
- Distinguish real crashes (SIGSEGV / exit code / NaN / `No module`) from benign log noise before
  declaring failure.

## 4. Reading results
- Compare runs **only at the same FINAL step**, with `state==finished`. Never quote a mid-run number
  against finished runs. "Matches" means within the agreed tolerance (e.g. ~2e-3 for loss), not 4e-2.
- **Don't conclude under contention.** If preemptible jobs are starved/cycling, that's not a result —
  wait for compute and let all sweep points finish.

## 5. Relaunching after a code fix
- **Bump the run tag.** Deterministic run-ids reuse the same wandb run (collision) and resume a stale
  checkpoint. A fresh tag = a clean run.
- Commit the fix first (edit→lint→commit), so the bundled working tree ships the change.

## 6. When the controller wedges (symptoms + response)
- **Symptoms:** `cluster status`, `job list`, AND `job stop` all hang *past long timeouts* (e.g. 200–
  1800 s). Restart hangs on checkpoint. This is the documented bloated-DB/OOM mode — the wedge is
  **below the RPC layer**.
- **What does NOT work:** submitting stops to "drain the backlog" (stop RPCs hang too; there is no bulk
  `--all`), client-side waiting, or a plain restart (reloads the same bloated DB and re-wedges).
- **What to do:** stop poking it (each attempt just burns an 84 s tunnel). Set up a **periodic poll**
  (~every 15 min) that detects recovery and pings you; **escalate to the cluster owner/admin** for the
  VM-level fix (clean restart not reusing the bloated DB, or a DB checkpoint rollback — see
  `lib/iris/OPS.md`). Recovery of shared infra is **their** call, not yours.

## 7. Auth sanity (new machine)
Iris auth here is GCP-identity-based: `gcloud auth login` + `gcloud auth application-default login`
(writes the `authorized_user` ADC) + `gcloud config set project <proj>`; your user needs
`serviceAccountTokenCreator` on the controller SA + IAP access. Then `iris --config <cfg> cluster
status` should tunnel and return. No separate `iris login`/JWT in current marin iris.
