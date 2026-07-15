#!/bin/bash
# Automated pipeline: wait for experiments -> copy data -> analyze -> git push
# Runs in background on the lab. Logs to /data/lab/results/v2/pipeline.out
set -u
export GIT_TERMINAL_PROMPT=0

LOG=/data/lab/results/v2/pipeline.out
MARKER=/data/lab/results/v2/PIPELINE_DONE.marker
REPO=/data/lab/gsaca
SRC=/data/lab/results/v2

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "===== PIPELINE START ====="

# ---------- 1. Wait for all experiment processes ----------
log "waiting for run_experiment_local.py processes to finish..."
while true; do
  N=$(ps aux | grep "run_experiment_local.py" | grep -v grep | wc -l | tr -d ' ')
  if [ "$N" -eq 0 ]; then break; fi
  log "still running: $N process(es)"
  sleep 120
done
log "ALL EXPERIMENT PROCESSES FINISHED"

# ---------- 2. Sanity check ----------
BOT=$(find "$SRC/exp_b_20seed/battle_of_the_sexes" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
PG=$(find "$SRC/exp_b_20seed/public_goods" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')
log "metrics count: battle_of_the_sexes=$BOT  public_goods=$PG  (expected 80 each = 20 seeds x 4 cells)"

# ---------- 3. Copy results data into the git repo ----------
log "copying results into repo..."
rm -rf "$REPO/results/v2"
mkdir -p "$REPO/results/v2"
cp -a "$SRC/." "$REPO/results/v2/"
log "copied. repo results/v2 file count: $(find "$REPO/results/v2" -type f | wc -l | tr -d ' ')"

# ---------- 4. Run analyses ----------
cd "$REPO"
log "running analyze_results.py (full)..."
python3 analyze_results.py > "$REPO/ANALYSIS_FULL.txt" 2>&1
log "analyze_results.py exit=$? -> ANALYSIS_FULL.txt"

log "running analyze_v2_paired.py (paired Wilcoxon)..."
python3 analyze_v2_paired.py > "$REPO/ANALYSIS_EXP_B_PAIRED.stdout" 2>&1
log "analyze_v2_paired.py exit=$? -> ANALYSIS_EXP_B_PAIRED.md"

# ---------- 5. git commit + push ----------
cd "$REPO"
git config user.email "lab-experiment@local"
git config user.name "Lab Experiment Pipeline"
log "git add -A ..."
git add -A 2>&1 | tail -3
log "git commit ..."
git commit -m "Exp B 20-seed complete: BoS+public_goods results & paired Wilcoxon analysis

- battle_of_the_sexes + public_goods, 20 seeds (42-61), 4 heterogeneous cells
  (het_notom / het_gated_atom_talk / het_dp_gated_atom_talk=CGA / het_gsaca)
- models: Qwen2.5-7B-Instruct + GLM-4-9B-0414 (4-bit, RTX 5090)
- paired Wilcoxon signed-rank analysis (n=20) per EXPERIMENT_PLAN_V2 critique #2
- metrics: BoS=$BOT, PG=$PG" 2>&1 | tail -5
COMMIT_RC=$?
log "git commit exit=$COMMIT_RC"

log "git push origin main ..."
git push origin main 2>&1 | tail -8
PUSH_RC=$?
log "git push exit=$PUSH_RC"

# ---------- 6. marker ----------
log "===== PIPELINE DONE (push_rc=$PUSH_RC) ====="
echo "PIPELINE_DONE push_rc=$PUSH_RC bot=$BOT pg=$PG date=$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$MARKER"
