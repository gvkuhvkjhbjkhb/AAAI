# S1 — Coverage-Certified Safe-SCA

## Research question

Can an LLM-agent team decide whether to enable forced alignment **online and
without a game-class label**, while accounting for the complete cost of the
decision window?

The experiment tests the paper's pivot claim: alignment must be earned by
observable behavioral evidence. It does not reuse Stack-A split signs, game
names, payoff matrices, or an offline arm oracle in the deployable policy.

## Frozen policy

Every Safe-SCA cell starts with `W` episodes of `NoAlign`. For every warm-up
round, it logs the joint action profile and the mean team payoff. Let

\[
\hat{s}=\overline r_{\rm different}-\overline r_{\rm same}.
\]

After the warm-up, `het_safe_sca` selects `Gated` only when all conditions
hold:

1. at least `min_stratum_observations` same-action and differentiated-action
   observations have been seen;
2. the fraction of unique observed joint action profiles is at least
   `min_profile_coverage`;
3. the one-sided bootstrap upper confidence bound of \(\hat{s}\) is smaller
   than `-tau`.

Otherwise it selects `NoAlign`. The estimator and selection function are in
[`code/safe_sca.py`](code/safe_sca.py). They accept action/reward histories
only; no game name or oracle is an input.

The agent objects are *not re-created* when the policy switches from warm-up
to commit. Therefore a Gated trust-EMA transition cost is part of the logged
end-to-end outcome, rather than being hidden by a fresh-agent counterfactual.

## Experimental matrix

All cells use the fresh lab-replicated Stack-B target: Qwen2.5-7B-Instruct +
GLM-4-9B-0414, **vLLM 0.25.1**, PyTorch `2.11.0+cu128`, Transformers `5.14.1`,
bf16 on 2× RTX 5090 (32GB, sm_120), `top_p=0.9`, horizon 5, memory 2, and 30
episodes. Verify these versions before launch and record deviations in a new
result directory; do not mix them with the earlier vLLM 0.23.0 run. Policy
order within each `(game, seed)` is rotated by a fixed seed-keyed Latin square.
Per-request generation seeds are inherited from the existing runner.

| Cell | Role | Deployable? |
|---|---|---|
| `het_notom` | NoAlign safety baseline | Yes |
| `het_gated_atom_talk` | Always-Gated risk baseline | Yes |
| `het_gsaca` | Existing integrated conditional mechanism | Yes, legacy comparator |
| `het_point_sca` | Two-arm point-estimate decision without a confidence/coverage gate | Yes |
| `het_safe_sca` | Coverage-Certified Safe-SCA | Yes |
| `het_oracle_sca` | Same warm-up, but post-warm-up arm receives game class | **No — diagnostic ceiling only** |

The test is 6 games × 20 fresh seeds × 6 cells = **720 cells**. The 30
episodes include all warm-up episodes; `s1_total_team_payoff` is the primary
endpoint. Post-commitment payoff is retained only for diagnosis.

## Runtime and GPU budget

Do not plan S1 from a nominal cells-per-hour number. The complete lab report
records about 48 hours for its 120-cell G1 job and heterogeneous per-game
latencies. First complete the 60-cell development observer run with the exact
GPU servers and worker count intended for S1, record its wall time `T_dev`,
then reserve at least **12 x `T_dev`** for the 720-cell held-out test plus
recovery time for failed shards. This is intentionally conservative: the
Safe/Point/Oracle cells carry an additional warm-up/decision path and should
not be assumed faster than the observer cell.

## Development / held-out split

1. **Development**: seeds 42–51. Run only `het_notom` to collect 30-episode
   label-free warm-up trajectories (60 cells).
2. `select_s1_config.py` is the only script that can inspect known game class,
   and it does so only on these development trajectories. It chooses one
   configuration lexicographically: minimize anti-coordination false-aligns,
   maximize coordination aligns, then prefer shorter warm-up and more
   conservative thresholds. It writes an immutable JSON config.
3. **Held-out test**: fresh seeds 62–81. `run_s1_safe_sca.py` refuses to run
   the test unless the frozen config path is supplied. The policy code does not
   receive the development labels or held-out game class.

No threshold, analysis exclusion, or seed may be changed after held-out launch.

## Commands

Set `GSACA_ROOT` to this `g123_augmentation` directory on the GPU machine.
Start the existing two vLLM servers first.

```bash
export GSACA_ROOT=/data/lab/AAAI/g123_augmentation
bash "$GSACA_ROOT/code/start_vllm.sh"

# 0. Fail closed unless the lab-replicated Stack-B environment and both local
# vLLM endpoints (with the expected model IDs) are available on exactly two
# visible RTX 5090 GPUs. Run once before development and once before
# held-out test, changing only the output directory.
python3 "$GSACA_ROOT/code/preflight_s1.py" \
  --out-dir "$GSACA_ROOT/v2_results/exp_s1_dev_warmup"

# 1. Development observer data: 6 games × 10 seeds × NoAlign.
python3 "$GSACA_ROOT/code/run_s1_safe_sca.py" --phase dev --workers 12

# 2. Freeze one label-free test configuration from development only.
python3 "$GSACA_ROOT/code/select_s1_config.py" \
  --dev-results "$GSACA_ROOT/v2_results/exp_s1_dev_warmup" \
  --frozen-config "$GSACA_ROOT/v2_results/s1_safe_sca_frozen.json"

# 3. Held-out test: 6 games × 20 new seeds × 6 policies.
python3 "$GSACA_ROOT/code/run_s1_safe_sca.py" --phase test --workers 12 \
  --safe-config "$GSACA_ROOT/v2_results/s1_safe_sca_frozen.json"

# 4. Do not analyze a partial or provenance-contaminated result directory.
python3 "$GSACA_ROOT/code/validate_s1_results.py" \
  --results "$GSACA_ROOT/v2_results/exp_s1_safe_sca_test" \
  --frozen-config "$GSACA_ROOT/v2_results/s1_safe_sca_frozen.json"

# 5. Preregistered total-horizon analysis.
python3 "$GSACA_ROOT/code/analyze_s1_safe_sca.py" \
  --results "$GSACA_ROOT/v2_results/exp_s1_safe_sca_test" \
  --safety-margin 0.10
```

Before a GPU run, inspect the commands without launching workers:

```bash
python3 "$GSACA_ROOT/code/run_s1_safe_sca.py" --phase dev --dry-run
```

## Success criteria

The primary safety criterion is met only when Safe-SCA's paired 95% bootstrap
lower confidence bound against NoAlign is at least `-0.10` in **each** of
Chicken, Deadlock, and Hawk-Dove. Report, regardless of outcome:

- total-horizon team payoff and paired bootstrap intervals against NoAlign;
- anti-coordination false-align rate;
- coordination/boundary false-abstain rate;
- Safe-SCA's recovery of the Always-Gated gain in coordination games;
- warm-up and commit payoff separately;
- `decision.json` coverage evidence and reasons for every Safe-SCA cell.

The method-paper utility gate is fixed before launch: in at least **two** of
Stag Hunt, Battle of the Sexes, and Public Goods where Always-Gated has a
positive total-horizon gain over NoAlign, Safe-SCA must recover at least
**30%** of that gain. `analyze_s1_safe_sca.py` reports `method_paper_pass`
only when this gate and the three-game safety gate both pass. This is a writing
gate, not a claim that Safe-SCA must dominate Always-Gated.

If Safe-SCA is safe but frequently abstains, report this as the safety–utility
tradeoff. If it fails the anti-coordination safety criterion, the paper must
not claim a deployable safe controller; the result remains evidence for the
online-identification-failure story.

## Output contract

```text
v2_results/
├── exp_s1_dev_warmup/
│   ├── {ENVIRONMENT_MANIFEST_S1.json,CONFIG_SNAPSHOT_S1.json}
│   └── <game>/seed_<seed>/het_notom/{metrics.json,trajectories.jsonl}
├── s1_safe_sca_frozen.json
├── s1_safe_sca_frozen_selection_report.json
└── exp_s1_safe_sca_test/
    ├── CONFIG_SNAPSHOT_S1.json
    ├── ENVIRONMENT_MANIFEST_S1.json
    ├── <game>/seed_<seed>/<cell>/metrics.json
    ├── <game>/seed_<seed>/{arm_order.json,...}
    ├── <game>/seed_<seed>/het_safe_sca/decision.json
    └── s1_safe_sca_summary.{json,md}
    └── S1_INTEGRITY_REPORT.json
```

`CONFIG_SNAPSHOT_S1.json` is immutable on resume. A changed command or frozen
configuration must use a new result directory rather than silently mixing
protocols.

## CPU-only checks

```bash
cd "$GSACA_ROOT/code"
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m py_compile safe_sca.py run_experiment_local.py \
  run_s1_safe_sca.py select_s1_config.py analyze_s1_safe_sca.py \
  preflight_s1.py validate_s1_results.py
```
