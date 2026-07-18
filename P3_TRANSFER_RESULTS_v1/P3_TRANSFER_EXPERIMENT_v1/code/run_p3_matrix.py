#!/usr/bin/env python3
"""Run one or more frozen P3 matrix/seed tasks using the S1/S2 runner.

The only game-specific code is the hidden environment registration.  Every
agent prompt receives the same generic game name and an opaque action surface;
Safe-SCA itself receives action/reward observations only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from p3_matrices import PROMPT_GAME_NAME, get_spec, register_with_baseline
from p3_protocol import load_protocol


def build_runner_args(protocol: dict, output: Path, *, force: bool):
    """Build the Namespace consumed by run_experiment_local.run_game_seed."""
    import argparse as _argparse
    safe = protocol["safe_sca"]
    return _argparse.Namespace(
        out_dir=str(output), episodes=protocol["episodes"], horizon=protocol["horizon"],
        memory=protocol["memory"], cells=list(protocol["policies"]),
        log_every=100, gsaca_warmup=5, gate_trust_threshold=0.6,
        gate_ema_alpha=0.3, atom_warmup=3, payoff_noise_std=0.0,
        abstain_tau=0.4, bandit_k=5, role_asymmetric_hint=False,
        history_split_hint=False, adaptive_intervention=False,
        adaptive_interv_threshold=0.3, point_sca_tau=0.0,
        safe_warmup=safe["warmup_episodes"], safe_tau=safe["tau"],
        safe_confidence=safe["confidence"], safe_bootstrap_samples=safe["bootstrap_samples"],
        safe_min_profile_coverage=safe["min_profile_coverage"],
        safe_min_stratum_observations=safe["min_stratum_observations"],
        top_p=protocol["top_p"], models_het=list(protocol["models_het"]),
        model_homo=protocol["models_het"][0], use_vllm=True,
        latin_square=True, gen_seed_base=protocol["gen_seed_base"], force=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run immutable P3 transfer matrix tasks")
    parser.add_argument("--base-root", type=Path, required=True,
                        help="g123_augmentation directory containing S1/S2 base code")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--matrix-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--force", action="store_true",
                        help="overwrite completed cells; prohibited for confirmatory P3")
    args = parser.parse_args()
    if args.force:
        parser.error("--force is prohibited for confirmatory P3; use a new result root instead")
    protocol = load_protocol(args.protocol.resolve())
    if args.matrix_id not in protocol["matrix_ids"]:
        parser.error("matrix-id is not frozen in the P3 protocol")
    if not args.seeds or any(seed not in protocol["seeds"] for seed in args.seeds):
        parser.error("seeds must be a nonempty subset of frozen P3 seeds 102..111")
    base_code = args.base_root.resolve() / "code"
    if not (base_code / "run_experiment_local.py").exists():
        parser.error(f"base-root does not contain code/run_experiment_local.py: {args.base_root}")

    sys.path.insert(0, str(base_code))
    import hettom_baseline as hb
    import run_experiment_local as experiment

    register_with_baseline(hb)
    # The base runner's vLLM patch retains the normal model endpoint map.
    experiment.hb.build_agents = experiment._patched_build_agents_vllm
    task_args = build_runner_args(protocol, args.out_dir.resolve(), force=False)
    spec = get_spec(args.matrix_id)
    visibility = {
        "matrix_id_for_filesystem_only": spec.matrix_id,
        "prompt_game_name": PROMPT_GAME_NAME,
        "prompt_action_labels": list(spec.action_labels),
        "payoff_in_prompt": False,
        "controller_receives": ["realized_actions", "realized_rewards"],
        "controller_does_not_receive": ["matrix_id", "analysis_category", "payoff_table"],
    }
    matrix_root = args.out_dir.resolve() / args.matrix_id
    matrix_root.mkdir(parents=True, exist_ok=True)
    (matrix_root / "P3_CONTROLLER_VISIBILITY.json").write_text(
        __import__("json").dumps(visibility, indent=2) + "\n", encoding="utf-8")
    for seed in args.seeds:
        experiment.run_game_seed(task_args, args.matrix_id, seed)


if __name__ == "__main__":
    main()
