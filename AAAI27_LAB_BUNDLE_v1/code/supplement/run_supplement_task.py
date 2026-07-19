#!/usr/bin/env python3
"""Run one immutable matrix/game seed task for supplemental P0/P1/P2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from p3_label_variants import labels_for, register_with_baseline as register_label_swap
from p3_matrices import PROMPT_GAME_NAME, get_spec, register_with_baseline as register_p3
from supplement_protocol import SOURCE_GAMES, load_protocol


def runner_namespace(protocol: dict, out_dir: Path, cells: list[str], episodes: int):
    safe = protocol["safe_sca"]
    sampling = protocol["sampling"]
    return argparse.Namespace(
        out_dir=str(out_dir), episodes=episodes,
        horizon=sampling["horizon"], memory=sampling["memory"], cells=cells,
        log_every=100, gsaca_warmup=5, gate_trust_threshold=0.6,
        gate_ema_alpha=0.3, atom_warmup=3, payoff_noise_std=0.0,
        abstain_tau=0.4, bandit_k=protocol["p2"]["bandit_k"],
        role_asymmetric_hint=False, history_split_hint=False,
        adaptive_intervention=False, adaptive_interv_threshold=0.3,
        point_sca_tau=0.0, safe_warmup=safe["warmup_episodes"],
        safe_tau=safe["tau"], safe_confidence=safe["confidence"],
        safe_bootstrap_samples=safe["bootstrap_samples"],
        safe_min_profile_coverage=safe["min_profile_coverage"],
        safe_min_stratum_observations=safe["min_stratum_observations"],
        top_p=sampling["top_p"], models_het=list(protocol["models_het"]),
        model_homo=protocol["models_het"][0], use_vllm=True,
        latin_square=protocol["execution"]["latin_square"],
        gen_seed_base=sampling["gen_seed_base"], force=False,
    )


def episode_count(protocol: dict, domain: str, context: str) -> int:
    if domain == "p3":
        return int(protocol["p3"]["episodes"])
    if context == "public_goods":
        return int(protocol["source"]["episodes_public_goods"])
    return int(protocol["source"]["episodes_default"])


def write_visibility(root: Path, experiment_name: str, domain: str,
                     context: str) -> None:
    record: dict = {
        "schema_version": 1,
        "experiment": experiment_name,
        "domain": domain,
        "context": context,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if domain == "p3":
        spec = get_spec(context)
        record.update({
            "prompt_game_name": PROMPT_GAME_NAME,
            "analysis_category_available_to_agent_or_controller": False,
            "matrix_id_available_to_agent_or_controller": False,
            "payoff_tensor_sha256_is_recorded_only_in_campaign_snapshot": True,
        })
        if experiment_name == "p0":
            record.update({
                "prompt_action_labels": list(spec.action_labels),
                "payoff_table_available_only_in_policy": "het_payoff_prompt",
                "fixed_arm_controls_receive_payoff_table": False,
                "safe_sca_controller_receives": ["realized_actions", "realized_rewards"],
                "payoff_prompt_controller": "none",
            })
        elif experiment_name == "p1":
            record.update({
                "original_action_labels": list(spec.action_labels),
                "prompt_action_labels": list(labels_for(context)),
                "payoff_table_available_to_agents": False,
                "safe_sca_controller_receives": ["realized_actions", "realized_rewards"],
            })
        else:
            record.update({
                "prompt_action_labels": list(spec.action_labels),
                "payoff_table_available_to_agents": False,
                "bandit_receives": ["realized_team_mean_payoff"],
            })
    target = root / context / f"{experiment_name.upper()}_VISIBILITY.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def run_teammean_bandit(experiment, protocol: dict, output: Path,
                        context: str, seed: int, episodes: int) -> None:
    """Run K/K probes, select using team mean, and record probe-cost total."""
    n_agents = 4 if context == "public_goods" else None
    args = runner_namespace(protocol, output, [], episodes)
    seed_dir = output / context / f"seed_{seed}"
    metrics_path = seed_dir / "het_bandit_teammean" / "metrics.json"
    if metrics_path.exists():
        print(f"[skip] {context}/seed_{seed}/het_bandit_teammean exists", flush=True)
        return
    experiment.random.seed(seed)
    experiment.np.random.seed(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "arm_order.json").write_text(json.dumps({
        "game": context, "seed": seed,
        "probe_order": ["Gated", "NoAlign"] if seed % 2 else ["NoAlign", "Gated"],
        "tie_break": "NoAlign", "selection_endpoint": "team_mean_payoff",
    }, indent=2) + "\n", encoding="utf-8")
    config = experiment.make_config(
        context, "het_bandit", seed, n_agents=n_agents,
        horizon=args.horizon, memory=args.memory, args=args,
    )
    config["cell_name"] = "het_bandit_teammean"
    metrics = experiment.run_bandit_cell(
        config, episodes, str(seed_dir), log_every=100,
        bandit_k=protocol["p2"]["bandit_k"], noise_std=0.0,
    )
    probe_no = list(metrics["bandit_probe_payoffs_NoAlign"])
    probe_gated = list(metrics["bandit_probe_payoffs_Gated"])
    n_commit = int(metrics["bandit_n_commit"])
    commit_mean = float(metrics["team_mean_payoff"])
    total = (sum(probe_no) + sum(probe_gated) + n_commit * commit_mean) / episodes
    metrics.update({
        "supplement_schema_version": 1,
        "bandit_selection_endpoint": "team_mean_payoff",
        "bandit_commit_team_mean_payoff": commit_mean,
        "bandit_online_total_team_mean_payoff": float(total),
        "bandit_total_includes_both_probe_arms": True,
        "bandit_total_episode_count": episodes,
        "historical_cooperation_payoff_is_not_an_endpoint": True,
    })
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one P0/P1/P2 supplemental task")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--experiment", choices=["p0", "p1", "p2"], required=True)
    parser.add_argument("--domain", choices=["p3", "source"], required=True)
    parser.add_argument("--context", required=True, help="P3 matrix ID or source game")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol.resolve())
    if args.domain == "p3":
        if args.context not in protocol["p3"]["matrix_ids"] or args.seed not in protocol["p3"]["seeds"]:
            parser.error("P3 context/seed is outside the frozen grid")
    else:
        if args.context not in SOURCE_GAMES or args.seed not in protocol["source"]["seeds"]:
            parser.error("Source context/seed is outside the frozen S2 grid")
        if args.experiment in {"p0", "p1"}:
            parser.error("P0 and P1 are confirmatory P3-grid controls; source is unsupported")

    import hettom_baseline as hb
    import run_experiment_local as experiment

    if args.domain == "p3":
        if args.experiment == "p1":
            register_label_swap(hb)
        else:
            register_p3(hb)
    experiment.hb.build_agents = experiment._patched_build_agents_vllm
    output = args.out_dir.resolve() / {
        "p0": "p0_payoff_prompt",
        "p1": "p1_label_swap",
        "p2": "p2_teammean_bandit_p3" if args.domain == "p3" else "p2_teammean_bandit_source",
    }[args.experiment]
    write_visibility(output, args.experiment, args.domain, args.context)
    episodes = episode_count(protocol, args.domain, args.context)
    if args.experiment == "p0":
        run_args = runner_namespace(protocol, output, list(protocol["p0"]["policies"]), episodes)
        experiment.run_game_seed(run_args, args.context, args.seed)
    elif args.experiment == "p1":
        run_args = runner_namespace(protocol, output, list(protocol["p1"]["policies"]), episodes)
        experiment.run_game_seed(run_args, args.context, args.seed)
    else:
        run_teammean_bandit(experiment, protocol, output, args.context, args.seed, episodes)


if __name__ == "__main__":
    main()
