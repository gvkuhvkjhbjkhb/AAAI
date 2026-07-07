import argparse
import csv
import os
import random
from collections import defaultdict


METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]
BUDGET_METRICS = [
    "last_llm_fd_records",
    "last_llm_fd_shaping_triggers",
    "last_llm_fd_shaping_penalty_total",
    "last_llm_fd_shaping_terminal_bonus_total",
    "last_llm_fd_shaping_episode_steps_total",
    "last_llm_fd_shaping_avg_penalty_per_trigger",
    "last_llm_fd_shaping_avg_steps_per_trigger",
]
PREFIX = "vmascal_vmas-navigation_"
METHODS = ["baseline", "adaptive", "uniform", "random"]


def load_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_method(full_method):
    method = full_method[len(PREFIX):] if full_method.startswith(PREFIX) else full_method
    for prefix in METHODS:
        tag = prefix + "_p"
        if method.startswith(tag):
            return prefix, method[len(tag):]
    return method, "unknown"


def as_float(value):
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def ci(values, n=10000, seed=53):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        draws.append(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals))
    draws.sort()
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def grouped_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        method, penalty = parse_method(row.get("method", ""))
        groups[(penalty, method)].append(row)
    return groups


def penalty_key(penalty):
    if penalty == "unknown":
        return 999.0
    return float(penalty.replace("m", "-"))


def paired(groups, penalty, left="adaptive"):
    output = []
    for right in ["baseline", "uniform", "random"]:
        lrows = {row["seed"]: row for row in groups.get((penalty, left), [])}
        rrows = {row["seed"]: row for row in groups.get((penalty, right), [])}
        shared = sorted(set(lrows) & set(rrows), key=lambda x: int(x) if str(x).isdigit() else str(x))
        if not shared:
            continue
        for metric in METRICS:
            diffs = []
            for seed in shared:
                lv = as_float(lrows[seed].get(metric))
                rv = as_float(rrows[seed].get(metric))
                if lv is not None and rv is not None:
                    diffs.append(lv - rv)
            if diffs:
                lo, hi = ci(diffs)
                output.append((left, right, metric, len(diffs), mean(diffs), lo, hi, ",".join(shared)))
    return output


def score_penalty(groups, penalty):
    scores = []
    for right in ["baseline", "uniform", "random"]:
        lrows = {row["seed"]: row for row in groups.get((penalty, "adaptive"), [])}
        rrows = {row["seed"]: row for row in groups.get((penalty, right), [])}
        shared = sorted(set(lrows) & set(rrows), key=lambda x: int(x) if str(x).isdigit() else str(x))
        for metric, weight in [("last_test_return", 1.0), ("train_auc", 0.5)]:
            diffs = []
            for seed in shared:
                lv = as_float(lrows[seed].get(metric))
                rv = as_float(rrows[seed].get(metric))
                if lv is not None and rv is not None:
                    diffs.append(lv - rv)
            if diffs:
                scores.append(weight * mean(diffs))
    return sum(scores) if scores else None


def write_report(rows, out_dir):
    groups = grouped_rows(rows)
    penalties = sorted({penalty for penalty, _ in groups}, key=penalty_key)
    path = os.path.join(out_dir, "ROUND11_VMAS_CALIBRATION_REPORT.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Round 11 VMAS Reward-Scale Calibration Report\n\n")
        handle.write("Round 11 tests whether Round 9's VMAS weakness was caused by transferring the LBF-tuned 0.0003 reward-shaping scale into a dense-navigation reward landscape.\n\n")
        for penalty in penalties:
            if penalty == "unknown":
                continue
            handle.write(f"## Penalty {penalty}\n\n")
            handle.write("| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for method in METHODS:
                items = groups.get((penalty, method), [])
                if not items:
                    continue
                vals = {metric: [as_float(row.get(metric)) for row in items] for metric in METRICS}
                lo, hi = ci(vals["last_test_return"])
                handle.write(f"| {method} | {len(items)} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n")
            handle.write("\n")
            handle.write(f"## Penalty {penalty} Budget Accounting\n\n")
            handle.write("| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for method in METHODS:
                items = groups.get((penalty, method), [])
                if not items:
                    continue
                vals = {metric: [as_float(row.get(metric)) for row in items] for metric in BUDGET_METRICS}
                handle.write(f"| {method} | {fmt(mean(vals['last_llm_fd_records']))} | {fmt(mean(vals['last_llm_fd_shaping_triggers']))} | {fmt(mean(vals['last_llm_fd_shaping_penalty_total']))} | {fmt(mean(vals['last_llm_fd_shaping_terminal_bonus_total']))} | {fmt(mean(vals['last_llm_fd_shaping_episode_steps_total']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_penalty_per_trigger']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_steps_per_trigger']))} |\n")
            handle.write("\n")
            pairs = paired(groups, penalty)
            if pairs:
                handle.write(f"## Penalty {penalty} Paired Comparisons\n\n")
                handle.write("| comparison | metric | n | mean delta | 95% CI | shared seeds |\n")
                handle.write("|---|---|---:|---:|---:|---|\n")
                for left, right, metric, n, md, lo, hi, seeds in pairs:
                    handle.write(f"| {left} - {right} | {metric} | {n} | {fmt(md)} | [{fmt(lo)}, {fmt(hi)}] | {seeds} |\n")
                handle.write("\n")
        handle.write("## Penalty Selection\n\n")
        scored = [(penalty, score_penalty(groups, penalty)) for penalty in penalties if penalty != "unknown"]
        scored = [(p, s) for p, s in scored if s is not None]
        scored.sort(key=lambda item: item[1], reverse=True)
        handle.write("| rank | penalty | heuristic score |\n")
        handle.write("|---:|---:|---:|\n")
        for rank, (penalty, score) in enumerate(scored, 1):
            handle.write(f"| {rank} | {penalty} | {fmt(score)} |\n")
        handle.write("\n")
        if scored:
            handle.write(f"Recommended full-run candidate: penalty `{scored[0][0]}` if adaptive is not clearly worse than random-type and is positive versus baseline. If all scores are negative or random-type dominates, keep VMAS as a transparent reward-scale limitation instead of expanding.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(os.path.join(args.out_dir, "summary.csv"))
    write_report(rows, args.out_dir)


if __name__ == "__main__":
    main()
