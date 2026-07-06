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
PREFIXES = [
    "newlbf_lbforaging_Foraging-10x10-4p-4f-v3_",
    "sensitivity10_lbforaging_Foraging-10x10-3p-3f-v3_",
]


def load_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_name(method):
    if method.startswith("newlbf_"):
        return "newlbf"
    if method.startswith("sensitivity10_"):
        return "sensitivity10"
    return "other"


def short_name(method):
    for prefix in PREFIXES:
        if method.startswith(prefix):
            return method[len(prefix):]
    return method


def f(value):
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def ci(values, n=10000, seed=41):
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


def paired(rows, group, comparisons):
    by = defaultdict(dict)
    for row in rows:
        if group_name(row.get("method", "")) != group:
            continue
        by[short_name(row["method"])][row["seed"]] = row
    output = []
    for left, right in comparisons:
        if left not in by or right not in by:
            continue
        for metric in METRICS:
            shared = sorted(set(by[left]) & set(by[right]), key=lambda x: int(x) if str(x).isdigit() else str(x))
            diffs = []
            for seed in shared:
                lv = f(by[left][seed].get(metric))
                rv = f(by[right][seed].get(metric))
                if lv is not None and rv is not None:
                    diffs.append(lv - rv)
            if diffs:
                lo, hi = ci(diffs)
                output.append((left, right, metric, len(diffs), mean(diffs), lo, hi, ",".join(shared)))
    return output


def write_group(handle, rows, group, title):
    selected = [r for r in rows if group_name(r.get("method", "")) == group]
    if not selected:
        return
    grouped = defaultdict(list)
    for row in selected:
        grouped[short_name(row["method"])].append(row)
    handle.write(f"## {title}\n\n")
    handle.write("| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n")
    handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for method in sorted(grouped):
        vals = {metric: [f(row.get(metric)) for row in grouped[method]] for metric in METRICS}
        lo, hi = ci(vals["last_test_return"])
        handle.write(f"| {method} | {len(grouped[method])} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n")
    handle.write("\n")
    handle.write(f"## {title} Budget Accounting\n\n")
    handle.write("| method | records | triggers | penalty total | terminal bonus total | shaped steps | avg penalty/trigger | avg steps/trigger |\n")
    handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for method in sorted(grouped):
        vals = {metric: [f(row.get(metric)) for row in grouped[method]] for metric in BUDGET_METRICS}
        handle.write(f"| {method} | {fmt(mean(vals['last_llm_fd_records']))} | {fmt(mean(vals['last_llm_fd_shaping_triggers']))} | {fmt(mean(vals['last_llm_fd_shaping_penalty_total']))} | {fmt(mean(vals['last_llm_fd_shaping_terminal_bonus_total']))} | {fmt(mean(vals['last_llm_fd_shaping_episode_steps_total']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_penalty_per_trigger']))} | {fmt(mean(vals['last_llm_fd_shaping_avg_steps_per_trigger']))} |\n")
    handle.write("\n")


def write_report(rows, out_dir):
    path = os.path.join(out_dir, "ROUND10_OPTIONS12_REPORT.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Round 10 Options 1+2 Report\n\n")
        handle.write("Round 10 implements the conservative AAAI package: mechanism defense through budget accounting and sensitivity, plus one LBF-family generalization task.\n\n")
        write_group(handle, rows, "newlbf", "New LBF Family Task")
        pairs = paired(rows, "newlbf", [
            ("adaptive_0.0003_late045", "baseline"),
            ("adaptive_0.0003_late045", "uniform_budget_matched_0.0003_late045"),
            ("adaptive_0.0003_late045", "random_type_budget_matched_0.0003_late045"),
        ])
        if pairs:
            handle.write("## New LBF Paired Comparisons\n\n")
            handle.write("| comparison | metric | n | mean delta | 95% CI | seeds |\n")
            handle.write("|---|---|---:|---:|---:|---|\n")
            for left, right, metric, n, md, lo, hi, seeds in pairs:
                handle.write(f"| {left} - {right} | {metric} | {n} | {fmt(md)} | [{fmt(lo)}, {fmt(hi)}] | {seeds} |\n")
            handle.write("\n")
        write_group(handle, rows, "sensitivity10", "10x10 Sensitivity and Budget Controls")
        handle.write("## Interpretation Guide\n\n")
        handle.write("Use the new LBF family task as main-text generalization evidence only if adaptive beats baseline and remains competitive with phase-uniform/random controls. Use the sensitivity panel to show that Round 8's 0.0003/late0.45 setting is not an isolated cherry-pick and to quantify actual shaping budget.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(os.path.join(args.out_dir, "summary.csv"))
    write_report(rows, args.out_dir)

if __name__ == "__main__":
    main()
