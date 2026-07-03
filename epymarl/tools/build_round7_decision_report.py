import argparse
import csv
import os
import random
from collections import defaultdict

METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]
MAIN_PREFIX = "main10x10_lbforaging_Foraging-10x10-3p-3f-v3_"
GEN_PREFIX = "generalization_lbforaging_Foraging-12x12-3p-4f-v3_"


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_method(name):
    if name.startswith(MAIN_PREFIX):
        return "main10x10", name[len(MAIN_PREFIX):]
    if name.startswith(GEN_PREFIX):
        return "generalization", name[len(GEN_PREFIX):]
    return "unknown", name


def mean(vals):
    vals = [float(v) for v in vals if v not in ("", None)]
    return sum(vals) / len(vals) if vals else None


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def bootstrap_ci(vals, n_boot=10000, seed=7):
    vals = [float(v) for v in vals if v not in ("", None)]
    if not vals:
        return None, None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def merge_rows(paths):
    best = {}
    for path in paths:
        for row in load_csv(path):
            if not row.get("last_test_return"):
                continue
            phase, method = normalize_method(row["method"])
            key = (phase, method, row["seed"])
            best[key] = dict(row, phase=phase, method=method)
    return list(best.values())


def paired(rows, phase, left, right, metric):
    by = defaultdict(dict)
    for row in rows:
        if row["phase"] != phase or not row.get(metric):
            continue
        by[row["method"]][row["seed"]] = float(row[metric])
    seeds = sorted(set(by[left]) & set(by[right]), key=int)
    return [by[left][seed] - by[right][seed] for seed in seeds], seeds


def write_group_table(h, rows, phase):
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == phase:
            groups[row["method"]].append(row)
    h.write(f"## {phase} Grouped Results\n\n")
    h.write("| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n")
    h.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for method in sorted(groups):
        items = groups[method]
        vals = [r.get("last_test_return", "") for r in items]
        lo, hi = bootstrap_ci(vals)
        h.write(
            f"| {method} | {len(items)} | {fmt(mean(vals))} | "
            f"[{fmt(lo)}, {fmt(hi)}] | {fmt(mean(r.get('train_auc','') for r in items))} | "
            f"{fmt(mean(r.get('best_train_return','') for r in items))} | {fmt(mean(r.get('stability_gap','') for r in items))} |\n"
        )
    h.write("\n")


def write_report(rows, output):
    with open(output, "w", encoding="utf-8") as h:
        h.write("# Round 7 Decisive AAAI Report\n\n")
        h.write("This report merges Round 6 controls with the completed Round 7 decisive controls while keeping 10x10 and 12x12 results separate.\n\n")
        write_group_table(h, rows, "main10x10")
        write_group_table(h, rows, "generalization")
        comparisons = [
            ("semantic_gate_0.0003_late045", "random_type_matched_0.0003_late045"),
            ("semantic_gate_0.0003_late045", "random_type_0.0003_late045"),
            ("semantic_gate_0.0003_late045", "random_type_0.0003_late060"),
            ("semantic_gate_0.0003_late045", "adaptive_0.0003_late045"),
            ("semantic_gate_0.0003_late045", "uniform_0.0003"),
            ("semantic_gate_0.0003_late045", "baseline"),
            ("random_type_matched_0.0003_late045", "adaptive_0.0003_late045"),
            ("random_type_matched_0.0003_late045", "uniform_0.0003"),
        ]
        h.write("## main10x10 Paired Decision Comparisons\n\n")
        h.write("| comparison | metric | n | mean delta | 95% CI | shared seeds |\n")
        h.write("|---|---|---:|---:|---:|---|\n")
        for left, right in comparisons:
            for metric in METRICS:
                diffs, seeds = paired(rows, "main10x10", left, right, metric)
                if not diffs:
                    continue
                lo, hi = bootstrap_ci(diffs)
                h.write(f"| {left} - {right} | {metric} | {len(diffs)} | {fmt(mean(diffs))} | [{fmt(lo)}, {fmt(hi)}] | {','.join(seeds)} |\n")
        h.write("\n## generalization Paired Stress-Test Comparisons\n\n")
        h.write("| comparison | metric | n | mean delta | 95% CI | shared seeds |\n")
        h.write("|---|---|---:|---:|---:|---|\n")
        gen_comparisons = [
            ("semantic_gate_0.0003_late045", "baseline"),
            ("semantic_gate_0.0003_late045", "uniform_0.0002"),
            ("semantic_gate_0.0003_late045", "adaptive_0.0003_late045"),
            ("adaptive_0.0003_late045", "baseline"),
            ("uniform_0.0002", "baseline"),
        ]
        for left, right in gen_comparisons:
            for metric in ["last_test_return", "train_auc"]:
                diffs, seeds = paired(rows, "generalization", left, right, metric)
                if not diffs:
                    continue
                lo, hi = bootstrap_ci(diffs)
                h.write(f"| {left} - {right} | {metric} | {len(diffs)} | {fmt(mean(diffs))} | [{fmt(lo)}, {fmt(hi)}] | {','.join(seeds)} |\n")
        h.write("\n## Submission Rule\n\n")
        h.write("The strong semantic-causality claim is not supported unless semantic_gate beats matched random-type controls. When matched random remains tied or stronger, the safest AAAI framing is calibrated failure-triggered adaptive reward shaping, with semantic diagnosis used for interpretable gating and analysis rather than claimed as the sole causal mechanism.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("csvs", nargs="+")
    args = parser.parse_args()
    write_report(merge_rows(args.csvs), args.output)

if __name__ == "__main__":
    main()
