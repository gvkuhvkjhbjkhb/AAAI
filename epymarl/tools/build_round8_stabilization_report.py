import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict


METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]


def load_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short_name(method):
    for marker in [
        "main10x10_lbforaging_Foraging-10x10-3p-3f-v3_",
        "generalization12_lbforaging_Foraging-12x12-3p-4f-v3_",
        "generalization_lbforaging_Foraging-12x12-3p-4f-v3_",
        "pilot10x10_lbforaging_Foraging-10x10-3p-3f-v3_",
        "pilot12x12_lbforaging_Foraging-12x12-3p-4f-v3_",
    ]:
        if method.startswith(marker):
            return method[len(marker) :]
    return method.rsplit("_", 1)[-1] if method else method


def env_group(method):
    if method.startswith("main10x10_") or method.startswith("pilot10x10_"):
        return "10x10"
    if method.startswith("generalization12_") or method.startswith("pilot12x12_"):
        return "12x12"
    if "Foraging-10x10" in method:
        return "10x10"
    if "Foraging-12x12" in method:
        return "12x12"
    return "other"


def as_float(value):
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except ValueError:
        return None


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def std(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def ci(values, n=10000, seed=17):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return draws[int(0.025 * n)], draws[int(0.975 * n)]


def cliffs_delta(xs, ys):
    xs = [x for x in xs if x is not None]
    ys = [y for y in ys if y is not None]
    if not xs or not ys:
        return None
    greater = sum(1 for x in xs for y in ys if x > y)
    lower = sum(1 for x in xs for y in ys if x < y)
    return (greater - lower) / (len(xs) * len(ys))


def paired_table(rows, group, comparisons):
    by = defaultdict(dict)
    for row in rows:
        if env_group(row.get("method", "")) != group:
            continue
        by[short_name(row["method"])][row["seed"]] = row
    output = []
    for left, right in comparisons:
        for metric in METRICS:
            shared = sorted(set(by.get(left, {})) & set(by.get(right, {})), key=lambda x: int(x) if str(x).isdigit() else str(x))
            diffs = []
            left_vals = []
            right_vals = []
            for seed in shared:
                lv = as_float(by[left][seed].get(metric))
                rv = as_float(by[right][seed].get(metric))
                if lv is None or rv is None:
                    continue
                diffs.append(lv - rv)
                left_vals.append(lv)
                right_vals.append(rv)
            if not diffs:
                continue
            lo, hi = ci(diffs)
            output.append((left, right, metric, len(diffs), mean(diffs), lo, hi, cliffs_delta(left_vals, right_vals), ",".join(shared)))
    return output


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def write_report(rows, out_path):
    groups = defaultdict(lambda: defaultdict(list))
    by_values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = env_group(row.get("method", ""))
        method = short_name(row.get("method", ""))
        groups[group][method].append(row)
        for metric in METRICS:
            by_values[(group, method)][metric].append(as_float(row.get(metric)))

    comparisons = [
        ("adaptive_0.0003_late045", "baseline"),
        ("adaptive_0.0003_late045", "uniform_budget_matched_0.0003_late045"),
        ("adaptive_0.0003_late045", "random_type_budget_matched_0.0003_late045"),
        ("adaptive_0.0003_late045", "semantic_shuffled_budget_matched_0.0003_late045"),
        ("adaptive_0.0003_late045", "diagnosis_only"),
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("# Round 8 AAAI Stabilization Report\n\n")
        handle.write("This report is organized around the conservative AAAI claim: failure-triggered adaptive reward shaping, not LLM semantic causality, is the main mechanism.\n\n")
        for group in ["10x10", "12x12", "other"]:
            if group not in groups:
                continue
            handle.write(f"## {group} Grouped Results\n\n")
            handle.write("| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for method in sorted(groups[group]):
                vals = by_values[(group, method)]
                lo, hi = ci(vals["last_test_return"])
                handle.write(
                    f"| {method} | {len(groups[group][method])} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n"
                )
            handle.write("\n")
            pairs = paired_table(rows, group, comparisons)
            if pairs:
                handle.write(f"## {group} Paired Strong-Control Comparisons\n\n")
                handle.write("| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |\n")
                handle.write("|---|---|---:|---:|---:|---:|---|\n")
                for left, right, metric, n, md, lo, hi, cliff, seeds in pairs:
                    handle.write(f"| {left} - {right} | {metric} | {n} | {fmt(md)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(cliff)} | {seeds} |\n")
                handle.write("\n")
        handle.write("## Decision Use\n\n")
        handle.write("Use 10x10 as the established main-task evidence and 12x12 as the required generalization/stress-test panel. A submission-grade result requires adaptive shaping to beat baseline on both environments and to remain competitive with budget-matched uniform and shuffled/random controls.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extra-summary", action="append", default=[])
    args = parser.parse_args()
    rows = []
    for path in args.extra_summary + [os.path.join(args.out_dir, "summary.csv")]:
        rows.extend(load_csv(path))
    write_report(rows, os.path.join(args.out_dir, "ROUND8_STABILIZATION_REPORT.md"))


if __name__ == "__main__":
    main()
