import argparse
import csv
import math
import os
import random
from collections import defaultdict

METRICS = ["last_test_return", "train_auc", "best_train_return", "stability_gap"]
PREFIXES = [
    "lbf12ext_lbforaging_Foraging-12x12-3p-4f-v3_",
    "rwaretiny_rware_rware-tiny-2ag-v2_",
    "vmasnav_vmas-navigation_",
]

def load_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def env_group(method):
    if method.startswith("lbf12ext_"):
        return "lbf12ext"
    if method.startswith("rwaretiny_"):
        return "rwaretiny"
    if method.startswith("vmasnav_"):
        return "vmasnav"
    return "other"

def short_name(method):
    for prefix in PREFIXES:
        if method.startswith(prefix):
            return method[len(prefix):]
    return method

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

def ci(values, n=10000, seed=23):
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
    gt = sum(1 for x in xs for y in ys if x > y)
    lt = sum(1 for x in xs for y in ys if x < y)
    return (gt - lt) / (len(xs) * len(ys))

def fmt(value):
    return "NA" if value is None else f"{value:.4f}"

def paired(rows, group):
    by = defaultdict(dict)
    for row in rows:
        if env_group(row.get("method", "")) != group:
            continue
        by[short_name(row["method"])][row["seed"]] = row
    pairs = [
        ("adaptive_0.0003_late045", "baseline"),
        ("adaptive_0.0003_late045", "diagnosis_only"),
        ("adaptive_0.0003_late045", "uniform_budget_matched_0.0003_late045"),
        ("adaptive_0.0003_late045", "random_type_budget_matched_0.0003_late045"),
        ("adaptive_0.0003_late045", "semantic_shuffled_budget_matched_0.0003_late045"),
    ]
    output = []
    for left, right in pairs:
        if left not in by or right not in by:
            continue
        for metric in METRICS:
            shared = sorted(set(by[left]) & set(by[right]), key=lambda x: int(x) if str(x).isdigit() else str(x))
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
            if diffs:
                lo, hi = ci(diffs)
                output.append((left, right, metric, len(diffs), mean(diffs), lo, hi, cliffs_delta(left_vals, right_vals), ",".join(shared)))
    return output

def write_report(rows, out_dir):
    grouped = defaultdict(lambda: defaultdict(list))
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = env_group(row.get("method", ""))
        method = short_name(row.get("method", ""))
        grouped[group][method].append(row)
        for metric in METRICS:
            values[(group, method)][metric].append(as_float(row.get(metric)))
    path = os.path.join(out_dir, "ROUND9_SUPPLEMENT_REPORT.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Round 9 Supplemental Stabilization Report\n\n")
        handle.write("Round 9 targets the remaining AAAI risk: cross-domain generalization and stronger 12x12 seed coverage. It keeps the conservative failure-triggered adaptive shaping claim and treats semantic diagnosis as exploratory.\n\n")
        for group, title in [("lbf12ext", "LBF 12x12 Seed Extension"), ("rwaretiny", "RWARE Tiny Cross-Domain"), ("vmasnav", "VMAS Navigation Cross-Domain")]:
            if group not in grouped:
                continue
            handle.write(f"## {title}\n\n")
            handle.write("| method | n | last test | last test 95% CI | train AUC | best train | stability gap |\n")
            handle.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for method in sorted(grouped[group]):
                vals = values[(group, method)]
                lo, hi = ci(vals["last_test_return"])
                handle.write(f"| {method} | {len(grouped[group][method])} | {fmt(mean(vals['last_test_return']))} | [{fmt(lo)}, {fmt(hi)}] | {fmt(mean(vals['train_auc']))} | {fmt(mean(vals['best_train_return']))} | {fmt(mean(vals['stability_gap']))} |\n")
            handle.write("\n")
            pairs = paired(rows, group)
            if pairs:
                handle.write(f"## {title} Paired Comparisons\n\n")
                handle.write("| comparison | metric | n | mean delta | 95% CI | Cliff delta | shared seeds |\n")
                handle.write("|---|---|---:|---:|---:|---:|---|\n")
                for left, right, metric, n, md, lo, hi, cliff, seeds in pairs:
                    handle.write(f"| {left} - {right} | {metric} | {n} | {fmt(md)} | [{fmt(lo)}, {fmt(hi)}] | {fmt(cliff)} | {seeds} |\n")
                handle.write("\n")
        handle.write("## Decision Rule\n\n")
        handle.write("Use RWARE/VMAS as cross-domain evidence only if adaptive is directionally positive against baseline and not worse than budget-matched controls. If cross-domain results are mixed, report them transparently as limits and rely on the completed LBF 10x10/12x12 package.\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(os.path.join(args.out_dir, "summary.csv"))
    write_report(rows, args.out_dir)

if __name__ == "__main__":
    main()
