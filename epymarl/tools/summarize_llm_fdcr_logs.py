import argparse
import csv
import math
import os
import random
import re
from collections import defaultdict


METRICS = [
    "last_train_return",
    "best_train_return",
    "train_auc",
    "stability_gap",
    "last_test_return",
    "best_test_return",
]


def parse_log(path):
    text = open(path, encoding="utf-8", errors="ignore").read()
    returns = [float(x) for x in re.findall(r"return_mean:\s+([-0-9.]+)", text)]
    tests = [float(x) for x in re.findall(r"test_return_mean:\s+([-0-9.]+)", text)]
    records = [float(x) for x in re.findall(r"llm_fd_records:\s+([-0-9.]+)", text)]
    return {
        "last_train_return": returns[-1] if returns else "",
        "best_train_return": max(returns) if returns else "",
        "train_auc": sum(returns) / len(returns) if returns else "",
        "stability_gap": (max(returns) - returns[-1]) if returns else "",
        "last_test_return": tests[-1] if tests else "",
        "best_test_return": max(tests) if tests else "",
        "last_llm_fd_records": records[-1] if records else "",
    }


def mean(values):
    values = [float(v) for v in values if v != ""]
    return sum(values) / len(values) if values else ""


def std(values):
    values = [float(v) for v in values if v != ""]
    if len(values) < 2:
        return ""
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def bootstrap_ci(values, n_boot=10000, seed=0):
    values = [float(v) for v in values if v != ""]
    if not values:
        return "", ""
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def paired_differences(rows, baseline, metric):
    by_method_seed = defaultdict(dict)
    for row in rows:
        if row[metric] == "":
            continue
        by_method_seed[row["method"]][row["seed"]] = float(row[metric])
    baseline_values = by_method_seed.get(baseline, {})
    diffs = {}
    for method, seed_values in by_method_seed.items():
        if method == baseline:
            continue
        shared = sorted(set(seed_values) & set(baseline_values), key=int)
        diffs[method] = [seed_values[seed] - baseline_values[seed] for seed in shared]
    return diffs


def write_outputs(out_dir, baseline):
    rows = []
    for path in sorted(os.listdir(os.path.join(out_dir, "logs"))):
        if not path.endswith(".log"):
            continue
        name = path[:-4]
        method, seed = name.rsplit("_seed", 1)
        row = {"method": method, "seed": seed}
        row.update(parse_log(os.path.join(out_dir, "logs", path)))
        rows.append(row)

    fieldnames = ["method", "seed"] + METRICS + ["last_llm_fd_records"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"Summary for {out_dir}\n\n")
        handle.write("Grouped mean, std, and 95 percent bootstrap CI\n")
        for method, items in sorted(groups.items()):
            handle.write(f"method={method} n={len(items)}\n")
            for metric in METRICS:
                vals = [row[metric] for row in items]
                lo, hi = bootstrap_ci(vals)
                handle.write(
                    f"  {metric}: mean={mean(vals)} std={std(vals)} ci95=({lo}, {hi})\n"
                )
        handle.write("\nPaired differences against baseline by shared seed\n")
        for metric in ["last_test_return", "best_train_return", "train_auc", "stability_gap"]:
            handle.write(f"metric={metric}\n")
            for method, diffs in sorted(paired_differences(rows, baseline, metric).items()):
                lo, hi = bootstrap_ci(diffs)
                handle.write(
                    f"  {method}: n={len(diffs)} mean_diff={mean(diffs)} std_diff={std(diffs)} ci95=({lo}, {hi})\n"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline", default="baseline")
    args = parser.parse_args()
    write_outputs(args.out_dir, args.baseline)


if __name__ == "__main__":
    main()
