import argparse
import csv
import json
from collections import Counter, defaultdict


LABELS = [
    "target_miscoordination",
    "insufficient_cooperation",
    "inefficient_exploration",
    "low_value_overcommitment",
    "timeout_near_success",
    "unknown",
]


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_sample(rows):
    return {row.get("sample_id"): row for row in rows if row.get("sample_id")}


def cohen_kappa(pairs):
    pairs = [(a, b) for a, b in pairs if a and b]
    total = len(pairs)
    if total == 0:
        return None
    observed = sum(1 for a, b in pairs if a == b) / total
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[label] / total) * (right[label] / total) for label in LABELS)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def raw_agreement(pairs):
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def per_class_f1(gold, pred):
    out = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for g in gold if g == label)}
    return out


def confusion(gold, pred):
    matrix = defaultdict(Counter)
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return {g: dict(matrix[g]) for g in LABELS if matrix[g]}


def metrics(gold, pred):
    pairs = list(zip(gold, pred))
    f1 = per_class_f1(gold, pred)
    return {
        "n": len(gold),
        "accuracy": raw_agreement(pairs),
        "cohen_kappa": cohen_kappa(pairs),
        "macro_f1": sum(v["f1"] for v in f1.values()) / len(LABELS),
        "per_class_f1": f1,
        "gold_counts": dict(Counter(gold)),
        "pred_counts": dict(Counter(pred)),
        "confusion": confusion(gold, pred),
    }


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator-a", required=True)
    parser.add_argument("--annotator-b", required=True)
    parser.add_argument("--adjudicated", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = by_sample(load_rows(args.annotator_a))
    b = by_sample(load_rows(args.annotator_b))
    common = sorted(set(a) & set(b))
    human_pairs = [(a[s].get("human_label"), b[s].get("human_label")) for s in common]

    gold_rows = by_sample(load_rows(args.adjudicated))
    gold_ids = sorted(s for s, row in gold_rows.items() if row.get("human_label"))
    gold = [gold_rows[s]["human_label"] for s in gold_ids]

    report = {
        "human_agreement": {
            "n": len(common),
            "raw_agreement": raw_agreement(human_pairs),
            "cohen_kappa": cohen_kappa(human_pairs),
            "annotator_a_counts": dict(Counter(a[s].get("human_label") for s in common)),
            "annotator_b_counts": dict(Counter(b[s].get("human_label") for s in common)),
        },
        "automatic": {},
    }

    for name, path in zip(args.names, args.predictions):
        pred_rows = by_sample(load_rows(path))
        pred = [pred_rows.get(s, {}).get("new_label", "unknown") for s in gold_ids]
        report["automatic"][name] = metrics(gold, pred)

    with open(args.output + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with open(args.output + ".md", "w", encoding="utf-8") as handle:
        ha = report["human_agreement"]
        handle.write("# Phase 2A/2B Diagnosis Validation\n\n")
        handle.write(
            f"Human-human n={ha['n']} raw_agreement={fmt(ha['raw_agreement'])} "
            f"cohen_kappa={fmt(ha['cohen_kappa'])}\n\n"
        )
        handle.write("| classifier | n | accuracy | macro-F1 | kappa |\n|---|---:|---:|---:|---:|\n")
        for name, m in report["automatic"].items():
            handle.write(f"| {name} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['macro_f1'])} | {fmt(m['cohen_kappa'])} |\n")
        handle.write("\n## Counts and Confusions\n\n")
        for name, m in report["automatic"].items():
            handle.write(f"### {name}\n\n")
            handle.write(f"gold_counts={m['gold_counts']}\n\n")
            handle.write(f"pred_counts={m['pred_counts']}\n\n")
            handle.write(f"confusion={m['confusion']}\n\n")
            handle.write(f"per_class_f1={m['per_class_f1']}\n\n")


if __name__ == "__main__":
    main()
