import argparse
import csv
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


def macro_f1(rows, label_field="human_label", pred_field="new_label"):
    scores = []
    for label in LABELS:
        tp = sum(1 for row in rows if row.get(label_field) == label and row.get(pred_field) == label)
        fp = sum(1 for row in rows if row.get(label_field) != label and row.get(pred_field) == label)
        fn = sum(1 for row in rows if row.get(label_field) == label and row.get(pred_field) != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def accuracy(rows, label_field="human_label", pred_field="new_label"):
    valid = [row for row in rows if row.get(label_field)]
    if not valid:
        return "NA"
    return sum(1 for row in valid if row.get(label_field) == row.get(pred_field)) / len(valid)


def confusion(rows, label_field="human_label", pred_field="new_label"):
    matrix = defaultdict(Counter)
    for row in rows:
        gold = row.get(label_field, "")
        if gold:
            matrix[gold][row.get(pred_field, "unknown")] += 1
    return matrix


def agreement(rows_a, rows_b):
    total = min(len(rows_a), len(rows_b))
    if total == 0:
        return 0.0
    matches = 0
    for left, right in zip(rows_a[:total], rows_b[:total]):
        matches += int(left.get("new_label") == right.get("new_label"))
    return matches / total


def cohen_kappa(rows, label_field="human_label", pred_field="new_label"):
    valid = [row for row in rows if row.get(label_field)]
    total = len(valid)
    if total == 0:
        return "NA"
    observed = sum(1 for row in valid if row.get(label_field) == row.get(pred_field)) / total
    gold_counts = Counter(row.get(label_field) for row in valid)
    pred_counts = Counter(row.get(pred_field, "unknown") for row in valid)
    expected = sum((gold_counts[label] / total) * (pred_counts[label] / total) for label in LABELS)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    datasets = [(name, load_rows(path)) for name, path in zip(args.names, args.predictions)]
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("Diagnosis quality summary\n\n")
        for name, rows in datasets:
            labeled = [row for row in rows if row.get("human_label")]
            handle.write(f"{name}: records={len(rows)} human_labeled={len(labeled)}\n")
            handle.write(f"  predicted_counts={dict(Counter(row.get('new_label', 'unknown') for row in rows))}\n")
            handle.write(f"  human_accuracy={accuracy(rows)}\n")
            handle.write(f"  cohen_kappa={cohen_kappa(labeled)}\n")
            if labeled:
                handle.write(f"  macro_f1={macro_f1(labeled)}\n")
                handle.write(f"  confusion={dict((gold, dict(counts)) for gold, counts in confusion(labeled).items())}\n")
        if len(datasets) >= 2:
            handle.write("\nPairwise agreement\n")
            for i, (name_a, rows_a) in enumerate(datasets):
                for name_b, rows_b in datasets[i + 1 :]:
                    handle.write(f"  {name_a} vs {name_b}: {agreement(rows_a, rows_b):.4f}\n")


if __name__ == "__main__":
    main()
