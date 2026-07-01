import argparse
import csv
import glob
import json
import os
import random
from collections import Counter

from .failure_classifier import FailureClassifier, FailureDiagnosis
from .prompts import build_failure_prompt


class HFClassifier:
    def __init__(self, model_name, max_new_tokens=160):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def classify(self, summary):
        import json
        import torch

        prompt = build_failure_prompt(summary)
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return FailureDiagnosis("unknown", 0.0, raw.strip()[:300], "hf_parse_error")
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return FailureDiagnosis("unknown", 0.0, raw.strip()[:300], "hf_parse_error")
        failure_type = payload.get("failure_type", "unknown")
        valid = {
            "target_miscoordination",
            "insufficient_cooperation",
            "inefficient_exploration",
            "low_value_overcommitment",
            "timeout_near_success",
            "unknown",
        }
        if failure_type not in valid:
            failure_type = "unknown"
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return FailureDiagnosis(failure_type, confidence, str(payload.get("evidence", ""))[:500], "hf")


def enhanced_heuristic(summary):
    text = summary.lower()
    load_counts = []
    if "load action counts by agent:" in text:
        raw = text.split("load action counts by agent:", 1)[1].split("\n", 1)[0].strip()
        raw = raw.strip("[]")
        for item in raw.split(","):
            item = item.strip()
            if item:
                try:
                    load_counts.append(int(item))
                except ValueError:
                    pass
    if load_counts:
        active_loaders = sum(1 for value in load_counts if value > 0)
        if active_loaders <= 1:
            return FailureDiagnosis(
                "insufficient_cooperation",
                0.68,
                "Only one or no agents executed load actions, indicating weak cooperative collection.",
                "enhanced_heuristic",
            )
        if max(load_counts) >= 3 * max(1, min(value for value in load_counts if value > 0)):
            return FailureDiagnosis(
                "target_miscoordination",
                0.57,
                "Load attempts are highly imbalanced across agents, suggesting incompatible target choices.",
                "enhanced_heuristic",
            )
    if "episode length: 50" in text and "positive reward steps: 0" not in text:
        return FailureDiagnosis(
            "timeout_near_success",
            0.55,
            "The episode reached the time limit despite some positive reward events, suggesting slow completion.",
            "enhanced_heuristic",
        )
    if "positive reward steps: 0" in text:
        return FailureDiagnosis(
            "inefficient_exploration",
            0.62,
            "No positive reward events were observed, indicating ineffective exploration or failure to reach food.",
            "enhanced_heuristic",
        )
    return FailureDiagnosis(
        "inefficient_exploration",
        0.45,
        "The summary lacks stronger evidence for a more specific coordination failure.",
        "enhanced_heuristic",
    )


def iter_records(paths):
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                payload = json.loads(line)
                payload["source_file"] = path
                payload["source_line"] = line_idx + 1
                yield payload


def load_human_labels(path):
    if not path or not os.path.exists(path):
        return {}
    labels = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("source_file"), int(row.get("source_line", 0)))
            label = row.get("human_label", "").strip()
            if label:
                labels[key] = label
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", default="results/llm_fd/*/failure_records.jsonl")
    parser.add_argument("--output-dir", default="results/offline_relabel")
    parser.add_argument("--mode", default="enhanced_heuristic", choices=["enhanced_heuristic", "ollama", "hf", "mock", "heuristic"])
    parser.add_argument("--model", default="")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--human-labels", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    paths = sorted(glob.glob(args.input_glob))
    records = list(iter_records(paths))
    if args.sample_size > 0 and len(records) > args.sample_size:
        records = random.sample(records, args.sample_size)
    os.makedirs(args.output_dir, exist_ok=True)

    classifier = None
    if args.mode == "hf":
        classifier = HFClassifier(args.model or "Qwen/Qwen2.5-0.5B-Instruct")
    elif args.mode in {"ollama", "mock", "heuristic"}:
        classifier = FailureClassifier(mode=args.mode, model=args.model)

    human_labels = load_human_labels(args.human_labels)
    original_counts = Counter()
    relabel_counts = Counter()
    agreement = 0
    human_total = 0
    human_correct = 0

    out_jsonl = os.path.join(args.output_dir, "relabels.jsonl")
    audit_csv = os.path.join(args.output_dir, "audit_sample.csv")
    with open(out_jsonl, "w", encoding="utf-8") as out, open(audit_csv, "w", newline="", encoding="utf-8") as audit:
        fieldnames = [
            "source_file",
            "source_line",
            "original_label",
            "new_label",
            "confidence",
            "human_label",
            "evidence",
            "summary",
        ]
        writer = csv.DictWriter(audit, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            summary = record.get("summary", "")
            original = record.get("diagnosis", {}).get("failure_type", "unknown")
            if args.mode == "enhanced_heuristic":
                diagnosis = enhanced_heuristic(summary)
            else:
                if classifier is None:
                    raise RuntimeError("Classifier was not initialized for the selected relabel mode.")
                diagnosis = classifier.classify(summary)
            key = (record["source_file"], record["source_line"])
            human_label = human_labels.get(key, "")
            original_counts[original] += 1
            relabel_counts[diagnosis.failure_type] += 1
            agreement += int(original == diagnosis.failure_type)
            if human_label:
                human_total += 1
                human_correct += int(human_label == diagnosis.failure_type)
            payload = {
                "source_file": record["source_file"],
                "source_line": record["source_line"],
                "return": record.get("return"),
                "original_diagnosis": record.get("diagnosis", {}),
                "new_diagnosis": diagnosis.to_dict(),
                "summary": summary,
                "human_label": human_label,
            }
            out.write(json.dumps(payload, ensure_ascii=True) + "\n")
            writer.writerow(
                {
                    "source_file": record["source_file"],
                    "source_line": record["source_line"],
                    "original_label": original,
                    "new_label": diagnosis.failure_type,
                    "confidence": diagnosis.confidence,
                    "human_label": human_label,
                    "evidence": diagnosis.evidence,
                    "summary": summary.replace("\n", " | "),
                }
            )

    total = len(records)
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("Offline failure relabel summary\n")
        handle.write(f"mode: {args.mode}\n")
        handle.write(f"model: {args.model}\n")
        handle.write(f"records: {total}\n")
        handle.write(f"original_counts: {dict(original_counts)}\n")
        handle.write(f"relabel_counts: {dict(relabel_counts)}\n")
        handle.write(f"original_relabel_agreement: {agreement / total if total else 0.0:.4f}\n")
        if human_total:
            handle.write(f"human_accuracy: {human_correct / human_total:.4f}\n")
            handle.write(f"human_labeled_records: {human_total}\n")
        else:
            handle.write("human_accuracy: NA (no human labels supplied)\n")
            handle.write("human_labeled_records: 0\n")


if __name__ == "__main__":
    main()
