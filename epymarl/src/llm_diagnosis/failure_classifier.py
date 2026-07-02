import json
import os
import subprocess
from dataclasses import dataclass

from .prompts import FAILURE_TYPES, build_failure_prompt


@dataclass
class FailureDiagnosis:
    failure_type: str
    confidence: float
    evidence: str
    source: str

    def to_dict(self):
        return {
            "failure_type": self.failure_type,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source": self.source,
        }


class FailureClassifier:
    def __init__(self, mode="heuristic", model="", timeout=60):
        self.mode = mode
        self.model = model
        self.timeout = timeout

    def classify(self, summary):
        if self.mode == "ollama":
            return self._classify_with_ollama(summary)
        if self.mode == "mock":
            return FailureDiagnosis("unknown", 0.0, "Mock classifier used for pipeline debugging.", "mock")
        if self.mode == "enhanced_heuristic":
            return self._classify_with_enhanced_heuristics(summary)
        return self._classify_with_heuristics(summary)

    def _classify_with_heuristics(self, summary):
        lowered = summary.lower()
        if "load action counts" in lowered and "[0" in lowered:
            return FailureDiagnosis(
                "insufficient_cooperation",
                0.55,
                "At least one agent appears not to execute load actions in the failed episode summary.",
                "heuristic",
            )
        if "zero reward steps" in lowered:
            return FailureDiagnosis(
                "inefficient_exploration",
                0.50,
                "The episode contains many zero-reward steps, suggesting ineffective exploration or delayed coordination.",
                "heuristic",
            )
        return FailureDiagnosis("unknown", 0.30, "No reliable heuristic matched the summary.", "heuristic")

    def _classify_with_enhanced_heuristics(self, summary):
        lowered = summary.lower()
        load_counts = self._extract_load_counts(lowered)
        if load_counts:
            active_loaders = sum(1 for value in load_counts if value > 0)
            if active_loaders <= 1:
                return FailureDiagnosis(
                    "insufficient_cooperation",
                    0.68,
                    "Only one or no agents executed load actions, indicating weak cooperative collection.",
                    "enhanced_heuristic",
                )
            nonzero = [value for value in load_counts if value > 0]
            if nonzero and max(nonzero) >= 3 * max(1, min(nonzero)):
                return FailureDiagnosis(
                    "target_miscoordination",
                    0.57,
                    "Load attempts are highly imbalanced across agents, suggesting incompatible target choices.",
                    "enhanced_heuristic",
                )
        if "episode length: 50" in lowered and "positive reward steps: 0" not in lowered:
            return FailureDiagnosis(
                "timeout_near_success",
                0.55,
                "The episode reached the time limit despite some positive reward events, suggesting slow completion.",
                "enhanced_heuristic",
            )
        if "positive reward steps: 0" in lowered or "zero reward steps" in lowered:
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

    def _extract_load_counts(self, lowered_summary):
        if "load action counts by agent:" not in lowered_summary:
            return []
        raw = lowered_summary.split("load action counts by agent:", 1)[1].split("\n", 1)[0].strip()
        raw = raw.strip("[]")
        counts = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                counts.append(int(item))
            except ValueError:
                pass
        return counts

    def _classify_with_ollama(self, summary):
        if not self.model:
            raise ValueError("llm_fd_model must be set when llm_fd_classifier=ollama")
        prompt = build_failure_prompt(summary)
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            return FailureDiagnosis(
                "unknown",
                0.0,
                f"Ollama failed: {result.stderr.strip()[:300]}",
                "ollama_error",
            )
        return self._parse_json(result.stdout)

    def _parse_json(self, raw_text):
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return FailureDiagnosis("unknown", 0.0, raw_text.strip()[:300], "parse_error")
        try:
            payload = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return FailureDiagnosis("unknown", 0.0, raw_text.strip()[:300], "parse_error")
        failure_type = payload.get("failure_type", "unknown")
        if failure_type not in FAILURE_TYPES:
            failure_type = "unknown"
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        evidence = str(payload.get("evidence", ""))[:500]
        return FailureDiagnosis(failure_type, confidence, evidence, "ollama")
