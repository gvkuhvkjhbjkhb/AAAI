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
