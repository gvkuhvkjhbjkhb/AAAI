import json
import os
from collections import Counter, deque

import numpy as np

from .failure_classifier import FailureClassifier
from .trajectory_summarizer import summarize_episode_batch


class FailureTrajectoryRecorder:
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.enabled = bool(getattr(args, "llm_fd_enabled", False))
        self.output_dir = os.path.join(
            getattr(args, "local_results_path", "results"),
            "llm_fd",
            getattr(args, "unique_token", "debug_run"),
        )
        self.failure_percentile = float(getattr(args, "llm_fd_failure_percentile", 30.0))
        self.min_history = int(getattr(args, "llm_fd_min_history", 20))
        self.max_records = int(getattr(args, "llm_fd_max_records", 500))
        self.history = deque(maxlen=max(self.min_history * 5, 100))
        self.counter = Counter()
        self.records_written = 0
        self.classifier = FailureClassifier(
            mode=getattr(args, "llm_fd_classifier", "heuristic"),
            model=getattr(args, "llm_fd_model", ""),
            timeout=int(getattr(args, "llm_fd_timeout", 60)),
        )
        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)

    def process_batch(self, episode_batch, t_env, episode):
        if not self.enabled or self.records_written >= self.max_records:
            return []
        returns = self._episode_returns(episode_batch)
        if not returns:
            return []
        threshold = self._failure_threshold(returns)
        diagnoses = []
        for batch_idx, episode_return in enumerate(returns):
            self.history.append(float(episode_return))
            if episode_return > threshold or self.records_written >= self.max_records:
                continue
            single_episode = episode_batch[batch_idx : batch_idx + 1]
            summary = summarize_episode_batch(
                single_episode,
                env_name=self._env_name(),
                success_reward=getattr(self.args, "llm_fd_success_reward", None),
            )
            diagnosis = self.classifier.classify(summary)
            self.counter[diagnosis.failure_type] += 1
            record = {
                "record_id": self.records_written,
                "t_env": int(t_env),
                "episode": int(episode),
                "batch_index": int(batch_idx),
                "return": float(episode_return),
                "failure_threshold": float(threshold),
                "summary": summary,
                "diagnosis": diagnosis.to_dict(),
            }
            self._write_record(record)
            diagnoses.append((batch_idx, diagnosis))
        return diagnoses

    def log_stats(self, logger, t_env):
        if not self.enabled:
            return
        logger.log_stat("llm_fd_records", self.records_written, t_env)
        total = sum(self.counter.values())
        if total == 0:
            return
        for failure_type, count in self.counter.items():
            logger.log_stat(f"llm_fd_{failure_type}_ratio", count / total, t_env)

    def _episode_returns(self, episode_batch):
        rewards = episode_batch["reward"]
        filled = episode_batch["filled"]
        if hasattr(rewards, "detach"):
            rewards = rewards.detach().cpu().numpy()
            filled = filled.detach().cpu().numpy()
        returns = []
        for batch_idx in range(rewards.shape[0]):
            valid = filled[batch_idx, :, 0].astype(bool)
            if not valid.any():
                continue
            returns.append(float(rewards[batch_idx, valid].sum()))
        return returns

    def _failure_threshold(self, current_returns):
        if len(self.history) < self.min_history:
            values = current_returns
        else:
            values = list(self.history)
        return float(np.percentile(values, self.failure_percentile))

    def _write_record(self, record):
        path = os.path.join(self.output_dir, "failure_records.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.records_written += 1

    def _env_name(self):
        env_args = getattr(self.args, "env_args", {})
        if isinstance(env_args, dict):
            return env_args.get("key", env_args.get("map_name", getattr(self.args, "env", "unknown")))
        return getattr(self.args, "env", "unknown")
