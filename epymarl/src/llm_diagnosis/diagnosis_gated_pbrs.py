"""Diagnosis-Gated Potential-Based Reward Shaping (DG-PBS).

Extends FA-PBS with a divergence-gating mechanism that diagnoses whether
the shaping signal is aligned with learning progress. When the correlation
between shaping magnitude and episode return turns negative (shaping hurts),
the gate automatically reduces the effective lambda toward zero. When
positive (shaping helps), the gate restores full lambda.

This directly addresses the oscillatory failure mode identified by ARMS
(Abboud & Gal, 2026) and the reward-global misalignment barrier noted by
Akella (2025): instead of always applying shaping and hoping it helps, we
diagnose the shaping-return relationship and gate accordingly.
"""
from collections import deque

import numpy as np

from .failure_aware_pbrs import FailureAwarePBRS


class DiagnosisGatedPBRS(FailureAwarePBRS):
    def __init__(self, args):
        super().__init__(args)
        self.use_diagnosis_gate = True
        self.gate_window = int(getattr(args, "llm_fd_dg_window", 50))
        self.gate_warmup = int(getattr(args, "llm_fd_dg_warmup", 20))
        self.gate_smoothing = float(getattr(args, "llm_fd_dg_smoothing", 0.8))
        self.gate_temperature = float(getattr(args, "llm_fd_dg_temperature", 5.0))

        self._shaping_mag_history = deque(maxlen=self.gate_window)
        self._return_history_dg = deque(maxlen=self.gate_window)
        self._current_gate = 1.0
        self._gate_history = []

    def _compute_gate_factor(self):
        if len(self._shaping_mag_history) < self.gate_warmup:
            return 1.0
        mags = np.array(self._shaping_mag_history)
        rets = np.array(self._return_history_dg)
        if mags.std() < 1e-8 or rets.std() < 1e-8:
            return self._current_gate
        m_mean, m_std = mags.mean(), mags.std()
        r_mean, r_std = rets.mean(), rets.std()
        if m_std < 1e-8 or r_std < 1e-8:
            return self._current_gate
        corr = float(np.mean((mags - m_mean) * (rets - r_mean)) / (m_std * r_std))
        corr = np.clip(corr, -1.0, 1.0)
        raw_gate = 1.0 / (1.0 + np.exp(-self.gate_temperature * corr))
        raw_gate = float(np.clip(raw_gate, 0.05, 1.0))
        smoothed = self.gate_smoothing * self._current_gate + (1.0 - self.gate_smoothing) * raw_gate
        return smoothed

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled:
            return episode_batch

        effective_lambda = self.lambda_scale * self._current_gate

        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]

        returns = self._episode_returns(rewards, filled)
        if returns:
            self._update_weights(returns, diagnoses)

        states_np = self._get_states_numpy(episode_batch)
        filled_np = self._get_filled_numpy(filled)

        batch_shaping_mags = []
        batch_returns = []

        for batch_idx in range(rewards.shape[0]):
            valid_mask = filled_np[batch_idx, :, 0].astype(bool)
            n = int(valid_mask.sum())
            if n == 0:
                continue
            episode_steps = n

            if self.use_random_features:
                potentials = self._compute_potentials_random_batch(states_np[batch_idx, :n])
            else:
                potentials = self._compute_potentials_structured_batch(states_np[batch_idx, :n], batch_idx)

            shaping = np.zeros(n, dtype=np.float32)
            shaping[1:] = self.gamma * potentials[1:] - potentials[:-1]
            shaping[0] = self.gamma * potentials[0]

            shaped_values = shaping * (effective_lambda / max(self.lambda_scale, 1e-8)) if self.lambda_scale > 1e-8 else shaping

            for t in range(n):
                rewards[batch_idx, t, 0] += float(shaped_values[t])

            mag = float(np.abs(shaped_values).mean())
            ret = float(returns[batch_idx]) if batch_idx < len(returns) else 0.0
            batch_shaping_mags.append(mag)
            batch_returns.append(ret)

            self.shaping_trigger_count += 1
            self.shaping_episode_steps_total += episode_steps

        for mag, ret in zip(batch_shaping_mags, batch_returns):
            self._shaping_mag_history.append(mag)
            self._return_history_dg.append(ret)

        self._current_gate = self._compute_gate_factor()
        self._gate_history.append((t_env, self._current_gate, effective_lambda))

        return episode_batch

    def accounting(self):
        stats = super().accounting()
        stats["current_gate"] = self._current_gate
        stats["effective_lambda"] = self.lambda_scale * self._current_gate
        if self._gate_history:
            recent = self._gate_history[-20:]
            stats["gate_mean_recent"] = float(np.mean([g[1] for g in recent]))
            stats["gate_min_recent"] = float(np.min([g[1] for g in recent]))
        return stats

    def log_stats(self, logger, t_env):
        super().log_stats(logger, t_env)
        logger.log_stat("llm_fd_dg_gate", self._current_gate, t_env)
        logger.log_stat("llm_fd_dg_effective_lambda", self.lambda_scale * self._current_gate, t_env)
