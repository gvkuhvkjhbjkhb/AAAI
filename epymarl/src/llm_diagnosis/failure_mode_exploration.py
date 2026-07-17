"""Failure-Mode-aware Exploration (FME).

Instead of reward shaping (PBRS), FME adds a targeted intrinsic exploration
bonus to episodes diagnosed as failures. The bonus type is conditioned on
the failure diagnosis:
  - inefficient_exploration  -> novelty-based bonus (reward visiting new states)
  - insufficient_cooperation -> proximity bonus (reward agents staying close)
  - target_miscoordination   -> target-alignment bonus (reward converging on same food)
  - other                    -> small uniform exploration bonus

The bonus magnitude is modulated by a divergence signal: if recent shaping
correlation with return is negative, increase exploration; if positive,
decrease it. This is distinct from DG-PBS: FME does NOT shape the reward
signal — it injects exploration bonuses only on failed episodes.

Addresses ARMS (2026) oscillatory failure by triggering targeted exploration
rather than undirected exploration.
"""
from collections import deque

import numpy as np

from .failure_aware_pbrs import FailureAwarePBRS


FAILURE_TO_BONUS = {
    "insufficient_cooperation": "proximity",
    "inefficient_exploration": "novelty",
    "target_miscoordination": "target",
    "low_value_overcommitment": "target",
    "timeout_near_success": "proximity",
    "unknown": "novelty",
}


class FailureModeExploration(FailureAwarePBRS):
    def __init__(self, args):
        super().__init__(args)
        self.fme_enabled = True
        self.fme_bonus_scale = float(getattr(args, "llm_fd_fme_bonus_scale", 0.01))
        self.fme_divergence_window = int(getattr(args, "llm_fd_fme_window", 50))
        self.fme_warmup = int(getattr(args, "llm_fd_fme_warmup", 20))
        self.fme_smoothing = float(getattr(args, "llm_fd_fme_smoothing", 0.8))
        self.fme_max_scale = float(getattr(args, "llm_fd_fme_max_scale", 3.0))
        self.fme_min_scale = float(getattr(args, "llm_fd_fme_min_scale", 0.5))
        self.fme_failure_percentile = float(getattr(args, "llm_fd_fme_failure_percentile", 30.0))
        self.fme_min_history = int(getattr(args, "llm_fd_fme_min_history", 20))

        self._return_history_fme = deque(maxlen=self.fme_divergence_window)
        self._bonus_history = deque(maxlen=self.fme_divergence_window)
        self._current_scale = 1.0
        self._scale_history = []
        self._visited = set()
        self._bonus_trigger_count = 0
        self._bonus_total = 0.0

    def _compute_exploration_scale(self):
        if len(self._return_history_fme) < self.fme_warmup:
            return 1.0
        rets = np.array(self._return_history_fme)
        bons = np.array(self._bonus_history)
        if rets.std() < 1e-8 or bons.std() < 1e-8:
            return self._current_scale
        corr = float(np.corrcoef(bons, rets)[0, 1])
        corr = np.clip(corr, -1.0, 1.0)
        if corr >= 0:
            normalized = 1.0 / (1.0 + np.exp(-5.0 * corr))
            raw_scale = 1.0 - 0.5 * normalized * (1.0 - self.fme_min_scale)
        else:
            normalized = 1.0 / (1.0 + np.exp(5.0 * corr))
            raw_scale = 1.0 + (self.fme_max_scale - 1.0) * normalized
        raw_scale = float(np.clip(raw_scale, self.fme_min_scale, self.fme_max_scale))
        return self.fme_smoothing * self._current_scale + (1.0 - self.fme_smoothing) * raw_scale

    def _failure_threshold(self, returns):
        if len(self._return_history_fme) < self.fme_min_history:
            values = returns
        else:
            values = list(self._return_history_fme)
        return float(np.percentile(values, self.fme_failure_percentile))

    def _compute_novelty_bonus(self, states_slice, n, n_agents):
        agent_positions = states_slice[:, :n_agents * 2].reshape(n, n_agents, 2).astype(np.float64)
        bonuses = np.zeros(n, dtype=np.float32)
        for t in range(n):
            count = 0
            for a in range(n_agents):
                tp = tuple(agent_positions[t, a])
                if tp not in self._visited:
                    count += 1
            for a in range(n_agents):
                self._visited.add(tuple(agent_positions[t, a]))
            bonuses[t] = count / max(1, n_agents)
        if len(self._visited) > 100000:
            keep = list(self._visited)[-50000:]
            self._visited = set(keep)
        return bonuses

    def _compute_proximity_bonus(self, states_slice, n, n_agents, grid_size=10):
        agent_positions = states_slice[:, :n_agents * 2].reshape(n, n_agents, 2).astype(np.float64)
        max_dist = float(grid_size * np.sqrt(2))
        bonuses = np.zeros(n, dtype=np.float32)
        for t in range(n):
            diffs = agent_positions[t, :, None, :] - agent_positions[t, None, :, :]
            dists = np.sqrt((diffs ** 2).sum(axis=-1))
            triu_mask = np.triu(np.ones((n_agents, n_agents), dtype=bool), k=1)
            mean_dist = dists[triu_mask].mean() if n_agents > 1 else 0.0
            bonuses[t] = 1.0 - mean_dist / max(max_dist, 1.0)
        return bonuses

    def _compute_target_bonus(self, states_slice, n, n_agents):
        n_entities = len(states_slice[0]) // 2 if len(states_slice) > 0 else 0
        agent_positions = states_slice[:, :n_agents * 2].reshape(n, n_agents, 2).astype(np.float64)
        food_start = n_agents * 2
        n_food = max(1, n_entities - n_agents)
        food_end = min(food_start + n_food * 2, states_slice.shape[1])
        food_positions = states_slice[:, food_start:food_end].reshape(n, -1, 2).astype(np.float64) if food_end > food_start else np.zeros((n, 1, 2))
        bonuses = np.zeros(n, dtype=np.float32)
        for t in range(n):
            agreements = []
            for f in range(food_positions.shape[1]):
                fpos = food_positions[t, f]
                adists = np.sqrt(((agent_positions[t] - fpos) ** 2).sum(axis=1))
                if len(adists) < 2 or adists.max() < 1e-6:
                    continue
                mean_d = float(adists.mean())
                std_d = float(adists.std())
                agreements.append(1.0 - std_d / max(mean_d, 1.0))
            bonuses[t] = float(np.mean(agreements)) if agreements else 0.0
        return bonuses

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled:
            return episode_batch

        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]
        returns = self._episode_returns(rewards, filled)
        if returns:
            self._update_weights(returns, diagnoses)

        threshold = self._failure_threshold(returns)
        states_np = self._get_states_numpy(episode_batch)
        filled_np = self._get_filled_numpy(filled)

        batch_bonuses = []
        batch_returns = []
        effective_bonus = self.fme_bonus_scale * self._current_scale

        for batch_idx in range(rewards.shape[0]):
            valid_mask = filled_np[batch_idx, :, 0].astype(bool)
            n = int(valid_mask.sum())
            if n == 0:
                continue
            ret = float(returns[batch_idx]) if batch_idx < len(returns) else 0.0
            is_failure = ret <= threshold
            batch_returns.append(ret)
            if not is_failure:
                batch_bonuses.append(0.0)
                continue

            diag = None
            if batch_idx < len(diagnoses):
                d = diagnoses[batch_idx]
                diag = d[1] if isinstance(d, tuple) else d
            failure_type = getattr(diag, "failure_type", "unknown") if diag else "unknown"
            bonus_type = FAILURE_TO_BONUS.get(failure_type, "novelty")

            states_slice = states_np[batch_idx, :n]
            if bonus_type == "novelty":
                bonus = self._compute_novelty_bonus(states_slice, n, self.n_agents)
            elif bonus_type == "proximity":
                bonus = self._compute_proximity_bonus(states_slice, n, self.n_agents, self.grid_size)
            elif bonus_type == "target":
                bonus = self._compute_target_bonus(states_slice, n, self.n_agents)
            else:
                bonus = np.ones(n, dtype=np.float32) * 0.1

            scaled_bonus = bonus * effective_bonus
            for t in range(n):
                rewards[batch_idx, t, 0] += float(scaled_bonus[t])

            mag = float(np.abs(scaled_bonus).mean())
            batch_bonuses.append(mag)
            self._bonus_trigger_count += 1
            self._bonus_total += float(scaled_bonus.sum())
            self.shaping_trigger_count += 1
            self.shaping_episode_steps_total += n

        for mag, ret in zip(batch_bonuses, batch_returns):
            self._bonus_history.append(mag)
            self._return_history_fme.append(ret)

        self._current_scale = self._compute_exploration_scale()
        self._scale_history.append((t_env, self._current_scale, effective_bonus))

        return episode_batch

    def accounting(self):
        stats = super().accounting()
        stats["fme_current_scale"] = self._current_scale
        stats["fme_effective_bonus"] = self.fme_bonus_scale * self._current_scale
        stats["fme_trigger_count"] = self._bonus_trigger_count
        stats["fme_bonus_total"] = self._bonus_total
        if self._scale_history:
            recent = self._scale_history[-20:]
            stats["fme_scale_mean_recent"] = float(np.mean([s[1] for s in recent]))
        return stats

    def log_stats(self, logger, t_env):
        super().log_stats(logger, t_env)
        logger.log_stat("llm_fd_fme_scale", self._current_scale, t_env)
        logger.log_stat("llm_fd_fme_effective_bonus", self.fme_bonus_scale * self._current_scale, t_env)
        logger.log_stat("llm_fd_fme_triggers", self._bonus_trigger_count, t_env)
