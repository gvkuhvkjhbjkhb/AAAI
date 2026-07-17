"""Failure-Aware Potential-Based Reward Shaping (FA-PBS).

Implements Ng et al. (1999) potential-based shaping:
    F(s, s') = gamma * phi(s') - phi(s)
which provably preserves the optimal policy.

Vectorized apply() for speed: extracts all states at once, computes
features in batch with numpy, avoids per-step Python loops.
"""
import numpy as np

from .potential_features import PotentialFeatureExtractor, RandomPotentialExtractor


FAILURE_TO_FEATURE = {
    "insufficient_cooperation": "coop",
    "inefficient_exploration": "explore",
    "target_miscoordination": "target",
    "low_value_overcommitment": "target",
    "timeout_near_success": "coop",
    "unknown": "explore",
}


class FailureAwarePBRS:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "llm_fd_apply_reward_shaping", False))
        self.gamma = float(getattr(args, "gamma", 0.99))
        self.lambda_scale = float(getattr(args, "llm_fd_pbrs_lambda", 0.05))
        self.alpha = float(getattr(args, "llm_fd_pbrs_alpha", 0.01))
        self.beta = float(getattr(args, "llm_fd_pbrs_beta", 0.05))
        self.success_percentile = float(getattr(args, "llm_fd_pbrs_success_percentile", 70.0))
        self.success_min_return = float(getattr(args, "llm_fd_pbrs_success_min_return", 0.0))
        self.use_adaptive_weights = bool(getattr(args, "llm_fd_pbrs_adaptive", True))
        self.use_random_features = bool(getattr(args, "llm_fd_pbrs_random_features", False))
        self.use_random_weights = bool(getattr(args, "llm_fd_pbrs_random_weights", False))
        self.n_agents = int(getattr(args, "n_agents", 3))
        self.grid_size = int(getattr(args, "llm_fd_pbrs_grid_size", 10))

        if self.use_random_features:
            self.extractor = RandomPotentialExtractor(args)
        else:
            self.extractor = PotentialFeatureExtractor(args)

        if self.use_random_weights:
            rng = np.random.RandomState(int(getattr(args, "seed", 42)) + 100)
            if self.use_random_features:
                self.weights = {f"rand{i}": float(rng.uniform(-1, 1)) for i in range(3)}
            else:
                self.weights = {"coop": float(rng.uniform(-1, 1)), "explore": float(rng.uniform(-1, 1)), "target": float(rng.uniform(-1, 1))}
        else:
            if self.use_random_features:
                self.weights = {f"rand{i}": 1.0 for i in range(3)}
            else:
                self.weights = {"coop": 1.0, "explore": 1.0, "target": 1.0}

        self._return_history = []
        self._history_max = 200

        self.shaping_trigger_count = 0
        self.shaping_penalty_total = 0.0
        self.shaping_terminal_bonus_total = 0.0
        self.shaping_episode_steps_total = 0

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled:
            return episode_batch
        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]

        returns = self._episode_returns(rewards, filled)
        if returns:
            self._update_weights(returns, diagnoses)

        states_np = self._get_states_numpy(episode_batch)
        filled_np = self._get_filled_numpy(filled)

        for batch_idx in range(rewards.shape[0]):
            valid_mask = filled_np[batch_idx, :, 0].astype(bool)
            n = int(valid_mask.sum())
            if n == 0:
                continue
            terminal_t = n - 1
            episode_steps = n

            if self.use_random_features:
                potentials = self._compute_potentials_random_batch(states_np[batch_idx, :n])
            else:
                potentials = self._compute_potentials_structured_batch(states_np[batch_idx, :n], batch_idx)

            shaping = np.zeros(n, dtype=np.float32)
            shaping[1:] = self.gamma * potentials[1:] - potentials[:-1]
            shaping[0] = self.gamma * potentials[0]

            for t in range(n):
                rewards[batch_idx, t, 0] += float(shaping[t])

            self.shaping_trigger_count += 1
            self.shaping_episode_steps_total += episode_steps

        return episode_batch

    def _compute_potentials_structured_batch(self, states_slice, batch_idx):
        n = len(states_slice)
        n_entities = len(states_slice[0]) // 4 if len(states_slice) > 0 and len(states_slice[0]) > 0 else 0
        if n_entities == 0:
            return np.zeros(n, dtype=np.float32)

        agent_states = states_slice[:, :self.n_agents * 4].reshape(n, self.n_agents, 4)
        agent_positions = agent_states[:, :, :2].astype(np.float64)

        food_start = self.n_agents * 4
        n_food = max(1, (n_entities - self.n_agents))
        food_end = min(food_start + n_food * 4, states_slice.shape[1])
        food_states = states_slice[:, food_start:food_end].reshape(n, -1, 4) if food_end > food_start else np.zeros((n, 1, 4))
        food_positions = food_states[:, :, :2].astype(np.float64) if food_states.shape[1] > 0 else np.zeros((n, 1, 2))

        max_dist = float(self.grid_size * np.sqrt(2))

        diffs = agent_positions[:, :, None, :] - agent_positions[:, None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))
        triu_mask = np.triu(np.ones((self.n_agents, self.n_agents), dtype=bool), k=1)
        mean_dists = dists[:, triu_mask].mean(axis=1) if self.n_agents > 1 else np.zeros(n)
        coop = -mean_dists / max(max_dist, 1.0)

        for t in range(n):
            for pos in agent_positions[t]:
                self.extractor._visited.add(tuple(pos.astype(float)))
        if len(self.extractor._visited) > 100000:
            keep = list(self.extractor._visited)[-50000:]
            self.extractor._visited = set(keep)

        visited_arr = np.array([tuple(pos.astype(float)) for pos in agent_positions.reshape(-1, 2)]) if n > 0 else np.zeros((0, 2))
        novelty = np.zeros(n, dtype=np.float64)
        for t in range(n):
            count = 0
            for a in range(self.n_agents):
                tp = tuple(agent_positions[t, a])
                if tp not in self.extractor._visited:
                    count += 1
            novelty[t] = count / max(1, self.n_agents)

        target = np.zeros(n, dtype=np.float64)
        n_food_actual = food_positions.shape[1]
        for t in range(n):
            agreements = []
            for f in range(n_food_actual):
                fpos = food_positions[t, f]
                adists = np.sqrt(((agent_positions[t] - fpos) ** 2).sum(axis=1))
                if len(adists) < 2 or adists.max() < 1e-6:
                    continue
                mean_d = float(adists.mean())
                std_d = float(adists.std())
                agreements.append(1.0 - std_d / max(mean_d, 1.0))
            target[t] = float(np.mean(agreements)) if agreements else 0.0

        w_coop = self.weights.get("coop", 1.0)
        w_explore = self.weights.get("explore", 1.0)
        w_target = self.weights.get("target", 1.0)
        potentials = self.lambda_scale * (w_coop * coop + w_explore * novelty + w_target * target)
        return potentials.astype(np.float32)

    def _compute_potentials_random_batch(self, states_slice):
        state_arr = np.asarray(states_slice, dtype=np.float32)
        proj = self.extractor._proj
        bias = self.extractor._bias
        if state_arr.shape[1] != proj.shape[0]:
            return np.zeros(len(state_arr), dtype=np.float32)
        vals = np.tanh(state_arr @ proj + bias)
        w = np.array([self.weights.get(f"rand{i}", 1.0) for i in range(3)], dtype=np.float32)
        return (self.lambda_scale * (vals * w).sum(axis=1)).astype(np.float32)

    def _get_states_numpy(self, episode_batch):
        states = episode_batch["state"]
        if hasattr(states, "detach"):
            states = states.detach().cpu().numpy()
        if states.ndim == 3:
            return states
        if states.ndim == 4:
            return states[0]
        return states

    def _get_filled_numpy(self, filled):
        if hasattr(filled, "detach"):
            filled = filled.detach().cpu().numpy()
        if filled.ndim == 3:
            return filled
        if filled.ndim == 4:
            return filled[0]
        return filled

    def _episode_returns(self, rewards, filled):
        returns = []
        if hasattr(rewards, "detach"):
            rewards_np = rewards.detach().cpu().numpy()
        else:
            rewards_np = np.asarray(rewards)
        if hasattr(filled, "detach"):
            filled_np = filled.detach().cpu().numpy()
        else:
            filled_np = np.asarray(filled)
        if filled_np.ndim == 3:
            filled_np = filled_np[0] if filled_np.shape[0] == 1 else filled_np
        for batch_idx in range(rewards_np.shape[0]):
            valid_mask = filled_np[batch_idx, :, 0].astype(bool) if filled_np.ndim >= 2 else np.ones(rewards_np.shape[1], dtype=bool)
            n = int(valid_mask.sum())
            if n > 0:
                returns.append(float(rewards_np[batch_idx, :n, 0].sum()))
        return returns

    def _update_weights(self, returns, diagnoses):
        if not self.use_adaptive_weights:
            return
        self._return_history.extend(returns)
        if len(self._return_history) > self._history_max:
            self._return_history = self._return_history[-self._history_max:]
        if len(self._return_history) < 10:
            return
        success_threshold = max(
            self.success_min_return,
            float(np.percentile(self._return_history, self.success_percentile)),
        )
        for item in diagnoses:
            if isinstance(item, tuple):
                diagnosis = item[1]
            else:
                diagnosis = item
            if diagnosis is None:
                continue
            failure_type = getattr(diagnosis, "failure_type", "unknown")
            feature_name = FAILURE_TO_FEATURE.get(failure_type, "explore")
            if feature_name in self.weights:
                self.weights[feature_name] += self.alpha
        for r in returns:
            if r >= success_threshold:
                for k in self.weights:
                    self.weights[k] *= (1.0 - self.beta)

    def accounting(self):
        avg_penalty = self.shaping_penalty_total / max(1, self.shaping_trigger_count)
        avg_steps = self.shaping_episode_steps_total / max(1, self.shaping_trigger_count)
        return {
            "triggers": self.shaping_trigger_count,
            "penalty_total": self.shaping_penalty_total,
            "terminal_bonus_total": self.shaping_terminal_bonus_total,
            "episode_steps_total": self.shaping_episode_steps_total,
            "avg_penalty_per_trigger": avg_penalty,
            "avg_steps_per_trigger": avg_steps,
            "weights": dict(self.weights),
        }

    def log_stats(self, logger, t_env):
        stats = self.accounting()
        logger.log_stat("llm_fd_shaping_triggers", stats["triggers"], t_env)
        logger.log_stat("llm_fd_shaping_penalty_total", stats["penalty_total"], t_env)
        logger.log_stat("llm_fd_shaping_terminal_bonus_total", stats["terminal_bonus_total"], t_env)
        logger.log_stat("llm_fd_shaping_episode_steps_total", stats["episode_steps_total"], t_env)
        logger.log_stat("llm_fd_shaping_avg_penalty_per_trigger", stats["avg_penalty_per_trigger"], t_env)
        logger.log_stat("llm_fd_shaping_avg_steps_per_trigger", stats["avg_steps_per_trigger"], t_env)
        for name, w in stats["weights"].items():
            logger.log_stat(f"llm_fd_pbrs_w_{name}", w, t_env)
