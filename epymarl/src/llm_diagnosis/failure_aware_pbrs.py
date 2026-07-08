"""Failure-Aware Potential-Based Reward Shaping (FA-PBS).

Implements Ng et al. (1999) potential-based shaping:
    F(s, s') = gamma * phi(s') - phi(s)
which provably preserves the optimal policy.

Failure diagnosis adaptively updates feature weights:
    failure type k -> w_k += alpha (emphasize missing dimension)
    success episode -> w_k *= (1-beta) (decay, current policy sufficient)
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
    """Potential-based reward shaping with failure-driven weight adaptation."""

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

        if self.use_random_features:
            self.extractor = RandomPotentialExtractor(args)
        else:
            self.extractor = PotentialFeatureExtractor(args)

        if self.use_random_weights:
            rng = np.random.RandomState(int(getattr(args, "seed", 42)) + 100)
            self.weights = {"coop": float(rng.uniform(-1, 1)), "explore": float(rng.uniform(-1, 1)), "target": float(rng.uniform(-1, 1))}
            if self.use_random_features:
                self.weights = {f"rand{i}": float(rng.uniform(-1, 1)) for i in range(3)}
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
        self._weight_history = []

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled:
            return episode_batch
        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]
        returns = self._episode_returns(rewards, filled)

        if returns:
            self._update_weights(returns, diagnoses)
            self.extractor.reset_novelty()

        for batch_idx in range(rewards.shape[0]):
            valid = filled[batch_idx, :, 0].nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            terminal_t = int(valid[-1].item())
            episode_steps = terminal_t + 1
            prev_phi = None
            for t in range(terminal_t + 1):
                self.extractor.update_visit_count(episode_batch, batch_idx, t)
                curr_phi = self._compute_potential(episode_batch, batch_idx, t)
                if prev_phi is not None:
                    shaping = self.gamma * curr_phi - prev_phi
                    rewards[batch_idx, t, 0] += float(shaping)
                prev_phi = curr_phi
            self.shaping_trigger_count += 1
            self.shaping_episode_steps_total += episode_steps
            self.shaping_penalty_total += 0.0
            self.shaping_terminal_bonus_total += 0.0
        return episode_batch

    def _compute_potential(self, episode_batch, batch_idx, t):
        features = self.extractor.extract(episode_batch, batch_idx, t)
        phi = 0.0
        for name, val in features.items():
            w = self.weights.get(name, 0.0)
            phi += w * val
        return self.lambda_scale * phi

    def _episode_returns(self, rewards, filled):
        returns = []
        for batch_idx in range(rewards.shape[0]):
            valid_mask = filled[batch_idx, :, 0].bool()
            n = int(valid_mask.sum().item())
            if n == 0:
                returns.append(None)
            else:
                returns.append(float(rewards[batch_idx, :n, 0].sum().item()))
        return [r for r in returns if r is not None]

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
        self._weight_history.append(dict(self.weights))
        if len(self._weight_history) > 1000:
            self._weight_history = self._weight_history[-500:]

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
