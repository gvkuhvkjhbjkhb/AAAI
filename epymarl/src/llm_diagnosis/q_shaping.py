"""Failure-Aware Q-Shaping: adds state-dependent offset to critic targets.

For PPO, Q-shaping is implemented by adding a potential-based offset
to the critic's target value during training. This is mathematically
equivalent to initializing Q(s,a) with a heuristic prior.
"""
import numpy as np
import torch as th

FAILURE_TO_FEATURE = {
    "insufficient_cooperation": "coop",
    "inefficient_exploration": "explore",
    "target_miscoordination": "target",
    "low_value_overcommitment": "target",
    "timeout_near_success": "coop",
    "unknown": "explore",
}

class QShaper:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "qs_enabled", False))
        self.lambda_scale = float(getattr(args, "qs_lambda", 0.02))
        self.alpha = float(getattr(args, "qs_alpha", 0.01))
        self.beta = float(getattr(args, "qs_beta", 0.05))
        self.n_agents = int(getattr(args, "n_agents", 3))
        self.grid_size = int(getattr(args, "qs_grid_size", 10))
        self.use_adaptive = bool(getattr(args, "qs_adaptive", True))
        self.use_random_features = bool(getattr(args, "qs_random_features", False))
        self.use_random_weights = bool(getattr(args, "qs_random_weights", False))
        self.success_percentile = float(getattr(args, "qs_success_percentile", 70.0))
        self.success_min_return = float(getattr(args, "qs_success_min_return", 0.0))
        if self.use_random_weights:
            rng = np.random.RandomState(int(getattr(args, "seed", 42)) + 200)
            self.weights = {"coop": float(rng.uniform(-1, 1)), "explore": float(rng.uniform(-1, 1)), "target": float(rng.uniform(-1, 1))}
        else:
            self.weights = {"coop": 1.0, "explore": 1.0, "target": 1.0}
        self._return_history = []
        self._visited = set()
        if self.use_random_features:
            rng2 = np.random.RandomState(int(getattr(args, "seed", 42)) + 300)
            state_shape = int(getattr(args, "state_shape", 36))
            self._rand_proj = rng2.randn(max(1,state_shape), 3).astype(np.float32) * 0.1
            self._rand_bias = rng2.randn(3).astype(np.float32) * 0.1

    def compute_shaping_reward(self, episode_batch, t_env=0):
        """Return per-timestep shaping offsets [batch, t, 1] to add to rewards."""
        if not self.enabled:
            return None
        states = episode_batch["state"]
        if hasattr(states, 'detach'):
            states_np = states.detach().cpu().numpy()
        else:
            states_np = np.asarray(states)
        if states_np.ndim == 4:
            states_np = states_np[0]
        batch_size, timesteps = states_np.shape[0], states_np.shape[1]
        offsets = np.zeros((batch_size, timesteps, 1), dtype=np.float32)
        for b in range(batch_size):
            prev_phi = 0.0
            for t in range(timesteps):
                curr_phi = self._phi(states_np[b, t])
                offsets[b, t, 0] = curr_phi - prev_phi
                prev_phi = curr_phi
        return th.tensor(offsets, device=episode_batch.device if hasattr(episode_batch,'device') else 'cpu')

    def _phi(self, state):
        if state is None or len(state) == 0:
            return 0.0
        if self.use_random_features:
            state_arr = np.asarray(state, dtype=np.float32)
            if len(state_arr) != self._rand_proj.shape[0]:
                return 0.0
            vals = np.tanh(state_arr @ self._rand_proj + self._rand_bias)
            w = np.array([self.weights.get(f"rand{i}", 1.0) for i in range(3)], dtype=np.float32)
            return float(self.lambda_scale * (vals * w).sum())
        n_entities = len(state) // 4
        if n_entities == 0:
            return 0.0
        agent_pos = self._extract(state, 0, self.n_agents)
        food_pos = self._extract(state, self.n_agents * 4, max(1, n_entities - self.n_agents))
        coop = self._coop(agent_pos)
        explore = self._explore(agent_pos)
        target = self._target(agent_pos, food_pos)
        return float(self.lambda_scale * (self.weights.get("coop",1.0)*coop + self.weights.get("explore",1.0)*explore + self.weights.get("target",1.0)*target))

    def _extract(self, state, start, count):
        positions = []
        n_entities = len(state) // 4
        for i in range(start, min(start + count, n_entities)):
            base = i * 4
            if base + 1 >= len(state): break
            positions.append([float(state[base]), float(state[base + 1])])
        return positions

    def _coop(self, positions):
        n = len(positions)
        if n < 2: return 0.0
        dists = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dists.append(np.sqrt(dx*dx + dy*dy))
        return -float(np.mean(dists)) / max(1.0, float(self.grid_size * np.sqrt(2)))

    def _explore(self, positions):
        if not positions: return 0.0
        novelty = 0.0
        for pos in positions:
            tp = (round(pos[0],2), round(pos[1],2))
            if tp not in self._visited: novelty += 1.0
            self._visited.add(tp)
        if len(self._visited) > 50000:
            self._visited = set(list(self._visited)[-25000:])
        return float(novelty / max(1, len(positions)))

    def _target(self, agent_pos, food_pos):
        if not agent_pos or not food_pos: return 0.0
        agreements = []
        for food in food_pos:
            dists = [np.sqrt((food[0]-a[0])**2 + (food[1]-a[1])**2) for a in agent_pos]
            if len(dists) < 2 or max(dists) < 1e-6: continue
            agreements.append(1.0 - float(np.std(dists)) / max(float(np.mean(dists)), 1.0))
        return float(np.mean(agreements)) if agreements else 0.0

    def update_weights(self, returns, diagnoses):
        if not self.use_adaptive: return
        self._return_history.extend(returns)
        if len(self._return_history) > 200:
            self._return_history = self._return_history[-200:]
        if len(self._return_history) < 10: return
        success_threshold = max(self.success_min_return, float(np.percentile(self._return_history, self.success_percentile)))
        for item in diagnoses:
            diagnosis = item[1] if isinstance(item, tuple) else item
            if diagnosis is None: continue
            ft = getattr(diagnosis, "failure_type", "unknown")
            fn = FAILURE_TO_FEATURE.get(ft, "explore")
            if fn in self.weights: self.weights[fn] += self.alpha
        for r in returns:
            if r >= success_threshold:
                for k in self.weights: self.weights[k] *= (1.0 - self.beta)
