"""Potential function features for Failure-Aware PBRS."""
import numpy as np


class PotentialFeatureExtractor:
    """Extract potential features from episode batch states.

    For LBF/Foraging: states encode grid as (n_agents*4 + n_food*4) features,
    where each entity has (x, y, level, active). We extract agent positions
    and food positions to compute cooperation/exploration/target features.
    """

    def __init__(self, args):
        self.n_agents = int(getattr(args, "n_agents", 3))
        self.gamma = float(getattr(args, "gamma", 0.99))
        self.grid_size = int(getattr(args, "llm_fd_pbrs_grid_size", 10))
        self.novelty_decay = float(getattr(args, "llm_fd_pbrs_novelty_decay", 0.999))
        self.novelty_floor = float(getattr(args, "llm_fd_pbrs_novelty_floor", 0.05))
        self._visited = set()

    def extract(self, episode_batch, batch_idx, t):
        """Extract feature vector phi(s_t) for a single timestep.

        Returns dict of feature_name -> float.
        """
        state = self._get_state(episode_batch, batch_idx, t)
        if state is None:
            return {"coop": 0.0, "explore": 0.0, "target": 0.0}

        agent_positions = self._extract_positions(state, self.n_agents)
        n_food = (len(state) // 4) - self.n_agents
        if n_food <= 0:
            n_food = 3
        food_positions = self._extract_positions(state, self.n_agents, n_food)

        coop = self._coop_feature(agent_positions)
        explore = self._explore_feature(agent_positions)
        target = self._target_feature(agent_positions, food_positions)
        return {"coop": coop, "explore": explore, "target": target}

    def extract_final(self, episode_batch, batch_idx, terminal_t):
        """Extract features at terminal state s_{t+1}."""
        return self.extract(episode_batch, batch_idx, terminal_t)

    def update_visit_count(self, episode_batch, batch_idx, t):
        """Mark visited state for novelty tracking."""
        state = self._get_state(episode_batch, batch_idx, t)
        if state is None:
            return
        agent_positions = self._extract_positions(state, self.n_agents)
        for pos in agent_positions:
            self._visited.add(tuple(pos))
        if len(self._visited) > 100000:
            keep = list(self._visited)[-50000:]
            self._visited = set(keep)

    def reset_novelty(self):
        self._visited.clear()

    def _get_state(self, episode_batch, batch_idx, t):
        states = episode_batch["state"]
        arr = states
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        if arr.ndim == 3:
            return arr[batch_idx, t]
        if arr.ndim == 4:
            return arr[0, batch_idx, t]
        return None

    def _extract_positions(self, state, start_idx, count=None):
        if state is None or len(state) < 4:
            return []
        positions = []
        entities = len(state) // 4
        if count is None:
            count = entities - start_idx
        for i in range(start_idx, min(start_idx + count, entities)):
            base = i * 4
            if base + 1 >= len(state):
                break
            x = float(state[base])
            y = float(state[base + 1])
            positions.append([x, y])
        return positions

    def _coop_feature(self, agent_positions):
        n = len(agent_positions)
        if n < 2:
            return 0.0
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = agent_positions[i][0] - agent_positions[j][0]
                dy = agent_positions[i][1] - agent_positions[j][1]
                dists.append(np.sqrt(dx * dx + dy * dy))
        mean_dist = float(np.mean(dists))
        max_dist = float(self.grid_size * np.sqrt(2))
        return -mean_dist / max(max_dist, 1.0)

    def _explore_feature(self, agent_positions):
        if not agent_positions:
            return 0.0
        novelty = 0.0
        for pos in agent_positions:
            tp = tuple(pos)
            if tp not in self._visited:
                novelty += 1.0
        return float(novelty / max(1, len(agent_positions)))

    def _target_feature(self, agent_positions, food_positions):
        if not agent_positions or not food_positions:
            return 0.0
        agreements = []
        for food in food_positions:
            dists = []
            for agent in agent_positions:
                dx = food[0] - agent[0]
                dy = food[1] - agent[1]
                dists.append(np.sqrt(dx * dx + dy * dy))
            if len(dists) < 2 or max(dists) < 1e-6:
                continue
            mean_d = float(np.mean(dists))
            std_d = float(np.std(dists))
            agreements.append(1.0 - std_d / max(mean_d, 1.0))
        if not agreements:
            return 0.0
        return float(np.mean(agreements))


class RandomPotentialExtractor(PotentialFeatureExtractor):
    """Random potential function for control: random fixed weights on random
    projections of state. Same dimensionality, no meaningful structure."""

    def __init__(self, args):
        super().__init__(args)
        rng = np.random.RandomState(int(getattr(args, "seed", 42)))
        state_shape = int(getattr(args, "state_shape", 36))
        n_features = 3
        self._proj = rng.randn(state_shape, n_features).astype(np.float32) * 0.1
        self._bias = rng.randn(n_features).astype(np.float32) * 0.1

    def extract(self, episode_batch, batch_idx, t):
        state = self._get_state(episode_batch, batch_idx, t)
        if state is None:
            return {"rand0": 0.0, "rand1": 0.0, "rand2": 0.0}
        state_arr = np.asarray(state, dtype=np.float32)
        if len(state_arr) != self._proj.shape[0]:
            return {"rand0": 0.0, "rand1": 0.0, "rand2": 0.0}
        vals = np.tanh(state_arr @ self._proj + self._bias)
        return {f"rand{i}": float(vals[i]) for i in range(len(vals))}

    def update_visit_count(self, episode_batch, batch_idx, t):
        pass

    def reset_novelty(self):
        pass
