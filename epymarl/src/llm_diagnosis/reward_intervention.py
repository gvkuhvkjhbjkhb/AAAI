class FailureRewardIntervention:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "llm_fd_apply_reward_shaping", False))
        self.terminal_bonus = float(getattr(args, "llm_fd_terminal_bonus", 0.0))
        self.failure_penalty = float(getattr(args, "llm_fd_failure_penalty", 0.0))

    def apply(self, episode_batch, diagnosed_indices):
        if not self.enabled or not diagnosed_indices:
            return episode_batch
        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]
        for batch_idx in diagnosed_indices:
            valid = filled[batch_idx, :, 0].nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            terminal_t = int(valid[-1].item())
            if self.failure_penalty != 0.0:
                rewards[batch_idx, : terminal_t + 1] -= self.failure_penalty
            if self.terminal_bonus != 0.0:
                rewards[batch_idx, terminal_t] += self.terminal_bonus
        return episode_batch
