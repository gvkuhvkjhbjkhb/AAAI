import numpy as np


ACTION_NAMES = {
    0: "noop",
    1: "north",
    2: "south",
    3: "west",
    4: "east",
    5: "load",
}


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_actions(actions):
    arr = _to_numpy(actions)
    if arr.ndim == 4:
        arr = arr[0, :, :, 0]
    elif arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(int)


def _flatten_rewards(rewards):
    arr = _to_numpy(rewards)
    if arr.ndim == 3:
        arr = arr[0, :, 0]
    elif arr.ndim == 2:
        arr = arr[:, 0]
    return arr.astype(float)


def _valid_length(filled):
    arr = _to_numpy(filled)
    if arr.ndim == 3:
        arr = arr[0, :, 0]
    elif arr.ndim == 2:
        arr = arr[:, 0]
    return int(arr.sum())


def _action_name(action, env_name):
    if "foraging" in env_name.lower() or "lbforaging" in env_name.lower():
        return ACTION_NAMES.get(int(action), f"action_{int(action)}")
    return f"action_{int(action)}"


def _action_histogram(actions, length, env_name):
    counts = {}
    if length <= 0:
        return counts
    for action in actions[:length].reshape(-1):
        name = _action_name(action, env_name)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _state_stats(states, length):
    arr = _to_numpy(states)
    if arr.ndim == 3:
        arr = arr[0]
    arr = arr[: max(length, 1)]
    if arr.size == 0:
        return {"mean_abs_state_delta": 0.0, "state_variation": 0.0}
    deltas = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    return {
        "mean_abs_state_delta": float(deltas),
        "state_variation": float(np.std(arr)),
    }


def summarize_episode_batch(episode_batch, env_name="unknown", success_reward=None):
    length = _valid_length(episode_batch["filled"])
    actions = _flatten_actions(episode_batch["actions"])
    rewards = _flatten_rewards(episode_batch["reward"])
    states = _to_numpy(episode_batch["state"])
    n_agents = actions.shape[-1] if actions.ndim >= 2 else 1
    episode_return = float(rewards[:length].sum()) if length > 0 else 0.0
    positive_reward_steps = int((rewards[:length] > 0).sum()) if length > 0 else 0
    zero_reward_steps = int((np.isclose(rewards[:length], 0.0)).sum()) if length > 0 else 0
    is_lbf = "foraging" in env_name.lower() or "lbforaging" in env_name.lower()
    load_counts = []
    if is_lbf and actions.ndim >= 2 and length > 0:
        load_counts = [int((actions[:length, agent_id] == 5).sum()) for agent_id in range(n_agents)]
    stats = _state_stats(states, length)
    action_hist = _action_histogram(actions, length, env_name)
    success_clause = "not configured"
    if success_reward is not None:
        success_clause = str(episode_return >= float(success_reward))

    lines = [
        f"Environment: {env_name}",
        f"Episode length: {length}",
        f"Number of agents: {n_agents}",
        f"Episode return: {episode_return:.4f}",
        f"Success threshold reached: {success_clause}",
        f"Positive reward steps: {positive_reward_steps}",
        f"Zero reward steps: {zero_reward_steps}",
        f"Action histogram: {action_hist}",
        f"Mean absolute state change: {stats['mean_abs_state_delta']:.6f}",
        f"State variation: {stats['state_variation']:.6f}",
    ]
    if is_lbf:
        lines.insert(8, f"Load action counts by agent: {load_counts}")
    return "\n".join(lines)
