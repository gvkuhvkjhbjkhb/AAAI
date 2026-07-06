import hashlib


class FailureRewardIntervention:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "llm_fd_apply_reward_shaping", False))
        self.terminal_bonus = float(getattr(args, "llm_fd_terminal_bonus", 0.0))
        self.failure_penalty = float(getattr(args, "llm_fd_failure_penalty", 0.0))
        self.mode = getattr(args, "llm_fd_intervention_mode", "uniform")
        self.t_max = float(getattr(args, "t_max", 1.0))
        self.confidence_floor = float(getattr(args, "llm_fd_confidence_floor", 0.40))
        self.confidence_reference = float(getattr(args, "llm_fd_confidence_reference", 0.62))
        self.confidence_min_multiplier = float(getattr(args, "llm_fd_confidence_min_multiplier", 0.70))
        self.confidence_max_multiplier = float(getattr(args, "llm_fd_confidence_max_multiplier", 1.30))
        self.early_phase_weight = float(getattr(args, "llm_fd_early_phase_weight", 0.50))
        self.middle_phase_weight = float(getattr(args, "llm_fd_middle_phase_weight", 1.00))
        self.late_phase_weight = float(getattr(args, "llm_fd_late_phase_weight", 0.30))
        self.random_type_use_phase = bool(getattr(args, "llm_fd_random_type_use_phase", False))
        self.semantic_gate_threshold = float(getattr(args, "llm_fd_semantic_gate_threshold", 0.55))
        self.semantic_gate_fallback_weight = float(getattr(args, "llm_fd_semantic_gate_fallback_weight", 1.0))
        self.type_weights = {
            "inefficient_exploration": float(getattr(args, "llm_fd_weight_inefficient_exploration", 0.80)),
            "target_miscoordination": float(getattr(args, "llm_fd_weight_target_miscoordination", 1.20)),
            "insufficient_cooperation": float(getattr(args, "llm_fd_weight_insufficient_cooperation", 1.10)),
            "low_value_overcommitment": float(getattr(args, "llm_fd_weight_low_value_overcommitment", 1.40)),
            "timeout_near_success": float(getattr(args, "llm_fd_weight_timeout_near_success", 0.00)),
            "unknown": float(getattr(args, "llm_fd_weight_unknown", 0.80)),
        }
        self.shaping_trigger_count = 0
        self.shaping_penalty_total = 0.0
        self.shaping_terminal_bonus_total = 0.0
        self.shaping_episode_steps_total = 0

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled or not diagnoses:
            return episode_batch
        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]
        for item in diagnoses:
            if isinstance(item, tuple):
                batch_idx, diagnosis = item
            else:
                batch_idx, diagnosis = item, None
            valid = filled[batch_idx, :, 0].nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            terminal_t = int(valid[-1].item())
            penalty = self._penalty(diagnosis, t_env)
            terminal_bonus = self._terminal_bonus(diagnosis)
            episode_steps = terminal_t + 1
            self._record_accounting(penalty, terminal_bonus, episode_steps)
            if penalty != 0.0:
                rewards[batch_idx, : terminal_t + 1] -= penalty
            if terminal_bonus != 0.0:
                rewards[batch_idx, terminal_t] += terminal_bonus
        return episode_batch

    def _record_accounting(self, penalty, terminal_bonus, episode_steps):
        self.shaping_trigger_count += 1
        self.shaping_episode_steps_total += int(episode_steps)
        self.shaping_penalty_total += float(penalty) * int(episode_steps)
        self.shaping_terminal_bonus_total += float(terminal_bonus)

    def accounting(self):
        if self.shaping_trigger_count > 0:
            avg_penalty = self.shaping_penalty_total / self.shaping_trigger_count
            avg_steps = self.shaping_episode_steps_total / self.shaping_trigger_count
        else:
            avg_penalty = 0.0
            avg_steps = 0.0
        return {
            "triggers": self.shaping_trigger_count,
            "penalty_total": self.shaping_penalty_total,
            "terminal_bonus_total": self.shaping_terminal_bonus_total,
            "episode_steps_total": self.shaping_episode_steps_total,
            "avg_penalty_per_trigger": avg_penalty,
            "avg_steps_per_trigger": avg_steps,
        }

    def log_stats(self, logger, t_env):
        stats = self.accounting()
        logger.log_stat("llm_fd_shaping_triggers", stats["triggers"], t_env)
        logger.log_stat("llm_fd_shaping_penalty_total", stats["penalty_total"], t_env)
        logger.log_stat("llm_fd_shaping_terminal_bonus_total", stats["terminal_bonus_total"], t_env)
        logger.log_stat("llm_fd_shaping_episode_steps_total", stats["episode_steps_total"], t_env)
        logger.log_stat("llm_fd_shaping_avg_penalty_per_trigger", stats["avg_penalty_per_trigger"], t_env)
        logger.log_stat("llm_fd_shaping_avg_steps_per_trigger", stats["avg_steps_per_trigger"], t_env)

    def _penalty(self, diagnosis, t_env):
        if self.failure_penalty == 0.0:
            return 0.0
        if self.mode == "uniform" or diagnosis is None:
            return self.failure_penalty
        if self.mode == "phase_uniform":
            return self.failure_penalty * self._phase_weight(t_env)
        if self.mode not in {"adaptive", "type_specific", "semantic_gate"}:
            return self.failure_penalty
        failure_type = getattr(diagnosis, "failure_type", "unknown")
        raw_confidence = float(getattr(diagnosis, "confidence", self.confidence_floor))
        confidence = self._confidence_multiplier(diagnosis)
        if self.mode == "semantic_gate" and raw_confidence < self.semantic_gate_threshold:
            return self.failure_penalty * confidence * self.semantic_gate_fallback_weight * self._phase_weight(t_env)
        type_weight = self.type_weights.get(failure_type, self.type_weights["unknown"])
        phase_weight = self._phase_weight(t_env) if self.mode in {"adaptive", "semantic_gate"} else 1.0
        return self.failure_penalty * confidence * type_weight * phase_weight

    def _confidence_multiplier(self, diagnosis):
        confidence = max(self.confidence_floor, float(getattr(diagnosis, "confidence", self.confidence_floor)))
        reference = max(self.confidence_floor, self.confidence_reference)
        multiplier = confidence / reference
        return min(self.confidence_max_multiplier, max(self.confidence_min_multiplier, multiplier))

    def _terminal_bonus(self, diagnosis):
        if self.terminal_bonus == 0.0:
            return 0.0
        if diagnosis is None or self.mode == "uniform":
            return self.terminal_bonus
        if getattr(diagnosis, "failure_type", "unknown") == "timeout_near_success":
            return self.terminal_bonus
        return 0.0

    def _phase_weight(self, t_env):
        progress = float(t_env) / max(1.0, self.t_max)
        if progress < 0.20:
            return self.early_phase_weight
        if progress < 0.70:
            return self.middle_phase_weight
        return self.late_phase_weight


class RandomTypeFailureRewardIntervention(FailureRewardIntervention):
    def __init__(self, args):
        super().__init__(args)
        self._failure_types = sorted(self.type_weights)

    def _record_accounting(self, penalty, terminal_bonus, episode_steps):
        self.shaping_trigger_count += 1
        self.shaping_episode_steps_total += int(episode_steps)
        self.shaping_penalty_total += float(penalty) * int(episode_steps)
        self.shaping_terminal_bonus_total += float(terminal_bonus)

    def accounting(self):
        if self.shaping_trigger_count > 0:
            avg_penalty = self.shaping_penalty_total / self.shaping_trigger_count
            avg_steps = self.shaping_episode_steps_total / self.shaping_trigger_count
        else:
            avg_penalty = 0.0
            avg_steps = 0.0
        return {
            "triggers": self.shaping_trigger_count,
            "penalty_total": self.shaping_penalty_total,
            "terminal_bonus_total": self.shaping_terminal_bonus_total,
            "episode_steps_total": self.shaping_episode_steps_total,
            "avg_penalty_per_trigger": avg_penalty,
            "avg_steps_per_trigger": avg_steps,
        }

    def log_stats(self, logger, t_env):
        stats = self.accounting()
        logger.log_stat("llm_fd_shaping_triggers", stats["triggers"], t_env)
        logger.log_stat("llm_fd_shaping_penalty_total", stats["penalty_total"], t_env)
        logger.log_stat("llm_fd_shaping_terminal_bonus_total", stats["terminal_bonus_total"], t_env)
        logger.log_stat("llm_fd_shaping_episode_steps_total", stats["episode_steps_total"], t_env)
        logger.log_stat("llm_fd_shaping_avg_penalty_per_trigger", stats["avg_penalty_per_trigger"], t_env)
        logger.log_stat("llm_fd_shaping_avg_steps_per_trigger", stats["avg_steps_per_trigger"], t_env)

    def _penalty(self, diagnosis, t_env):
        if self.failure_penalty == 0.0 or diagnosis is None:
            return self.failure_penalty
        failure_type = self._sample_type(getattr(diagnosis, "evidence", ""), t_env)
        confidence = self._confidence_multiplier(diagnosis)
        phase_weight = self._phase_weight(t_env) if self.random_type_use_phase else 1.0
        return self.failure_penalty * confidence * self.type_weights.get(failure_type, self.type_weights["unknown"]) * phase_weight

    def _terminal_bonus(self, diagnosis):
        if self.terminal_bonus == 0.0 or diagnosis is None:
            return 0.0
        failure_type = self._sample_type(getattr(diagnosis, "evidence", ""), 0)
        if failure_type == "timeout_near_success":
            return self.terminal_bonus
        return 0.0

    def _sample_type(self, evidence, t_env):
        key = f"{evidence}|{t_env}".encode("utf-8", errors="ignore")
        index = int(hashlib.md5(key).hexdigest(), 16) % len(self._failure_types)
        return self._failure_types[index]


class MatchedRandomTypeFailureRewardIntervention(FailureRewardIntervention):
    def __init__(self, args):
        super().__init__(args)
        self._observed_types = []
        self._max_pool = int(getattr(args, "llm_fd_matched_random_pool", 10000))

    def apply(self, episode_batch, diagnoses, t_env=0):
        if not self.enabled or not diagnoses:
            return episode_batch
        current_types = [
            getattr(item[1] if isinstance(item, tuple) else None, "failure_type", "unknown")
            for item in diagnoses
        ]
        pool = self._observed_types or current_types or ["unknown"]
        rewards = episode_batch.data.transition_data["reward"]
        filled = episode_batch.data.transition_data["filled"]
        for item in diagnoses:
            if isinstance(item, tuple):
                batch_idx, diagnosis = item
            else:
                batch_idx, diagnosis = item, None
            valid = filled[batch_idx, :, 0].nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            terminal_t = int(valid[-1].item())
            if diagnosis is None:
                penalty = self.failure_penalty
                terminal_bonus = 0.0
            else:
                failure_type = self._sample_from_pool(pool, diagnosis, t_env, batch_idx)
                confidence = self._confidence_multiplier(diagnosis)
                phase_weight = self._phase_weight(t_env) if self.random_type_use_phase else 1.0
                penalty = self.failure_penalty * confidence * self.type_weights.get(failure_type, self.type_weights["unknown"]) * phase_weight
                terminal_bonus = self.terminal_bonus if failure_type == "timeout_near_success" else 0.0
            episode_steps = terminal_t + 1
            self._record_accounting(penalty, terminal_bonus, episode_steps)
            if penalty != 0.0:
                rewards[batch_idx, : terminal_t + 1] -= penalty
            if terminal_bonus != 0.0:
                rewards[batch_idx, terminal_t] += terminal_bonus
        self._observed_types.extend(current_types)
        if len(self._observed_types) > self._max_pool:
            self._observed_types = self._observed_types[-self._max_pool :]
        return episode_batch

    def _sample_from_pool(self, pool, diagnosis, t_env, batch_idx):
        evidence = getattr(diagnosis, "evidence", "")
        key = f"{evidence}|{t_env}|{batch_idx}|matched".encode("utf-8", errors="ignore")
        index = int(hashlib.md5(key).hexdigest(), 16) % len(pool)
        return pool[index]
