#!/usr/bin/env python3
"""
HetToM Baseline Layer 1 (Core): LLM-as-Agent verifying the Reasoning Trap.

This is the BASELINE script. It implements the four-cell experimental matrix
required to test whether the Reasoning Trap (Shin 2026) manifests in cooperative
games and whether heterogeneity + Theory-of-Mind breaks it.

Experimental matrix (2 x 2 + human reference):
  1. Homogeneous-NoToM   : same model + same temp, no ToM prompt   (reproduces trap)
  2. Homogeneous-ToM     : same model + same temp, with ToM prompt (key 2)
  3. Heterogeneous-NoToM : diff model/temp, no ToM prompt          (key 1)
  4. Heterogeneous-ToM   : diff model/temp, with ToM prompt        (full method)

Design choices grounded in repo constraints:
  - Uses OFFLINE-capable HuggingFace LLMs (no per-step online API needed;
    models load once, reused across episodes). Compatible with the repo's
    offline_relabel.py HF pipeline (fp16, device_map="auto").
  - Uses a MATRIX-GAME environment (Stag Hunt / Public Goods / Coordination)
    so episodes are SHORT (1-5 steps) -> single RTX 5090 can run thousands of
    LLM-driven episodes in hours, not days.
  - No EPyMARL training loop needed for Layer 1 (Layer 2 will migrate winners
    to LBF+MAPPO). This avoids the per-step-online-LLM infeasibility identified
    in Plan_Assessment.txt.
  - Logs in a format compatible with the repo's results/ convention.

Metrics (per the survey):
  1. Perspective diversity  : pairwise KL divergence of action distributions
  2. Cooperation payoff     : mean team return
  3. Equilibrium convergence : action-distribution stability over episode windows
  4. ToM prediction accuracy : predicted-vs-actual teammate action match rate
                              (only meaningful for ToM configs)

Usage:
  python3 hettoam_baseline.py --config configs/stag_hunt_hom_notom.yaml
  python3 hettoam_baseline.py --matrix    # run all 4 cells + analyze
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


# =============================================================================
# Environment: lightweight matrix games (single-shot & repeated)
# =============================================================================

@dataclass
class MatrixGame:
    """A symmetric 2-player matrix game. Generalizes Stag Hunt, Public Goods,
    Coordination. Payoffs stored as dict[(a1,a2)] = (r1, r2).

    For n>2 players we use the public-goods / coordination reduction:
    each player's payoff depends on own action + the aggregate of others.
    """
    name: str
    n_agents: int
    n_actions: int            # actions indexed 0..n_actions-1
    action_names: list        # human-readable action names
    payoff: Any               # callable(actions: tuple) -> tuple of rewards

    def payoff_vector(self, actions):
        """actions: tuple of length n_agents. returns list of rewards."""
        return list(self.payoff(actions))


def make_stag_hunt():
    """Classic 2-player Stag Hunt.
    Actions: 0=Stag (cooperate), 1=Hare (defect).
    Payoff matrix (row=agent1):
                 Stag        Hare
      Stag     (3,3)       (0,2)
      Hare     (2,0)       (2,2)
    Two pure Nash equilibria: (Stag,Stag) payoff-dominant, (Hare,Hare) risk-dominant.
    This is the canonical game for studying cooperation emergence.
    """
    payoff_matrix = {
        (0, 0): (3, 3),
        (0, 1): (0, 2),
        (1, 0): (2, 0),
        (1, 1): (2, 2),
    }
    def payoff(actions):
        return payoff_matrix[(actions[0], actions[1])]
    return MatrixGame(
        name="stag_hunt",
        n_agents=2,
        n_actions=2,
        action_names=["Stag", "Hare"],
        payoff=payoff,
    )


def make_public_goods(n_agents=4, multiplier=1.6, cost=1.0):
    """N-player Public Goods game (linear).
    Action 0 = Contribute (cooperate), Action 1 = Keep (defect).
    Each contributor pays `cost`, pool is multiplied by `multiplier` and split
    equally among all. Reward_i = endowment - contribution_i + share_of_pool.
    """
    endowment = 2.0
    def payoff(actions):
        n_c = sum(1 for a in actions if a == 0)
        pool = n_c * cost * multiplier
        share = pool / n_agents
        return [endowment - (cost if a == 0 else 0.0) + share for a in actions]
    return MatrixGame(
        name=f"public_goods_n{n_agents}",
        n_agents=n_agents,
        n_actions=2,
        action_names=["Contribute", "Keep"],
        payoff=payoff,
    )


def make_coordination(n_agents=2, n_actions=3, bonus=2.0):
    """Pure coordination game: all agents get `bonus` if they choose the same
    action, else 0. Multiple Pareto-ranked equilibria if actions have different
    base payoffs; here all equal -> tests pure convention emergence.
    """
    def payoff(actions):
        if len(set(actions)) == 1:
            return [bonus] * n_agents
        return [0.0] * n_agents
    return MatrixGame(
        name=f"coordination_n{n_agents}_k{n_actions}",
        n_agents=n_agents,
        n_actions=n_actions,
        action_names=[f"A{i}" for i in range(n_actions)],
        payoff=payoff,
    )


def make_battle_of_the_sexes():
    """2-player Battle of the Sexes.
    Actions: 0=Opera, 1=Football.
    Payoff matrix:
                 Opera      Football
      Opera    (3,2)       (0,0)
      Football (0,0)       (2,3)
    Two pure Nash equilibria: (Opera,Opera) and (Football,Football), but with
    conflicting preferences. Tests coordination under preference asymmetry —
    a harder case for ToM (agents must infer the other's preferred equilibrium).
    """
    payoff_matrix = {
        (0, 0): (3, 2),
        (0, 1): (0, 0),
        (1, 0): (0, 0),
        (1, 1): (2, 3),
    }
    def payoff(actions):
        return payoff_matrix[(actions[0], actions[1])]
    return MatrixGame(
        name="battle_of_the_sexes",
        n_agents=2,
        n_actions=2,
        action_names=["Opera", "Football"],
        payoff=payoff,
    )


def make_chicken():
    """2-player Chicken (Hawk-Dove).
    Actions: 0=Dove, 1=Hawk.
    Payoff matrix:
                 Dove       Hawk
      Dove     (3,3)       (1,5)
      Hawk     (5,1)       (0,0)
    Two pure Nash equilibria: (Hawk,Dove) and (Dove,Hawk). Tests anti-
    coordination under conflict — the worst outcome is mutual Hawk (crash).
    Directly probes LLM risk appetite and ToM (predicting defection).
    """
    payoff_matrix = {
        (0, 0): (3, 3),
        (0, 1): (1, 5),
        (1, 0): (5, 1),
        (1, 1): (0, 0),
    }
    def payoff(actions):
        return payoff_matrix[(actions[0], actions[1])]
    return MatrixGame(
        name="chicken",
        n_agents=2,
        n_actions=2,
        action_names=["Dove", "Hawk"],
        payoff=payoff,
    )


GAMES = {
    "stag_hunt": make_stag_hunt,
    "public_goods": lambda: make_public_goods(n_agents=4),
    "coordination": lambda: make_coordination(n_agents=2, n_actions=3),
    "battle_of_the_sexes": make_battle_of_the_sexes,
    "chicken": make_chicken,
}


# =============================================================================
# Repeated-game wrapper: same matrix game played T rounds with history
# =============================================================================

@dataclass
class RepeatedGame:
    """Repeated matrix game. State = history of last `memory` rounds.
    This gives LLM agents something to reason about (and ToM to model)."""
    base: MatrixGame
    horizon: int = 5          # rounds per episode
    memory: int = 2           # rounds of history visible to agents

    def reset(self):
        return {"round": 0, "history": []}

    def step(self, actions):
        rewards = self.base.payoff_vector(tuple(actions))
        return rewards

    def observation_for_agent(self, state, agent_idx):
        """Textual observation for an LLM agent."""
        recent = state["history"][-self.memory:] if self.memory > 0 else []
        lines = [f"Round {state['round']} of {self.horizon}."]
        if recent:
            lines.append("Recent rounds (oldest first):")
            for r, (acts, rews) in enumerate(recent):
                act_str = ", ".join(
                    f"Agent{i+1}={self.base.action_names[a]}"
                    for i, a in enumerate(acts)
                )
                rew_str = ", ".join(f"{r:.1f}" for r in rews)
                lines.append(f"  round-{r}: actions=[{act_str}] payoffs=[{rew_str}]")
        else:
            lines.append("No history yet (first round).")
        return "\n".join(lines)


# =============================================================================
# LLM Agent: loads a HF model, builds prompts, samples actions
# =============================================================================

class LLMAgent:
    """A single LLM-driven agent. Loads a HuggingFace causal LM once and reuses
    it. Supports a ToM mode where the agent is also asked to predict teammates'
    actions before choosing its own.

    Designed to match the repo's offline_relabel.py HF loading idiom:
      AutoModelForCausalLM + AutoTokenizer, fp16 if CUDA, device_map="auto".
    """

    def __init__(self, agent_id, model_name, temperature, role_label,
                 use_tom=False, tom_order=1, device="auto", seed=0,
                 use_talk=False, adaptive_tom=False, gated_talk_tom=False,
                 atom_max_order=3, atom_warmup=3, atom_epsilon=0.15,
                 atom_ema_alpha=0.3, gate_trust_threshold=0.6,
                 gate_ema_alpha=0.3, diversity_preserving_gate=False,
                 api_base=None, api_key=None):
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = max(temperature, 0.01)  # avoid div-by-zero
        self.role_label = role_label
        self.use_tom = use_tom
        self.tom_order = tom_order
        self.seed = seed
        self._rng = random.Random(seed * 1000 + agent_id)
        self._model = None
        self._tokenizer = None
        self._device = device
        # --- Silicon Flow / OpenAI-compatible API backend (Round-4):
        #     when api_base + api_key are set, LLM calls go through the API
        #     instead of loading a local HF model. This enables using 7B+
        #     models (Qwen2.5-7B, GLM-4-9B) from different families without
        #     a local GPU, addressing the Round-3 weakness of 3B/1.5B same-
        #     family models. ---
        self.api_base = api_base
        self.api_key = api_key
        self._api_client = None
        # --- cheap-talk channel (Madmoun & Lahlou 2025, EACL 2026) ---
        self.use_talk = use_talk
        # --- adaptive ToM (Mu et al. 2026, AAAI 2026): estimate partner's
        #     ToM order from hit-rate history, align own reasoning depth ---
        self.adaptive_tom = adaptive_tom
        # history of (predicted, actual) teammate actions per teammate id
        self._tom_history = defaultdict(list)
        # running estimated ToM order per teammate (init to self.tom_order)
        self._est_tom_order = defaultdict(lambda: self.tom_order)
        # --- improved A-ToM (Round-3): per-order EMA hit-rate tracking +
        #     epsilon-greedy bandit selection with warmup. Replaces the
        #     coarse "<0.4 deepen / >0.75 lower" threshold rule, which on a
        #     2-action game (random baseline 0.5) almost never fired and
        #     could only oscillate between 1 and 3. Scoring every computed
        #     chain order (not just the selected one) gives far more data
        #     per round, so the bandit converges on short histories too. ---
        self.atom_max_order = atom_max_order
        self.atom_warmup = atom_warmup
        self.atom_epsilon = atom_epsilon
        self.atom_ema_alpha = atom_ema_alpha
        # per-teammate per-order hit history: {tid: {order: [0/1, ...]}}
        self._tom_order_hits = defaultdict(lambda: defaultdict(list))
        # --- gated talk+ToM arbitration (Round-3): cheap-talk produces a
        #     public intent (signal); ToM judges whether the signal is
        #     trustworthy. Trust if signal agrees with ToM belief OR the
        #     teammate's historical signal-action match EMA >= threshold;
        #     otherwise fall back to the ToM belief. Resolves the
        #     belief-signal interference that made naive ToM+talk fail
        #     (het_tom_talk < het_tom). Motivated by Madmoun & Lahlou 2025
        #     (cheap-talk helps) + El Mir, Takac & Lahlou 2026 / Yao et al.
        #     2026 (cheap-talk signals can be unreliable and need repair). ---
        self.gated_talk_tom = gated_talk_tom
        self.gate_trust_threshold = gate_trust_threshold
        self.gate_ema_alpha = gate_ema_alpha
        # --- diversity-preserving gating (Round-4): the core innovation to
        #     resolve the diversity-alignment tension. When enabled, the gate
        #     ONLY intervenes (conditions the action on the gated belief)
        #     when there is a signal-belief CONFLICT. When signal and ToM
        #     belief agree, the agent is left to choose independently —
        #     preserving the cognitive diversity that heterogeneity provides.
        #     This directly addresses the Round-3 failure where gating
        #     compressed diversity to ~0 (0.013) while still not beating
        #     the homogeneous baseline (1.734 vs 2.325). ---
        self.diversity_preserving_gate = diversity_preserving_gate
        # per-teammate EMA of (signal == actual), init 0.5 (uninformed)
        self._signal_ema = defaultdict(lambda: 0.5)
        self._signal_history = defaultdict(list)

    # ---- lazy model loading (so configs without GPU still parse) ----
    # Global model cache: {model_name: (model, tokenizer)} so homogeneous
    # cells (2 agents, same model) only load ONE copy, not two. This cuts
    # VRAM usage in half and load time by 50% for hom cells.
    _MODEL_CACHE = {}

    def _ensure_model(self):
        if self._model is not None:
            return
        if self.api_base:
            import openai
            self._api_client = openai.OpenAI(
                base_url=self.api_base, api_key=self.api_key)
            self._model = "api"  # sentinel: API mode active
            return
        cache_key = self.model_name
        if cache_key in LLMAgent._MODEL_CACHE:
            self._model, self._tokenizer = LLMAgent._MODEL_CACHE[cache_key]
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True)
        # 4-bit quantization: fits 7B (~4GB) + 9B (~5GB) = 9GB total in 32GB
        # VRAM, so both heterogeneous models stay loaded simultaneously with
        # zero swap overhead. Quality loss is negligible for short game prompts.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, quantization_config=bnb_config,
            device_map=self._device, trust_remote_code=True)
        self._model.eval()
        LLMAgent._MODEL_CACHE[cache_key] = (self._model, self._tokenizer)

    # ---- prompt construction ----
    def _build_action_prompt(self, game, obs, teammate_preds=None, signals=None):
        role_line = f"You are Agent {self.agent_id}, role: {self.role_label}."
        game_line = (
            f"Game: {game.base.name}. You are one of {game.base.n_agents} agents. "
            f"Actions: " + ", ".join(
                f"{i}={n}" for i, n in enumerate(game.base.action_names)) + "."
        )
        obs_line = f"Observation:\n{obs}"
        instr = (
            f"Choose your action by replying with a single integer in "
            f"[0, {game.base.n_actions - 1}]. Reply with ONLY the integer."
        )
        parts = [role_line, game_line, obs_line]
        if self.use_tom and teammate_preds:
            tom_line = "Your Theory-of-Mind reasoning about teammates:\n"
            for tid, pred in teammate_preds.items():
                tom_line += f"  Agent {tid}: you predict action {pred}.\n"
            parts.append(tom_line)
        if self.use_talk and signals:
            talk_line = "Teammates signaled these intended actions this round:\n"
            for tid, s in signals.items():
                talk_line += f"  Agent {tid}: signaled action {s}.\n"
            parts.append(talk_line)
        parts.append(instr)
        return "\n\n".join(parts)

    # ---- low-level LLM call ----
    def _generate(self, prompt, max_new_tokens=8):
        self._ensure_model()
        if self.api_base:
            return self._generate_api(prompt, max_new_tokens)
        import torch
        msgs = [{"role": "user", "content": prompt}]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()

    def _generate_api(self, prompt, max_new_tokens=8, retries=3):
        """Generate text via OpenAI-compatible API (Silicon Flow).
        Includes retry with exponential backoff for rate-limit/transient
        errors. Strips markdown code fences from the response for cleaner
        JSON parsing."""
        import time as _time
        for attempt in range(retries):
            try:
                resp = self._api_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                )
                text = resp.choices[0].message.content.strip()
                # strip markdown code fences that some models add
                if text.startswith("```"):
                    lines = text.split("\n")
                    lines = [l for l in lines if not l.strip().startswith("```")]
                    text = "\n".join(lines).strip()
                return text
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    _time.sleep(wait)
                else:
                    print(f"[agent {self.agent_id}] API error after "
                          f"{retries} retries: {e}")
                    return ""

    # ---- action / ToM parsing ----
    def _parse_action(self, text, n_actions):
        for tok in text.replace(",", " ").split():
            if tok.isdigit():
                a = int(tok)
                if 0 <= a < n_actions:
                    return a
        return self._rng.randrange(n_actions)  # fallback random

    def _parse_tom_json(self, text, teammate_ids, n_actions):
        preds = {}
        try:
            obj = json.loads(text[text.find("{"):text.rfind("}") + 1])
            for t in teammate_ids:
                v = obj.get(str(t), obj.get(t))
                if isinstance(v, int) and 0 <= v < n_actions:
                    preds[t] = v
        except Exception:
            pass
        for t in teammate_ids:
            if t not in preds:
                preds[t] = self._rng.randrange(n_actions)
        return preds

    def _build_recursive_tom_prompt(self, game, obs, teammate_ids, order,
                                    lower_order_preds=None):
        """Build a ToM prompt of the given recursive order.

        order=1: "what will teammate X do?"  (predict teammate's action)
        order=2: "what does teammate X think I will do?"  (predict teammate's
                 belief about my action)
        order=3: "what does teammate X think I think X will do?" (one more level)

        lower_order_preds carries the predictions from order-1 so the prompt
        can condition on them (chain the recursion).
        """
        role_line = f"You are Agent {self.agent_id}, role: {self.role_label}."
        game_line = (
            f"Game: {game.base.name}. Actions: "
            + ", ".join(f"{i}={n}" for i, n in enumerate(game.base.action_names))
            + "."
        )
        obs_line = f"Observation:\n{obs}"
        tids = ", ".join(str(t) for t in teammate_ids)
        if order == 1:
            question = (
                f"Predict each teammate's next action. Consider what each "
                f"teammate would rationally choose given the same observation.")
        elif order == 2:
            question = (
                f"Now put yourself in each teammate's shoes: what action does "
                f"each teammate think YOU will choose? This is second-order "
                f"Theory of Mind.")
        else:
            question = (
                f"Go one level deeper: what does each teammate think you "
                f"believe THEY will choose? This is order-{order} Theory of Mind.")
        context = ""
        if lower_order_preds:
            context = "Your lower-order ToM predictions so far:\n"
            for o, preds in lower_order_preds.items():
                ctx = ", ".join(f"Agent{t}->{a}" for t, a in preds.items())
                context += f"  order {o}: {ctx}\n"
        instr = (
            f"Reply with ONLY a JSON object mapping teammate id to action "
            f'integer, e.g. {{"2": 0, "3": 1}} for teammates {tids}. No other text.')
        parts = [role_line, game_line, obs_line]
        if context:
            parts.append(context)
        parts.append(question)
        parts.append(instr)
        return "\n\n".join(parts)

    # ---- cheap-talk: announce intended action (Madmoun & Lahlou 2025) ----
    def _build_intent_prompt(self, game, obs, signals):
        """Build a prompt asking the agent to announce its intended action.
        `signals` is a dict {teammate_id: announced_intent} of signals
        received from teammates this round (may be empty on the first call
        or in no-talk cells)."""
        role_line = f"You are Agent {self.agent_id}, role: {self.role_label}."
        game_line = (
            f"Game: {game.base.name}. Actions: "
            + ", ".join(f"{i}={n}" for i, n in enumerate(game.base.action_names))
            + "."
        )
        obs_line = f"Observation:\n{obs}"
        parts = [role_line, game_line, obs_line]
        if signals:
            sig_line = "Teammates announced these intended actions this round:\n"
            for tid, s in signals.items():
                sig_line += f"  Agent {tid}: intends action {s}.\n"
            parts.append(sig_line)
            instr = (
                "A teammate has signaled their intent. Reply with the single "
                "integer action YOU intend to take, considering their signal. "
                "Reply with ONLY the integer.")
        else:
            instr = (
                "Announce the action you intend to take this round. Reply "
                "with ONLY a single integer in "
                f"[0, {game.base.n_actions - 1}].")
        parts.append(instr)
        return "\n\n".join(parts)

    def announce(self, game, obs, signals):
        """Cheap-talk: emit intended action before the real action step.
        Returns an integer action (the announced intent)."""
        prompt = self._build_intent_prompt(game, obs, signals)
        text = self._generate(prompt, max_new_tokens=8)
        return self._parse_action(text, game.base.n_actions)

    # ---- public API ----
    def act(self, game, obs, teammate_ids, signals=None):
        """Choose an action. If use_talk, `signals` (teammates' announced
        intents) condition the choice. If use_tom, run recursive ToM reasoning
        up to tom_order levels (or, if adaptive_tom, up to the estimated
        partner ToM order), then condition the action on the prediction.
        If gated_talk_tom, arbitrate between the cheap-talk signal and the
        ToM belief into a single gated belief (see _gate_signals) and feed
        ONLY that belief to the action prompt -- this is the controlled
        difference vs naive ToM+talk, which fed both and let them interfere.

        Returns (action, tom_preds, tom_chains, gate_decisions):
          tom_preds       : raw highest-order ToM prediction per teammate
                            (stored for the tom_prediction_accuracy metric,
                            comparable across cells)
          tom_chains      : {tid: {order: preds}} for A-ToM per-order scoring
          gate_decisions  : {tid: {...}} for the gated-arbitration metrics
                            (empty for non-gated cells)"""
        teammate_preds = None
        tom_chains = {}
        if self.use_tom:
            tom_orders_to_run = self._decide_tom_orders(teammate_ids)
            teammate_preds = {}
            for tid in teammate_ids:
                order = tom_orders_to_run.get(tid, self.tom_order)
                local_chain = {}
                for o in range(1, order + 1):
                    lower = local_chain if local_chain else None
                    prompt = self._build_recursive_tom_prompt(
                        game, obs, [tid], o, lower)
                    text = self._generate(prompt, max_new_tokens=32)
                    preds = self._parse_tom_json(text, [tid],
                                                 game.base.n_actions)
                    local_chain[o] = preds
                tom_chains[tid] = local_chain
                # highest-order prediction for this teammate
                teammate_preds.update(local_chain[order])
        # --- belief-signal arbitration (gated talk+ToM) ---
        gate_decisions = {}
        if self.gated_talk_tom and signals:
            belief, gate_decisions = self._gate_signals(
                teammate_ids, teammate_preds, signals)
            if self.diversity_preserving_gate:
                # Round-4 core innovation: only intervene (condition the
                # action on the gated belief) when there is a signal-belief
                # CONFLICT. When signal and ToM belief agree, agents are
                # left to choose independently — preserving the cognitive
                # diversity that heterogeneity provides. This directly
                # targets the Round-3 failure where gating compressed
                # diversity to ~0 (0.013) while not beating baseline.
                has_conflict = any(
                    d.get("signal") is not None
                    and d.get("tom_pred") is not None
                    and d.get("signal") != d.get("tom_pred")
                    for d in gate_decisions.values()
                )
                if has_conflict:
                    prompt_belief = belief
                    prompt_signals = None
                else:
                    prompt_belief = None
                    prompt_signals = None
            else:
                prompt_belief = belief
                prompt_signals = None
        else:
            prompt_belief = teammate_preds
            prompt_signals = signals       # existing naive behavior
        prompt = self._build_action_prompt(game, obs, prompt_belief,
                                           prompt_signals)
        text = self._generate(prompt, max_new_tokens=8)
        action = self._parse_action(text, game.base.n_actions)
        return action, teammate_preds, tom_chains, gate_decisions

    def _decide_tom_orders(self, teammate_ids):
        """A-ToM: decide per-teammate ToM reasoning order.

        Improved design (Round-3), replacing the coarse
        ``hit_rate < 0.4 -> deepen, > 0.75 -> lower`` rule. Problems with
        the old rule: (i) on a 2-action game the random baseline is 0.5, so
        ``< 0.4`` almost never triggered and ``> 0.75`` was rare -> the
        adaptation barely fired; (ii) it only read the last 10 predictions,
        so estimates were noisy on short histories; (iii) it could only
        oscillate between the two ends (1 <-> 3) without modelling which
        order is actually best.

        New design: a contextual bandit over ToM orders {1..max_order}.
        Per (teammate, order) we keep an EMA of the prediction hit-rate
        (computed over ALL history, not just last 10, so it is stable even
        with short interaction histories). Selection is epsilon-greedy with
        a warmup that forces every order to be sampled at least
        ``atom_warmup`` times before exploiting. This lets the agent
        discover the order that best aligns with each teammate's reasoning
        depth (Mu et al. 2026: misaligned ToM orders impair coordination).

        If not adaptive_tom, use the fixed ``tom_order``."""
        if not self.adaptive_tom:
            return {tid: self.tom_order for tid in teammate_ids}
        out = {}
        for tid in teammate_ids:
            hits = self._tom_order_hits.get(tid, {})
            chosen = None
            # warmup: force each order to be sampled >= atom_warmup times
            for o in range(1, self.atom_max_order + 1):
                if len(hits.get(o, [])) < self.atom_warmup:
                    chosen = o
                    break
            if chosen is None:
                if self._rng.random() < self.atom_epsilon:
                    chosen = self._rng.randrange(1, self.atom_max_order + 1)
                else:
                    ema = {o: self._ema(hits.get(o, []), self.atom_ema_alpha)
                           for o in range(1, self.atom_max_order + 1)}
                    best = max(ema.values())
                    cand = [o for o, v in ema.items()
                            if abs(v - best) < 1e-9]
                    chosen = cand[0] if len(cand) == 1 else self._rng.choice(cand)
            self._est_tom_order[tid] = chosen
            out[tid] = chosen
        return out

    @staticmethod
    def _ema(values, alpha):
        """Exponential moving average over a list of 0/1 hits. alpha is the
        weight on the newest observation. Returns 0.0 for empty input."""
        if not values:
            return 0.0
        v = float(values[0])
        for x in values[1:]:
            v = alpha * float(x) + (1.0 - alpha) * v
        return v

    def _gate_signals(self, teammate_ids, tom_preds, signals):
        """Gated talk+ToM arbitration. For each teammate, decide whether to
        trust their announced cheap-talk signal or fall back to the ToM
        belief, then return a single gated prediction per teammate.

        Trust policy (trust the signal if EITHER holds):
          (a) consistency: the signal agrees with the ToM belief, OR
          (b) reliability: the teammate's historical signal-action match
              EMA >= gate_trust_threshold.
        Otherwise distrust -> use the ToM belief. This directly attacks the
        Round-2 failure where naive ToM+talk (het_tom_talk) dumped both the
        signal and the belief into the prompt with no arbitration, letting
        them interfere (het_tom_talk 0.890 < het_tom 1.259).

        Returns (gated_preds, decisions) where decisions logs, per
        teammate, the signal, tom_pred, gated_pred, trusted flag, and
        trust_score for downstream metrics."""
        gated = {}
        decisions = {}
        for tid in teammate_ids:
            sig = signals.get(tid) if signals else None
            pred = tom_preds.get(tid) if tom_preds else None
            trust_score = self._signal_ema.get(tid, 0.5)
            trusted = False
            if sig is not None and pred is not None:
                if sig == pred:
                    trusted = True          # consistent -> trust signal
                    chosen = sig
                elif trust_score >= self.gate_trust_threshold:
                    trusted = True          # historically reliable -> trust
                    chosen = sig
                else:
                    trusted = False         # untrusted -> ToM belief
                    chosen = pred
            elif sig is not None:
                trusted = True
                chosen = sig
            elif pred is not None:
                trusted = False
                chosen = pred
            else:
                trusted = False
                chosen = self._rng.randrange(2) if self.use_tom else None
            gated[tid] = chosen
            decisions[tid] = {"signal": sig, "tom_pred": pred,
                              "gated_pred": chosen, "trusted": trusted,
                              "trust_score": float(trust_score)}
        return gated, decisions

    def update_signal_history(self, teammate_ids, signals, actual_actions):
        """Update the per-teammate signal-action trust EMA used by the gate.
        Called after actions are observed each round."""
        if not signals:
            return
        for tid in teammate_ids:
            sig = signals.get(tid)
            if sig is None:
                continue
            idx = tid - 1 if isinstance(tid, int) else int(tid) - 1
            if 0 <= idx < len(actual_actions):
                hit = 1.0 if sig == actual_actions[idx] else 0.0
                prev = self._signal_ema.get(tid, 0.5)
                self._signal_ema[tid] = (self.gate_ema_alpha * hit
                                         + (1.0 - self.gate_ema_alpha) * prev)
                self._signal_history[tid].append((sig, actual_actions[idx]))

    def update_tom_history(self, teammate_ids, tom_chains, actual_actions):
        """Record (predicted, actual) per teammate to calibrate A-ToM.

        Scores EVERY order computed in the chain (1..chosen) against the
        actual action, so each order's EMA hit-rate reflects its intrinsic
        accuracy and the bandit converges much faster on short histories
        (a Round-2 problem: 20 episodes gave too little per-order data under
        the old rule, which only scored the single selected prediction).
        Also keeps the legacy _tom_history (highest-order prediction) for
        backward compatibility."""
        if not tom_chains:
            return
        for tid in teammate_ids:
            chain = tom_chains.get(tid)
            if not chain:
                continue
            idx = tid - 1 if isinstance(tid, int) else int(tid) - 1
            if not (0 <= idx < len(actual_actions)):
                continue
            actual = actual_actions[idx]
            for o in sorted(chain.keys()):
                preds = chain[o]
                if tid in preds:
                    hit = 1.0 if preds[tid] == actual else 0.0
                    self._tom_order_hits[tid][o].append(hit)
            top_order = max(chain.keys())
            if tid in chain[top_order]:
                self._tom_history[tid].append(
                    (chain[top_order][tid], actual))


# =============================================================================
# Agent pool factory: builds the 4 experimental configurations
# =============================================================================

def build_agents(config):
    """Build N LLM agents per a config dict.

    Config fields:
      game          : one of GAMES keys
      n_agents      : override game's agent count (for public_goods)
      homogeneous   : bool — if True all agents share model+temp; else differ
      use_tom       : bool — if True agents run ToM prediction step
      models_homo   : single model name used for all agents when homogeneous
      models_het    : list of model names cycled across agents when heterogeneous
      temps_het     : list of temperatures cycled when heterogeneous
      temp_homo     : single temperature when homogeneous
      roles         : list of role labels cycled across agents
      seed          : base seed
    """
    game = GAMES[config["game"]]()
    if config.get("n_agents"):
        # rebuild public_goods with requested N
        if config["game"] == "public_goods":
            game = make_public_goods(n_agents=config["n_agents"])
    n = game.n_agents
    homogeneous = config["homogeneous"]
    use_tom = config["use_tom"]
    roles = config.get("roles", [f"player{i+1}" for i in range(n)])
    seed = config.get("seed", 0)

    agents = []
    for i in range(n):
        if homogeneous:
            model = config["models_homo"]
            temp = config["temp_homo"]
        else:
            models = config["models_het"]
            temps = config["temps_het"]
            model = models[i % len(models)]
            temp = temps[i % len(temps)]
        agents.append(LLMAgent(
            agent_id=i + 1,
            model_name=model,
            temperature=temp,
            role_label=roles[i % len(roles)],
            use_tom=use_tom,
            tom_order=config.get("tom_order", 1),
            seed=seed,
            use_talk=config.get("use_talk", False),
            adaptive_tom=config.get("adaptive_tom", False),
            gated_talk_tom=config.get("gated_talk_tom", False),
            atom_max_order=config.get("atom_max_order", 3),
            atom_warmup=config.get("atom_warmup", 3),
            atom_epsilon=config.get("atom_epsilon", 0.15),
            atom_ema_alpha=config.get("atom_ema_alpha", 0.3),
            gate_trust_threshold=config.get("gate_trust_threshold", 0.6),
            gate_ema_alpha=config.get("gate_ema_alpha", 0.3),
            diversity_preserving_gate=config.get(
                "diversity_preserving_gate", False),
            api_base=config.get("api_base"),
            api_key=config.get("api_key"),
        ))
    return game, agents


# =============================================================================
# Episode runner + metrics
# =============================================================================

def run_episode(game, agents, horizon, memory):
    rg = RepeatedGame(base=game, horizon=horizon, memory=memory)
    state = rg.reset()
    trajectory = []  # list of (round, actions, rewards, tom_preds, signals)
    use_talk = any(getattr(a, "use_talk", False) for a in agents)
    for _ in range(horizon):
        obs = rg.observation_for_agent(state, None)  # shared textual obs
        teammate_ids = [a.agent_id for a in agents]
        # --- cheap-talk phase: each agent announces intent (Madmoun 2025) ---
        signals = {}
        if use_talk:
            for ag in agents:
                sig_in = {t: signals[t] for t in signals if t != ag.agent_id}
                intent = ag.announce(rg, obs, sig_in)
                signals[ag.agent_id] = intent
        # --- action phase ---
        actions, tom_preds, tom_chains, gate_decisions = [], [], [], []
        for ag in agents:
            others = [t for t in teammate_ids if t != ag.agent_id]
            sig_for = {t: signals[t] for t in others} if use_talk else None
            act, preds, chains, gates = ag.act(rg, obs, others, sig_for)
            actions.append(act)
            tom_preds.append(preds)
            tom_chains.append(chains)
            gate_decisions.append(gates)
        rewards = rg.step(actions)
        # --- A-ToM: update per-order hit history (Mu 2026) ---
        for ai, ag in enumerate(agents):
            others = [t for t in teammate_ids if t != ag.agent_id]
            if getattr(ag, "use_tom", False):
                ag.update_tom_history(others, tom_chains[ai], actions)
            # --- gated talk+ToM: update signal trust EMA ---
            if getattr(ag, "gated_talk_tom", False) and use_talk:
                ag.update_signal_history(others, signals, actions)
        trajectory.append({
            "round": state["round"],
            "actions": actions,
            "rewards": rewards,
            "tom_preds": tom_preds,
            "signals": dict(signals) if use_talk else None,
            "gate_decisions": gate_decisions,
        })
        state["history"].append((actions, rewards))
        state["round"] += 1
    return trajectory


def _bootstrap_ci(values, n_boot=2000, ci=0.95, rng=None):
    """Bootstrap confidence interval for the mean of a list of scalars."""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    rng = rng or np.random.default_rng(0)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return lo, hi


def compute_metrics(episodes, game, n_boot=2000, seed=0):
    """Compute the 4 core metrics across a list of episodes, with bootstrap
    95% confidence intervals on per-episode scalars (cooperation payoff) and
    on the ToM accuracy rate. Diversity/convergence are distribution-level and
    reported as point estimates (CI computed at the multi-seed aggregation
    stage in analyze_layer1.py)."""
    rng = np.random.default_rng(seed)
    n_a = game.n_actions
    # 1. perspective diversity: mean pairwise KL of per-agent action histograms
    action_counts = [np.zeros(n_a) for _ in range(game.n_agents)]
    for ep in episodes:
        for step in ep:
            for i, a in enumerate(step["actions"]):
                action_counts[i][a] += 1.0
    dists = []
    for c in action_counts:
        s = c.sum()
        dists.append(c / s if s > 0 else np.ones(n_a) / n_a)
    kls = []
    for i in range(len(dists)):
        for j in range(i + 1, len(dists)):
            p, q = dists[i] + 1e-12, dists[j] + 1e-12
            p, q = p / p.sum(), q / q.sum()
            kls.append(float(np.sum(p * np.log(p / q))))
    perspective_div = float(np.mean(kls)) if kls else 0.0

    # 2. cooperation payoff: per-episode mean team return (per agent per round)
    per_ep_payoffs = [float(np.mean([s["rewards"][0] for s in ep]))
                      for ep in episodes]
    coop_payoff = float(np.mean(per_ep_payoffs)) if per_ep_payoffs else 0.0
    payoff_lo, payoff_hi = _bootstrap_ci(per_ep_payoffs, n_boot, 0.95, rng)

    # 3. equilibrium convergence: 1 - normalized action-distribution drift
    #    between first and second half of episodes
    def half_dist(ep_half):
        c = np.zeros(n_a)
        for ep in ep_half:
            for s in ep:
                c[s["actions"][0]] += 1.0  # agent 1's action distribution
        return c / c.sum() if c.sum() > 0 else np.ones(n_a) / n_a
    mid = len(episodes) // 2
    d1, d2 = half_dist(episodes[:mid]), half_dist(episodes[mid:])
    tv_dist = 0.5 * float(np.abs(d1 - d2).sum())
    convergence = 1.0 - tv_dist  # higher = more stable

    # 4. ToM prediction accuracy (only if tom_preds present)
    tom_acc = []
    for ep in episodes:
        for s in ep:
            for preds in s["tom_preds"]:
                if not preds:
                    continue
                for tid, pred in preds.items():
                    idx = tid - 1 if isinstance(tid, int) else int(tid) - 1
                    if 0 <= idx < len(s["actions"]):
                        tom_acc.append(1.0 if pred == s["actions"][idx] else 0.0)
    if tom_acc:
        tom_accuracy = float(np.mean(tom_acc))
        tom_lo, tom_hi = _bootstrap_ci(tom_acc, n_boot, 0.95, rng)
    else:
        tom_accuracy = float("nan")
        tom_lo = tom_hi = float("nan")

    # 5. Gated talk+ToM arbitration metrics (only if gate_decisions present)
    gate_trust, gated_acc, signal_acc, tom_belief_acc = [], [], [], []
    dp_conflict, dp_intervened = [], []
    for ep in episodes:
        for s in ep:
            gates = s.get("gate_decisions")
            if not gates:
                continue
            acts = s["actions"]
            for ag_gates in gates:
                for tid, d in ag_gates.items():
                    idx = tid - 1 if isinstance(tid, int) else int(tid) - 1
                    if not (0 <= idx < len(acts)):
                        continue
                    if d.get("trusted") is not None:
                        gate_trust.append(1.0 if d["trusted"] else 0.0)
                    if d.get("gated_pred") is not None:
                        gated_acc.append(1.0 if d["gated_pred"] == acts[idx] else 0.0)
                    if d.get("signal") is not None:
                        signal_acc.append(1.0 if d["signal"] == acts[idx] else 0.0)
                    if d.get("tom_pred") is not None:
                        tom_belief_acc.append(1.0 if d["tom_pred"] == acts[idx] else 0.0)
                    # diversity-preserving gate: track conflict/intervention
                    sig = d.get("signal")
                    pred = d.get("tom_pred")
                    if sig is not None and pred is not None:
                        is_conflict = 1.0 if sig != pred else 0.0
                        dp_conflict.append(is_conflict)
                        # intervened = conditioned action on belief (conflict)
                        dp_intervened.append(is_conflict)

    return {
        "perspective_diversity": perspective_div,
        "cooperation_payoff": coop_payoff,
        "cooperation_payoff_ci": [payoff_lo, payoff_hi],
        "equilibrium_convergence": convergence,
        "tom_prediction_accuracy": tom_accuracy,
        "tom_prediction_accuracy_ci": [tom_lo, tom_hi],
        "n_episodes": len(episodes),
        "action_dist_agent1": dists[0].tolist() if dists else [],
        "per_episode_payoffs": per_ep_payoffs,
        "gate_trust_rate": float(np.mean(gate_trust)) if gate_trust else float("nan"),
        "gated_prediction_accuracy": float(np.mean(gated_acc)) if gated_acc else float("nan"),
        "signal_accuracy": float(np.mean(signal_acc)) if signal_acc else float("nan"),
        "tom_belief_accuracy_in_gated": float(np.mean(tom_belief_acc)) if tom_belief_acc else float("nan"),
        "dp_conflict_rate": float(np.mean(dp_conflict)) if dp_conflict else float("nan"),
        "dp_intervention_rate": float(np.mean(dp_intervened)) if dp_intervened else float("nan"),
    }


# =============================================================================
# Config presets for the 4-cell matrix
# =============================================================================

DEFAULT_MODELS_HET = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
DEFAULT_MODEL_HOMO = "Qwen/Qwen2.5-7B-Instruct"

def make_matrix_configs(args):
    """Return configs for the experimental matrix. Base 4 cells (hom/het x
    noToM/ToM) plus, if --extend, the round-2 cells:
      cheap-talk channel (Madmoun & Lahlou 2025, EACL 2026): +talk variants
      adaptive ToM (Mu et al. 2026, AAAI 2026): +atom variants
    Full extended matrix = 4 base + 4 talk + 2 atom = 10 cells."""
    base = {
        "game": args.game,
        "n_agents": args.n_agents,
        "horizon": args.horizon,
        "memory": args.memory,
        "models_homo": args.model_homo,
        "models_het": args.models_het,
        "temp_homo": args.temp_homo,
        "temps_het": args.temps_het,
        "roles": [f"player{i+1}" for i in range(args.n_agents or 2)],
        "tom_order": args.tom_order,
        "seed": args.seed,
        "api_base": getattr(args, "api_base", None),
        "api_key": getattr(args, "api_key", None),
    }
    cells = [
        # --- base 4-cell matrix (round 1) ---
        ("hom_notom",  dict(homogeneous=True,  use_tom=False, use_talk=False, adaptive_tom=False)),
        ("hom_tom",    dict(homogeneous=True,  use_tom=True,  use_talk=False, adaptive_tom=False)),
        ("het_notom",  dict(homogeneous=False, use_tom=False, use_talk=False, adaptive_tom=False)),
        ("het_tom",    dict(homogeneous=False, use_tom=True,  use_talk=False, adaptive_tom=False)),
    ]
    if getattr(args, "extend", False):
        cells += [
            # --- cheap-talk variants (Madmoun & Lahlou 2025) ---
            ("hom_notom_talk", dict(homogeneous=True,  use_tom=False, use_talk=True,  adaptive_tom=False)),
            ("hom_tom_talk",   dict(homogeneous=True,  use_tom=True,  use_talk=True,  adaptive_tom=False)),
            ("het_notom_talk", dict(homogeneous=False, use_tom=False, use_talk=True,  adaptive_tom=False)),
            ("het_tom_talk",   dict(homogeneous=False, use_tom=True,  use_talk=True,  adaptive_tom=False)),
            # --- adaptive ToM variants (Mu et al. 2026) ---
            ("hom_atom",  dict(homogeneous=True,  use_tom=True,  use_talk=False, adaptive_tom=True)),
            ("het_atom",  dict(homogeneous=False, use_tom=True,  use_talk=False, adaptive_tom=True)),
            # --- combined: heterogeneity + cheap-talk + A-ToM (round-2 full method) ---
            ("het_atom_talk", dict(homogeneous=False, use_tom=True,  use_talk=True,  adaptive_tom=True)),
            # --- Round-3: gated talk+ToM arbitration. Cheap-talk produces a
            #     public intent; ToM judges whether the signal is trustworthy;
            #     if trusted (consistent with ToM belief OR historically
            #     reliable signaler) the agent follows the signal, else falls
            #     back to the ToM belief. This resolves the belief-signal
            #     interference that made naive ToM+talk fail. ---
            ("het_gated_talk_tom",  dict(homogeneous=False, use_tom=True, use_talk=True,
                                         adaptive_tom=False, gated_talk_tom=True)),
            # --- Round-3 full method: gated arbitration + improved A-ToM ---
            ("het_gated_atom_talk", dict(homogeneous=False, use_tom=True, use_talk=True,
                                         adaptive_tom=True,  gated_talk_tom=True)),
            # --- Round-4: diversity-preserving gating. The core innovation.
            #     Same gating mechanism as Round-3, but the gate ONLY
            #     intervenes (conditions the action on the gated belief)
            #     when there is a signal-belief CONFLICT. When signal and
            #     ToM belief agree, agents choose independently — preserving
            #     the cognitive diversity that heterogeneity provides. This
            #     targets the Round-3 failure where gating compressed
            #     diversity to ~0 while not beating the baseline. ---
            ("het_dp_gated_talk_tom",
             dict(homogeneous=False, use_tom=True, use_talk=True,
                  adaptive_tom=False, gated_talk_tom=True,
                  diversity_preserving_gate=True)),
            ("het_dp_gated_atom_talk",
             dict(homogeneous=False, use_tom=True, use_talk=True,
                  adaptive_tom=True, gated_talk_tom=True,
                  diversity_preserving_gate=True)),
        ]
    out = []
    for name, diff in cells:
        cfg = dict(base)
        cfg.update(diff)
        cfg["cell_name"] = name
        out.append(cfg)
    # --- optional cell subset filter (for focused round-3 runs) ---
    want = getattr(args, "cells", None)
    if want:
        want = set(want)
        out = [c for c in out if c["cell_name"] in want]
    return out


# =============================================================================
# Main: run one config or the full matrix
# =============================================================================

def run_config(cfg, n_episodes, out_dir, log_every=10):
    cell = cfg["cell_name"]
    t0 = time.time()
    print(f"\n[{cell}] building agents (homogeneous={cfg['homogeneous']}, "
          f"tom={cfg['use_tom']})...")
    game, agents = build_agents(cfg)
    horizon = cfg["horizon"]
    memory = cfg["memory"]

    episodes = []
    cell_dir = os.path.join(out_dir, cell)
    os.makedirs(cell_dir, exist_ok=True)
    traj_path = os.path.join(cell_dir, "trajectories.jsonl")
    with open(traj_path, "w") as ftraj:
        for ep_idx in range(n_episodes):
            ep = run_episode(game, agents, horizon, memory)
            episodes.append(ep)
            ftraj.write(json.dumps({"episode": ep_idx, "steps": ep}) + "\n")
            if (ep_idx + 1) % log_every == 0:
                elapsed = time.time() - t0
                print(f"[{cell}] ep {ep_idx+1}/{n_episodes}  "
                      f"({elapsed:.0f}s)")
    metrics = compute_metrics(episodes, game, n_boot=2000, seed=cfg.get("seed", 0))
    metrics["cell"] = cell
    metrics["wall_time_s"] = time.time() - t0
    metrics["config"] = {k: v for k, v in cfg.items()
                         if k not in ("models_het", "temps_het", "roles")}
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    pc = metrics.get("cooperation_payoff_ci", [float("nan"), float("nan")])
    print(f"[{cell}] metrics: diversity={metrics['perspective_diversity']:.4f} "
          f"payoff={metrics['cooperation_payoff']:.4f} "
          f"(CI[{pc[0]:.3f},{pc[1]:.3f}]) "
          f"conv={metrics['equilibrium_convergence']:.4f} "
          f"tom_acc={metrics['tom_prediction_accuracy']}")
    # free GPU memory between cells so the 4-cell matrix does not accumulate
    # models within one process (science is unchanged; only memory hygiene)
    try:
        del agents
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass
    return metrics


def main():
    ap = argparse.ArgumentParser(
        description="HetToM Layer-1 baseline: LLM-as-Agent Reasoning Trap test")
    ap.add_argument("--config", type=str, default=None,
                    help="path to a single YAML config (overrides matrix)")
    ap.add_argument("--matrix", action="store_true",
                    help="run the full 4-cell 2x2 matrix")
    ap.add_argument("--game", type=str, default="stag_hunt",
                    choices=list(GAMES.keys()))
    ap.add_argument("--n_agents", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=5,
                    help="rounds per episode")
    ap.add_argument("--memory", type=int, default=2,
                    help="rounds of history visible to agents")
    ap.add_argument("--episodes", type=int, default=50,
                    help="episodes per cell")
    ap.add_argument("--model_homo", type=str, default=DEFAULT_MODEL_HOMO)
    ap.add_argument("--models_het", type=str, nargs="+",
                    default=DEFAULT_MODELS_HET)
    ap.add_argument("--temp_homo", type=float, default=0.7)
    ap.add_argument("--temps_het", type=float, nargs="+",
                    default=[0.5, 0.8, 1.0])
    ap.add_argument("--tom_order", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one or more seeds; each runs a full matrix")
    ap.add_argument("--out_dir", type=str,
                    default="results/hettom_layer1")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--mock", action="store_true",
                    help="use random agents instead of LLMs (for fast "
                         "end-to-end pipeline testing without GPU/models)")
    ap.add_argument("--extend", action="store_true",
                    help="run the extended round-2/3 matrix: base 4 cells + "
                         "cheap-talk variants (Madmoun 2025) + adaptive-ToM "
                         "variants (Mu 2026) + combined het+atom+talk + "
                         "round-3 gated talk+ToM arbitration cells")
    ap.add_argument("--cells", type=str, nargs="+", default=None,
                    help="subset of cell names to run (filters the matrix); "
                         "useful for focused round-3 runs without re-running "
                         "all 13 cells")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mock:
        # replace LLM generation with random to test the pipeline quickly
        _install_mock()

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("cell_name", "single")
        run_config(cfg, args.episodes, args.out_dir, args.log_every)
        return

    if not args.matrix:
        ap.error("provide --config YAML or --matrix")

    # multi-seed: each seed runs a full 4-cell matrix into seed_<n>/
    all_seed_metrics = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        args.seed = seed
        configs = make_matrix_configs(args)
        seed_metrics = []
        for cfg in configs:
            cfg["seed"] = seed
            m = run_config(cfg, args.episodes, seed_dir, args.log_every)
            m["seed"] = seed
            seed_metrics.append(m)
        all_seed_metrics.extend(seed_metrics)
        # per-seed summary
        _write_summary(seed_metrics,
                       os.path.join(seed_dir, "summary.csv"))

    # cross-seed aggregate summary
    _write_summary(all_seed_metrics,
                   os.path.join(args.out_dir, "summary_all_seeds.csv"))
    _print_summary(all_seed_metrics)


def _write_summary(metrics_list, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "cell", "perspective_diversity",
                    "cooperation_payoff", "payoff_ci_lo", "payoff_ci_hi",
                    "equilibrium_convergence", "tom_prediction_accuracy",
                    "tom_ci_lo", "tom_ci_hi", "n_episodes", "wall_time_s"])
        for m in metrics_list:
            pc = m.get("cooperation_payoff_ci", [float("nan"), float("nan")])
            tc = m.get("tom_prediction_accuracy_ci", [float("nan"), float("nan")])
            w.writerow([m.get("seed", ""), m["cell"],
                        m["perspective_diversity"], m["cooperation_payoff"],
                        pc[0], pc[1], m["equilibrium_convergence"],
                        m["tom_prediction_accuracy"], tc[0], tc[1],
                        m["n_episodes"], f"{m['wall_time_s']:.1f}"])


def _print_summary(metrics_list):
    print("\n=== Matrix summary (all seeds) ===")
    print(f"{'seed':>5} {'cell':<12} {'diversity':>10} {'payoff':>8} "
          f"{'conv':>8} {'tom_acc':>8}")
    for m in metrics_list:
        ta = m["tom_prediction_accuracy"]
        ta_s = f"{ta:.3f}" if isinstance(ta, float) and ta == ta else "nan"
        print(f"{m.get('seed',''):>5} {m['cell']:<12} "
              f"{m['perspective_diversity']:>10.4f} "
              f"{m['cooperation_payoff']:>8.3f} "
              f"{m['equilibrium_convergence']:>8.3f} {ta_s:>8}")


# =============================================================================
# Mock mode: random agents for pipeline testing without GPU/models
# =============================================================================

def _install_mock():
    """Monkeypatch LLMAgent._generate to return random plausible text,
    so the full pipeline (env -> prompt -> parse -> metrics) can be tested
    on CPU in seconds before committing GPU hours."""
    def mock_generate(self, prompt, max_new_tokens=8):
        if "JSON" in prompt:
            # produce a JSON dict of teammate predictions.
            # parse teammate ids from the prompt to size the output.
            preds = {}
            for tok in prompt.replace(",", " ").split():
                if tok.isdigit():
                    tid = int(tok)
                    if tid != self.agent_id and tid not in preds:
                        preds[tid] = self._rng.randrange(2)
            return json.dumps({str(k): v for k, v in preds.items()})
        return str(self._rng.randrange(2))
    LLMAgent._generate = mock_generate
    print("[mock] LLM generation replaced with random (no GPU/models needed)")


if __name__ == "__main__":
    main()
