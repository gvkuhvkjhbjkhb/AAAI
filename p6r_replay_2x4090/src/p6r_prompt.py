"""
p6r_prompt.py -- the P6R prompt and action parser, isolated and versioned.

Why this file is separate.  P5c established that the prompt and the parser move
target fidelity by at most 0.002, so they are NOT the cause of the low-fidelity
cell.  That conclusion only holds if the replay uses the same prompt and parser
the original run did.  This module therefore does two things:

  1. It can MOUNT the archived P6R implementation, if you point it at the
     original code directory.  That is the faithful path and the default.
  2. Failing that, it falls back to a reconstruction and records a loud flag
     plus a SHA-256 of whichever source was used, so the run manifest always
     says which one was in force.

Never silently reconstruct.  `resolve_prompt_impl()` returns the provenance and
`replay_runner.py` writes it into RUN_MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# reconstruction (used only when the archived module is not supplied)
# ---------------------------------------------------------------------------

SYSTEM = "Reply exactly ACTION: A or ACTION: B. No explanation."


def matrix_text(payoff, role: int) -> str:
    """Role-relative payoff table.  Role 1 sees the transpose, with its own
    payoff first -- the convention the campaigns used."""
    rows = []
    for own in range(2):
        cells = []
        for other in range(2):
            pair = payoff[own][other] if role == 0 else payoff[other][own]
            if role == 0:
                a, b = pair[0], pair[1]
            else:
                a, b = pair[1], pair[0]
            cells.append(f"({a:+.2f},{b:+.2f})")
        rows.append(" | ".join(cells))
    return "Other A | Other B\nSelf A " + rows[0] + "\nSelf B " + rows[1]


def build_prompt(payoff, target, role: int, history, arm: str, memory: int):
    """The two-message chat prompt for one decision."""
    own = "Row" if role == 0 else "Column"
    assigned = "AB"[target[role]]
    hist = ("none" if not history else
            "; ".join(f"you={'AB'[s['actions'][role]]}, "
                      f"other={'AB'[s['actions'][1 - role]]}"
                      for s in history[-memory:]))
    cue = ""
    if arm == "Gated":
        # P6R's advisory cue: transparent, non-binding, states the joint target.
        cue = ("Advisory: the transparent team-payoff rule recommends "
               f"Action {assigned}. It is not forced.")
    user = (f"You are {own} in a simultaneous two-action game. "
            f"Entries are (your payoff, other payoff).\n"
            f"{matrix_text(payoff, role)}\n"
            f"Recent: {hist}\n{cue}\nChoose now.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


class ParseFailure(RuntimeError):
    pass


def parse_action(text: str) -> int:
    """The archived parser.  P5c showed the parser contrast moves fidelity by
    exactly 0.000, so this must not be 'improved' during a replay."""
    if not text or not text.strip():
        raise ParseFailure(f"empty model response: {text!r}")
    upper = text.upper()
    for token, val in (("ACTION: A", 0), ("ACTION:A", 0),
                       ("ACTION: B", 1), ("ACTION:B", 1)):
        if token in upper:
            return val
    m = re.search(r"(?i)self\s*:?\s*(a|b)", upper)
    if m:
        return 0 if m.group(1) == "A" else 1
    m2 = re.search(r"(?i)\b(a|b)\b", upper)
    if m2:
        return 0 if m2.group(1) == "A" else 1
    stripped = upper.strip().replace(".", "")
    if stripped in {"A", "0"}:
        return 0
    if stripped in {"B", "1"}:
        return 1
    raise ParseFailure(f"unparseable model response: {text!r}")


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_prompt_impl(archived_dir: str | None) -> dict[str, Any]:
    """Return {build_prompt, parse_action, provenance}.

    archived_dir: directory containing the original P6R runner module (any file
    exposing `prompt`/`build_prompt` and `parse_action`).  Pass None to use the
    reconstruction -- which is recorded as such.
    """
    if archived_dir:
        d = Path(archived_dir)
        cands = [p for p in d.rglob("*.py")
                 if p.stem in {"p6r_runner", "p6_runner", "run_p6r_pair3_campaign",
                               "p7_runner", "p6r", "runner"}]
        for p in sorted(cands):
            spec = importlib.util.spec_from_file_location(f"_archived_{p.stem}", p)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:                       # noqa: BLE001 - best effort
                continue
            pf = getattr(mod, "build_prompt", None) or getattr(mod, "prompt", None)
            pa = getattr(mod, "parse_action", None)
            if pf and pa:
                return {"build_prompt": pf, "parse_action": pa,
                        "provenance": {"source": "archived", "file": str(p),
                                       "sha256": _sha(p), "reconstructed": False}}
        raise SystemExit(
            f"--p6r-code-dir={archived_dir} given but no module exposing both a "
            f"prompt builder and parse_action was found. Either fix the path or "
            f"pass --allow-reconstructed-prompt explicitly.")

    return {"build_prompt": build_prompt, "parse_action": parse_action,
            "provenance": {"source": "reconstructed",
                           "file": str(Path(__file__).resolve()),
                           "sha256": _sha(Path(__file__)),
                           "reconstructed": True,
                           "warning": "Prompt/parser reconstructed, not the "
                                      "archived originals. P5c's 'prompt and "
                                      "parser are not the cause' result is only "
                                      "transferable to this run if the two agree."}}
