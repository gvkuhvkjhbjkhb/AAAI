FAILURE_TYPES = {
    "target_miscoordination": "Agents pursue incompatible targets or fail to converge on the same cooperative target.",
    "insufficient_cooperation": "A task requires multiple agents, but agents act alone or remain split.",
    "inefficient_exploration": "Agents spend substantial time far from useful objectives or revisit unproductive regions.",
    "low_value_overcommitment": "Agents overcommit to low-value objectives and ignore more important cooperative objectives.",
    "timeout_near_success": "Agents approach a successful configuration but coordinate too slowly and time out.",
    "unknown": "The trajectory does not clearly match the fixed taxonomy.",
}


def build_failure_prompt(summary):
    taxonomy = "\n".join(f"- {name}: {desc}" for name, desc in FAILURE_TYPES.items())
    return f"""You are diagnosing a failed cooperative multi-agent reinforcement learning episode.
Choose exactly one failure_type from the taxonomy and return valid JSON only.

Taxonomy:
{taxonomy}

Episode summary:
{summary}

Return this JSON schema:
{{
  "failure_type": "one taxonomy key",
  "confidence": 0.0,
  "evidence": "one concise sentence grounded in the summary"
}}
"""
