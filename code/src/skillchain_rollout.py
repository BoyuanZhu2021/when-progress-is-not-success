"""On-policy rollout for the capability-gated skill-chain (H1 regime-map redesign).

Single-agent loop (no LLM victim): the trained policy reads the current step's observation and
emits one action ``ACTION <tool> <token>``; the deterministic :class:`SkillChainEnv` gates it;
``Phi = hops_completed / m``. The skill is *referential recall*: each step's observation issues
the token needed at the NEXT step, so to clear step ``k`` the policy must recall the token issued
at step ``k-1`` and pick it out of ``D+1`` confusable candidates. An untrained policy that does
not track the issued token clears a step with probability ``~1/(D+1)`` (the knob ``D`` sets the
potential strength, validated in ``h1_skillchain_potential_sim``); a trained one learns to copy
it and ``s -> 1``.

Text-level and generation-agnostic (``gen_batch_fn`` is injected), so the whole loop is
unit-testable on CPU with a mock generator before the real 4B policy runs on GPU. Returns
per-trajectory results whose turns carry ``(prompt_ids, resp_ids, phi)`` -- exactly what the
GRPO token-normalized update consumes.
"""
from __future__ import annotations

import re
from collections.abc import Callable

from .domains.skillchain_env import SkillChainEnv, SkillChainSpec

SYSTEM_PROMPT = (
    "You are an agent completing a fixed sequence of tool calls. At each step you are shown the "
    "tool to call, a source token, and a list of candidate tokens. Exactly one candidate is the "
    "VALID token for the step; it is derived from the source token by a fixed rule that is the "
    "same at every step (you must work out the rule). Reply with exactly one line and nothing "
    "else:\nACTION <tool> <token>\n"
    "where <token> is the valid candidate."
)

# gen_batch_fn(batch_messages) -> list of {"text", "prompt_ids"?, "resp_ids"?}
GenBatchFn = Callable[[list], list]

_ACTION_RE = re.compile(r"ACTION\s+(\S+)\s+(\S+)")


def _source_line(spec: SkillChainSpec, next_hop: int) -> str:
    """Issue the raw source token for ``next_hop`` (0-indexed), to be recalled one step later.
    The valid token there will be ``hop_transform(this)`` -- the rule is never stated."""
    if next_hop >= spec.m:
        return "This was the final step; no further token is issued."
    return f"Source token for the NEXT step ({spec.tools[next_hop]}): {spec.raw[next_hop]}"


def render_step(env: SkillChainEnv, hop: int) -> str:
    """Observation for target ``hop``: which tool, the candidate tokens (one valid), and the raw
    source token issued for the following step (recalled and transformed when the policy arrives)."""
    obs = env.observation(hop)
    candidates = "  ".join(obs["candidates"])
    return (
        f"Step {hop + 1} of {env.spec.m}. Call tool `{obs['tool']}` and submit the valid token "
        f"for this step.\nCandidate tokens: {candidates}\n{_source_line(env.spec, hop + 1)}"
    )


def initial_messages(env: SkillChainEnv) -> list[dict]:
    """System + first user turn. The goal prompt issues step 1's source token (nothing precedes it)."""
    spec = env.spec
    task = (
        f"Goal: complete all {spec.m} steps of this tool chain in order.\n"
        f"Source token for step 1 ({spec.tools[0]}): {spec.raw[0]}\n\n"
        + render_step(env, 0)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]


def parse_action(text: str, valid_tools: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Extract ``(tool, token)`` from the policy's output. Tolerant: scans for an ``ACTION`` line;
    if none, falls back to the first valid tool name found plus the next whitespace token."""
    for tool, token in _ACTION_RE.findall(text):
        if tool in valid_tools:
            return tool, token
    # Fallback: first mentioned valid tool, then the next token after it.
    for tool in valid_tools:
        idx = text.find(tool)
        if idx >= 0:
            rest = text[idx + len(tool):].split()
            if rest:
                return tool, rest[0].strip(".,:;`\"'")
    return None, None


def _feedback(env: SkillChainEnv, correct: bool) -> str:
    """Next user message after an attempt: the new observation on success, a retry on failure."""
    if env.success:
        return "All steps complete."
    head = "" if correct else "That token was not accepted for this step. Try again.\n"
    return head + render_step(env, env.hop)


def skillchain_rollout_batch(specs: list[SkillChainSpec], gen_batch_fn: GenBatchFn, *,
                             T: int, tau: float = 1.0) -> list[dict]:
    """Advance one on-policy trajectory per spec in lockstep, batching policy generation each turn.

    Each turn: batch the active trajectories' message lists into ONE ``gen_batch_fn`` call, parse
    each action, step each env, append the resulting observation as feedback. A trajectory ends
    when it reaches ``Phi >= tau`` (full chain) or the turn budget ``T`` is exhausted. Returns a
    per-trajectory dict with ``phi_trace`` and ``turns`` (each carrying prompt_ids/resp_ids for the
    update), the same shape the GRPO loop reads.
    """
    st = [{
        "env": SkillChainEnv(spec), "messages": None, "turns": [], "phi_trace": [],
        "active": True,
    } for spec in specs]
    for s in st:
        s["messages"] = initial_messages(s["env"])

    for _t in range(1, T + 1):
        active = [i for i in range(len(st)) if st[i]["active"]]
        if not active:
            break
        gens = gen_batch_fn([st[i]["messages"] for i in active])
        if len(gens) != len(active):
            raise RuntimeError(f"generator returned {len(gens)} for {len(active)} active trajectories")
        for k, i in enumerate(active):
            s = st[i]
            env = s["env"]
            g = gens[k] if isinstance(gens[k], dict) else {"text": gens[k]}
            raw = g["text"]
            tool, token = parse_action(raw, env.spec.tools)
            target_hop = env.hop
            if tool is None:
                correct = False
            else:
                _obs, _phi, _done, info = env.step(tool, token)
                correct = bool(info["correct"])
            phi = env.phi
            s["turns"].append({
                "turn": _t,
                "prompt_messages": [dict(m) for m in s["messages"]],
                "response": raw,
                "action": {"tool": tool, "token": token, "target_hop": target_hop},
                "correct": correct,
                "phi": phi,
                "prompt_ids": g.get("prompt_ids"),
                "resp_ids": g.get("resp_ids"),
            })
            s["phi_trace"].append(phi)
            if phi >= tau:
                s["active"] = False
            else:
                s["messages"] = s["messages"] + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": _feedback(env, correct)},
                ]
    return [{
        "turns": s["turns"],
        "phi_trace": s["phi_trace"],
        "success": s["env"].success,
        "max_phi": max(s["phi_trace"]) if s["phi_trace"] else 0.0,
        "n_turns": len(s["turns"]),
        "final_hop": s["env"].hop,
    } for s in st]
