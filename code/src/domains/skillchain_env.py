"""Deterministic capability-gated skill-chain environment (H1 regime-map redesign).

A goal is a chain of ``m`` hops ``t_1 .. t_m``. Each hop ``k`` has a correct *key* that is
revealed only in hop ``k-1``'s observation (hop 1's key is in the goal prompt). Advancing from
state ``k-1`` to ``k`` requires calling ``tool_k`` with the correct key. Gates are
**deterministic** -- there is no aligned LLM victim to refuse -- so

    Phi = (hops completed) / m

is cumulative *by construction* and Phi *tracks value*: reaching hop ``k`` means ``k``
independent gates are cleared and only ``m-k`` remain, so ``P(finish | Phi=k/m)`` is strictly
increasing in ``k``.

**The potential-strength knob is the per-hop distractor count ``D``.** Each observation shows
the true next key alongside ``D`` distractor keys, so a policy that cannot reliably identify the
true key clears a hop with probability ``~1/(D+1)``. With per-hop success ``s`` and independent
gates,

    P(finish | at hop k) = s^(m-k),   lift(Phi=k/m) = P(finish|hop k)/P(finish|hop 0) = (1/s)^k.

Larger ``D`` -> lower ``s`` -> larger lift -> stronger potential. Sweeping ``D`` sweeps the
regime-map x-axis; this is why the knob works, proven on CPU by ``h1_skillchain_potential_sim``.

This module is the reusable core: the same :class:`SkillChainEnv` scores mock-policy rollouts
(the zero-GPU A0 potential check) and, once wired into the RL harness, real-policy rollouts.
Nothing here loads a model or needs a GPU.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


def hop_transform(raw: str) -> str:
    """The fixed per-hop *skill*: reverse the hex suffix of a ``pfx-<hex>`` token.

    The gate accepts ``transform(issued)``, not the issued token itself, and the rule is NOT
    revealed to the policy -- so an untrained model cannot simply copy the visible issued token
    (which is why the earlier copy-only task saturated at s~1, EXP-063). It must either guess
    among the candidates (untrained -> s ~ 1/(D+1), restoring the A0 regime) or learn the
    transform from reward (trained -> s rises). One fixed rule across all hops makes it a single
    learnable, reusable skill."""
    prefix, sep, suffix = raw.partition("-")
    if not sep:
        raise ValueError(f"token {raw!r} has no 'pfx-suffix' shape")
    return f"{prefix}{sep}{suffix[::-1]}"


@dataclass(frozen=True)
class SkillChainSpec:
    """A frozen chain goal. At hop ``k`` the policy is issued ``raw[k]`` and must submit
    ``keys[k] = hop_transform(raw[k])`` (the gate's accepted token), which is shown among
    ``distractors[k]`` -- the transform itself is never shown."""

    id: str
    tools: tuple[str, ...]          # tool name per hop, length m
    raw: tuple[str, ...]            # issued (pre-transform) token per hop, length m
    keys: tuple[str, ...]           # accepted token = hop_transform(raw), length m
    distractors: tuple[tuple[str, ...], ...]  # D distractors per hop, length m
    split: str = "train"            # 'train' | 'calibration' | 'ood'
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int:
        return len(self.tools)

    def __post_init__(self):
        if not (len(self.tools) == len(self.raw) == len(self.keys) == len(self.distractors) >= 1):
            raise ValueError("tools, raw, keys, distractors must share a length >= 1")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError(f"chain {self.id} repeats a tool; hops must be distinguishable")
        for k, (raw, key, dis) in enumerate(zip(self.raw, self.keys, self.distractors)):
            if hop_transform(raw) != key:
                raise ValueError(f"chain {self.id} hop {k}: key is not hop_transform(raw)")
            if key in dis:
                raise ValueError(f"chain {self.id} hop {k}: accepted token appears among distractors")
            if raw in (key, *dis):
                raise ValueError(f"chain {self.id} hop {k}: raw token must not be a candidate")


def phi_of_hop(hop: int, m: int) -> float:
    """Potential at 'completed ``hop`` of ``m`` hops'. Monotone, Phi(0)=0, Phi(m)=1."""
    if not (0 <= hop <= m):
        raise ValueError(f"hop {hop} out of range 0..{m}")
    return hop / m


class SkillChainEnv:
    """Deterministic capability gate over a :class:`SkillChainSpec`.

    ``reset()`` returns the hop-1 candidate set (the goal reveals hop 1's candidates). Each
    ``step(tool, key)`` advances exactly one hop iff ``tool``/``key`` match the current hop's
    correct pair; otherwise the state is unchanged (a wasted call). The candidate order shown at
    each hop is a deterministic function of ``(spec.id, hop)`` so rollouts are reproducible and
    two arms sharing a spec see identical observations (common random numbers by construction).
    """

    def __init__(self, spec: SkillChainSpec):
        self.spec = spec
        self.hop = 0            # hops completed so far (0..m); current target hop is self.hop
        self.done = False
        self.n_calls = 0

    # -- observations -------------------------------------------------------------------------
    def candidates(self, hop: int) -> list[str]:
        """The shuffled key candidates revealed for target ``hop`` (0-indexed): true key +
        distractors, ordered deterministically so the true position is not a fixed slot."""
        key = self.spec.keys[hop]
        pool = [key, *self.spec.distractors[hop]]
        # Deterministic per-(chain, hop) shuffle: reproducible, and independent of arm/policy.
        random.Random(f"{self.spec.id}|{hop}|cand").shuffle(pool)
        return pool

    def observation(self, hop: int) -> dict:
        """Structured observation for target ``hop``: which tool to call and the candidate keys
        (exactly one of which is correct). This is the state a policy must act on."""
        return {"target_hop": hop, "tool": self.spec.tools[hop], "candidates": self.candidates(hop)}

    def reset(self) -> dict:
        self.hop = 0
        self.done = False
        self.n_calls = 0
        return self.observation(0)

    # -- gate ---------------------------------------------------------------------------------
    def step(self, tool: str, key: str) -> tuple[dict | None, float, bool, dict]:
        """Attempt the current target hop. Returns ``(observation_or_None, phi, done, info)``.

        Advances one hop iff ``(tool, key)`` matches the current target hop's correct pair.
        A wrong tool or wrong key leaves the state unchanged -- capability gating: you cannot
        skip ahead, and you cannot pass a hop whose key you did not recover from the prior hop.
        """
        if self.done:
            return None, phi_of_hop(self.hop, self.spec.m), True, {"already_done": True}
        self.n_calls += 1
        target = self.hop
        correct = tool == self.spec.tools[target] and key == self.spec.keys[target]
        info = {"target_hop": target, "correct": correct, "n_calls": self.n_calls}
        if correct:
            self.hop += 1
            self.done = self.hop >= self.spec.m
            nxt = None if self.done else self.observation(self.hop)
            return nxt, phi_of_hop(self.hop, self.spec.m), self.done, info
        # wrong: state unchanged, re-present the same target hop
        return self.observation(target), phi_of_hop(self.hop, self.spec.m), False, info

    @property
    def phi(self) -> float:
        return phi_of_hop(self.hop, self.spec.m)

    @property
    def success(self) -> bool:
        return self.hop >= self.spec.m


# -- chain generation -------------------------------------------------------------------------

_TOOL_VERBS = (
    "lookup", "fetch", "decode", "resolve", "unlock", "query", "expand", "trace",
    "derive", "extract", "map", "join", "search", "read", "compose", "route",
)


def _token(rng: random.Random, prefix: str) -> str:
    return prefix + "-" + "".join(rng.choice("0123456789abcdef") for _ in range(6))


def build_chain(seed_key: str, m: int, n_distractors: int, tool_pool: tuple[str, ...],
                split: str = "train") -> SkillChainSpec:
    """Deterministically build one chain from a string seed. ``tool_pool`` supplies the ordered
    set of tools the hops are drawn from (reused across chains -> compositional OOD: a held-out
    chain recombines known tools in an unseen order)."""
    if m < 1 or m > len(tool_pool):
        raise ValueError(f"m={m} must be in 1..{len(tool_pool)} (tool_pool size)")
    if n_distractors < 0:
        raise ValueError("n_distractors must be >= 0")
    rng = random.Random(seed_key)
    tools = tuple(rng.sample(tool_pool, m))
    # Issued tokens whose transform (reversed hex suffix) differs from themselves, so copying the
    # issued token never accidentally passes the gate.
    raw = []
    while len(raw) < m:
        candidate = _token(rng, "tok")
        if hop_transform(candidate) != candidate:
            raw.append(candidate)
    raw = tuple(raw)
    keys = tuple(hop_transform(r) for r in raw)
    # Distractors share the accepted tokens' surface form ('tok-<hex>') and exclude the accepted
    # token and the raw token, so neither guessing-by-format nor copying-the-issued-token works.
    distractors = []
    for k in range(m):
        pool_bad = set()
        while len(pool_bad) < n_distractors:
            d = _token(rng, "tok")
            if d not in (keys[k], raw[k]):
                pool_bad.add(d)
        distractors.append(tuple(sorted(pool_bad)))
    return SkillChainSpec(
        id=seed_key, tools=tools, raw=raw, keys=keys, distractors=tuple(distractors), split=split,
        meta={"n_distractors": n_distractors, "m": m},
    )


def make_tool_pool(size: int) -> tuple[str, ...]:
    """A pool of ``size`` distinct tool names built from a fixed verb list (deterministic)."""
    if size < 1:
        raise ValueError("tool pool size must be >= 1")
    pool = []
    i = 0
    while len(pool) < size:
        verb = _TOOL_VERBS[i % len(_TOOL_VERBS)]
        suffix = "" if i < len(_TOOL_VERBS) else f"_{i // len(_TOOL_VERBS)}"
        pool.append(f"{verb}{suffix}")
        i += 1
    return tuple(pool[:size])
