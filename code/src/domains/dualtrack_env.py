"""Dual-track skill-chain: the Phi-FIDELITY knob rho for the dense-vs-sparse DIVERGENCE study.

Phase 0 of the open-ended exploration "under what conditions do dense and sparse RL training
stably diverge?" (supersedes the H1-proof framing). A goal is a REAL **spine** (all ``m_r`` hops
required; terminal success = spine complete) plus a decoupled **decoy** track (``m_d`` hops that
raise *reported* Phi but are NOT on the success path). Reported potential

    Phi_scored = rho * (spine_done / m_r) + (1 - rho) * (decoy_done / m_d)

while success = ``1[spine_done == m_r]``, independent of rho.

**Why structural, not statistical.** Under a FINITE turn budget ``T`` each turn advances at most
one hop, so climbing the decoy has a real OPPORTUNITY COST -- turns not spent on the spine. This
is the structural Goodhart mechanism behind EXP-2026W32-001 (dense raised process-Phi but LOWERED
terminal ASR), NOT a statistically-decoupled counter (which is just noise -> an A2-flavoured null).

**Orthogonality.** rho reweights only which cleared hops the *reward* reads; the gates, the
success set, and per-hop difficulty are all independent of rho. So sweeping rho holds task
difficulty fixed and moves only the reward's fidelity to terminal success -- the axis that sets
the SIGN of the dense-sparse gap. (Reported ``base_success`` invariance across rho is the CPU
diagnostic that this holds; see the goldens.)

Reward wiring (no change to mt_grpo needed): feed **dense** the ``phi_scored`` trace (r_t=Delta
Phi_scored) and **sparse** the ``phi_true`` trace with tau=1 (phi_true hits 1.0 exactly when the
spine completes = success, so sparse fires on TRUE success, never on a decoy-driven Phi crossing).

Reuses ``skillchain_env`` primitives (``SkillChainSpec``, ``build_chain``, ``hop_transform``,
``phi_of_hop``). The single-track env is the ``m_d == 0`` special case.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .skillchain_env import SkillChainSpec, build_chain, phi_of_hop


@dataclass(frozen=True)
class DualTrackSpec:
    """A dual-track goal = a spine chain (required for success) + a decoy chain (raises reported
    Phi only). Both are ordinary :class:`SkillChainSpec` chains built from disjoint tools so an
    action unambiguously targets one track."""

    id: str
    spine: SkillChainSpec
    decoy: SkillChainSpec
    split: str = "train"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if set(self.spine.tools) & set(self.decoy.tools):
            raise ValueError(f"{self.id}: spine and decoy must use disjoint tools")

    @property
    def m_r(self) -> int:
        return self.spine.m

    @property
    def m_d(self) -> int:
        return self.decoy.m


def build_dualtrack(seed_key: str, *, m_r: int, m_d: int, n_distractors: int,
                    tool_pool: tuple[str, ...], split: str = "train") -> DualTrackSpec:
    """Build a dual-track goal from a string seed. Spine and decoy draw from disjoint halves of
    ``tool_pool`` so their tools never collide. Both use the same per-hop distractor count and the
    same fixed ``hop_transform`` skill -- decoy hops are just as clearable as spine hops (only the
    reward's attention differs), which is what makes a decoy genuinely divert a Phi-chasing policy."""
    if m_r < 1:
        raise ValueError("m_r must be >= 1")
    if m_d < 0:
        raise ValueError("m_d must be >= 0")
    if m_r + m_d > len(tool_pool):
        raise ValueError(f"m_r+m_d={m_r + m_d} exceeds tool_pool size {len(tool_pool)}")
    half = len(tool_pool) // 2
    spine_pool = tool_pool[:max(m_r, half)]
    decoy_pool = tool_pool[max(m_r, half):]
    if m_d > len(decoy_pool):
        # fall back to a clean disjoint split by index parity if the halves are too small
        spine_pool = tool_pool[0::2]
        decoy_pool = tool_pool[1::2]
    spine = build_chain(f"{seed_key}|spine", m=m_r, n_distractors=n_distractors,
                        tool_pool=tuple(spine_pool), split=split)
    if m_d == 0:
        decoy = build_chain(f"{seed_key}|decoy", m=1, n_distractors=n_distractors,
                            tool_pool=tuple(decoy_pool), split=split)
        # a length-1 placeholder decoy that is simply never used when m_d == 0
        return DualTrackSpec(id=seed_key, spine=spine, decoy=decoy, split=split,
                             meta={"m_r": m_r, "m_d": 0, "n_distractors": n_distractors,
                                   "decoy_disabled": True})
    decoy = build_chain(f"{seed_key}|decoy", m=m_d, n_distractors=n_distractors,
                        tool_pool=tuple(decoy_pool), split=split)
    return DualTrackSpec(id=seed_key, spine=spine, decoy=decoy, split=split,
                         meta={"m_r": m_r, "m_d": m_d, "n_distractors": n_distractors})


class DualTrackEnv:
    """Deterministic dual-track capability gate. State = (spine_hop, decoy_hop). Each ``step``
    advances the spine if the action matches the next spine hop, else the decoy if it matches the
    next decoy hop, else nothing (a wasted call). success = spine complete. Candidate order at each
    hop is a deterministic function of (chain-id, track, hop) -> CRN across arms by construction."""

    def __init__(self, spec: DualTrackSpec):
        self.spec = spec
        self.spine_hop = 0
        self.decoy_hop = 0
        self.n_calls = 0
        self.attempts: set = set()          # distinct (tool,key) attempts -> the GAMEABLE surface proxy

    # -- state ---------------------------------------------------------------------------------
    @property
    def decoy_enabled(self) -> bool:
        return not self.spec.meta.get("decoy_disabled", False) and self.spec.m_d > 0

    @property
    def success(self) -> bool:
        return self.spine_hop >= self.spec.m_r

    @property
    def done(self) -> bool:
        return self.success

    def phi_true(self) -> float:
        return phi_of_hop(self.spine_hop, self.spec.m_r)

    def phi_decoy(self) -> float:
        return phi_of_hop(self.decoy_hop, self.spec.m_d) if self.decoy_enabled else 0.0

    def phi_scored(self, rho: float) -> float:
        """rho * (spine fraction) + (1-rho) * (decoy fraction). rho=1 -> faithful (= phi_true);
        rho=0 -> fully decoupled (= decoy only)."""
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho must be in [0,1]")
        if not self.decoy_enabled:
            return self.phi_true()
        return rho * self.phi_true() + (1.0 - rho) * self.phi_decoy()

    def phi_proxy(self) -> float:
        """GAMEABLE surface proxy (DISC-2026W33-001): fraction of the m_r-length budget covered by DISTINCT
        (tool,key) attempts, capped at 1. A policy MAXIMIZES it by spraying m_r distinct plausible attempts --
        including wasted/decoy ones -- so it is inflatable OFF the true path and its max leaves the spine
        floored. Decoupled from success by construction (not gated on spine advancement)."""
        return min(1.0, len(self.attempts) / self.spec.m_r) if self.spec.m_r else 0.0

    # -- observation (for the eventual text rollout; the mock reads keys directly) --------------
    def _candidates(self, chain: SkillChainSpec, hop: int) -> list[str]:
        key = chain.keys[hop]
        pool = [key, *chain.distractors[hop]]
        random.Random(f"{chain.id}|{hop}|cand").shuffle(pool)
        return pool

    def observation(self) -> dict:
        """Both tracks' next hops (whichever remain), so the policy has a genuine choice of track."""
        obs = {"spine_hop": self.spine_hop, "decoy_hop": self.decoy_hop}
        if self.spine_hop < self.spec.m_r:
            obs["spine"] = {"tool": self.spec.spine.tools[self.spine_hop],
                            "candidates": self._candidates(self.spec.spine, self.spine_hop)}
        if self.decoy_enabled and self.decoy_hop < self.spec.m_d:
            obs["decoy"] = {"tool": self.spec.decoy.tools[self.decoy_hop],
                            "candidates": self._candidates(self.spec.decoy, self.decoy_hop)}
        return obs

    def reset(self) -> dict:
        self.spine_hop = 0
        self.decoy_hop = 0
        self.n_calls = 0
        self.attempts = set()
        return self.observation()

    # -- gate ----------------------------------------------------------------------------------
    def step(self, tool: str, key: str) -> dict:
        """Attempt one hop. Advances the spine if (tool,key) matches the next spine hop; else the
        decoy if it matches the next decoy hop; else nothing. Returns an info dict."""
        self.n_calls += 1
        self.attempts.add((tool, key))          # surface proxy: distinct attempts, gameable off-path
        track = None
        if self.spine_hop < self.spec.m_r and tool == self.spec.spine.tools[self.spine_hop] \
                and key == self.spec.spine.keys[self.spine_hop]:
            self.spine_hop += 1
            track = "spine"
        elif self.decoy_enabled and self.decoy_hop < self.spec.m_d \
                and tool == self.spec.decoy.tools[self.decoy_hop] \
                and key == self.spec.decoy.keys[self.decoy_hop]:
            self.decoy_hop += 1
            track = "decoy"
        return {"advanced": track, "spine_hop": self.spine_hop, "decoy_hop": self.decoy_hop,
                "n_calls": self.n_calls, "success": self.success}


# -- reward-Phi transforms (pure; operate on a realized per-turn phi trace) ---------------------

def shuffle_increments(trace: list[float], rng: random.Random) -> list[float]:
    """Placebo: same final Phi and same MULTISET of per-turn increments, but permuted in TIME so
    the per-turn credit is misaligned with which action actually helped. Isolates 'faithful
    per-turn credit' from 'merely denser gradient': if dense still beats sparse under this, the
    advantage was gradient density, not faithful credit assignment."""
    incs = [trace[0]] + [trace[i] - trace[i - 1] for i in range(1, len(trace))]
    order = list(range(len(incs)))
    rng.shuffle(order)
    perm = [incs[i] for i in order]
    out, acc = [], 0.0
    for x in perm:
        acc += x
        out.append(acc)
    return out


def add_phi_noise(trace: list[float], sigma: float, rng: random.Random) -> list[float]:
    """Additive Gaussian noise on reported Phi (breaks strict monotonicity on purpose -- the sigma
    knob probes how noise-robust the dense credit signal is)."""
    return [t + rng.gauss(0.0, sigma) for t in trace]
