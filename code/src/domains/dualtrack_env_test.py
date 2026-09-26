"""CPU goldens for the dual-track env (Phase 0 of the dense-vs-sparse divergence study).

Asserts the properties the phase-diagram attribution depends on: rho-orthogonality (rho moves only
reported Phi, never the gates/success set), correct dual-track dynamics + opportunity cost, the
reward wiring (sparse fires on TRUE success, never on a decoy-driven Phi crossing), the shuffled-Phi
placebo invariants, CRN candidate determinism, and the mt_grpo starvation identity.

Run:  python -m code.src.domains.dualtrack_env_test    (or: python code/src/domains/dualtrack_env_test.py)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.domains.dualtrack_env import (  # noqa: E402
    DualTrackEnv, build_dualtrack, shuffle_increments, add_phi_noise)
from src.domains.skillchain_env import make_tool_pool  # noqa: E402
from src.mt_grpo import per_turn_rewards, group_advantages, frac_zero_gradient  # noqa: E402

POOL = make_tool_pool(16)


def _clear(env, chain, hop_attr):
    """Submit the correct (tool,key) for the next hop of `chain`."""
    hop = getattr(env, hop_attr)
    return env.step(chain.tools[hop], chain.keys[hop])


def test_build_disjoint():
    spec = build_dualtrack("g0", m_r=4, m_d=3, n_distractors=3, tool_pool=POOL)
    assert spec.m_r == 4 and spec.m_d == 3
    assert not (set(spec.spine.tools) & set(spec.decoy.tools)), "spine/decoy tools must be disjoint"


def test_dynamics_and_success():
    spec = build_dualtrack("g1", m_r=3, m_d=2, n_distractors=3, tool_pool=POOL)
    env = DualTrackEnv(spec); env.reset()
    # wrong action advances nothing
    info = env.step("nope", "nope")
    assert info["advanced"] is None and env.spine_hop == 0 and env.decoy_hop == 0
    # clear a decoy hop -> decoy advances, still no success
    _clear(env, spec.decoy, "decoy_hop")
    assert env.decoy_hop == 1 and not env.success
    # clear all spine hops -> success exactly at the last spine hop
    for _ in range(spec.m_r):
        assert not env.success
        _clear(env, spec.spine, "spine_hop")
    assert env.success and env.spine_hop == 3


def test_rho_orthogonality():
    """rho changes ONLY phi_scored; the gate dynamics and success set are identical for all rho."""
    spec = build_dualtrack("g2", m_r=3, m_d=3, n_distractors=3, tool_pool=POOL)
    # a fixed action script -> identical (spine_hop, decoy_hop, success) trajectory regardless of rho
    def run_script():
        env = DualTrackEnv(spec); env.reset(); trace = []
        script = ["decoy", "spine", "decoy", "spine", "spine", "decoy"]
        for tr in script:
            chain = spec.decoy if tr == "decoy" else spec.spine
            _clear(env, chain, f"{tr}_hop")
            trace.append((env.spine_hop, env.decoy_hop, env.success))
        return trace
    base = run_script()
    assert base == run_script(), "dynamics must be deterministic"
    # phi_scored at rho=1 equals phi_true; at rho=0 equals phi_decoy; success independent of rho
    env = DualTrackEnv(spec); env.reset()
    _clear(env, spec.spine, "spine_hop"); _clear(env, spec.decoy, "decoy_hop")
    assert abs(env.phi_scored(1.0) - env.phi_true()) < 1e-12
    assert abs(env.phi_scored(0.0) - env.phi_decoy()) < 1e-12
    mid = env.phi_scored(0.5)
    assert abs(mid - 0.5 * (env.phi_true() + env.phi_decoy())) < 1e-12
    # base_success (a fixed low-skill policy) is invariant to rho -- rho never touches success
    def base_success(rho, seed):
        rng = random.Random(seed); succ = 0
        for e in range(300):
            env = DualTrackEnv(build_dualtrack(f"bs|{e}", m_r=3, m_d=3, n_distractors=3,
                                               tool_pool=POOL)); env.reset()
            for _t in range(6):
                if env.success: break
                # rho-blind fixed policy: attempt spine w.p. 0.5 else decoy, clear w.p. 0.25
                tr = "spine" if rng.random() < 0.5 else "decoy"
                chain = spec.spine if tr == "spine" else spec.decoy
                hop = env.spine_hop if tr == "spine" else env.decoy_hop
                if hop < (env.spec.m_r if tr == "spine" else env.spec.m_d) and rng.random() < 0.25:
                    _clear(env, chain, f"{tr}_hop")
                else:
                    env.step("x", "x")
            succ += int(env.success)
        return succ
    # same seed => identical trajectories => identical success count for any rho (rho unused in the loop)
    assert base_success(1.0, 7) == base_success(0.0, 7), "base_success must not depend on rho"


def test_reward_wiring_sparse_on_true_success():
    """sparse must fire on TRUE success (spine complete), NEVER on a decoy-driven scored crossing."""
    # a trajectory that maxes the decoy but never completes the spine
    true_tr = [0.0, 0.0, 0.0, 0.0]          # spine never advances -> phi_true stays 0
    decoy_tr = [0.5, 1.0, 1.0, 1.0]         # decoy completes -> phi_decoy hits 1
    scored0 = [0.0 * t + 1.0 * d for t, d in zip(true_tr, decoy_tr)]  # rho=0
    dense_r = per_turn_rewards(scored0, tau=1.0, arm="dense")
    sparse_r = per_turn_rewards(true_tr, tau=1.0, arm="sparse")
    assert sum(dense_r) == 1.0, "dense telescopes to scored Phi_T (=1 via decoy)"
    assert sum(sparse_r) == 0.0, "sparse must NOT fire: the spine (true success) never completed"
    # now a real success: spine completes at turn 3
    true2 = [0.33, 0.67, 1.0]
    sparse2 = per_turn_rewards(true2, tau=1.0, arm="sparse")
    assert sparse2 == [0, 0, 1.0], "sparse fires exactly once at true-success"


def test_three_phi_poles_decoupled_from_flag():
    """DISC-2026W33-001: gameable (phi_proxy) and misaligned (phi_scored, rho<1) must be maxable OFF the true
    path (Phi high, flag=0), while faithful (phi_true) is the flag by construction -- the 'Phi-maxed-but-flag-
    failed' non-emptiness that makes the terminal DV genuinely decoupled (screen gate G4)."""
    spec = build_dualtrack("dc", m_r=4, m_d=2, n_distractors=2, tool_pool=POOL)
    env = DualTrackEnv(spec); env.reset()                              # GAMEABLE: spray m_r distinct WRONG attempts
    for i in range(spec.m_r):
        env.step("nope", f"wrong{i}")
    assert env.phi_proxy() == 1.0 and env.phi_true() == 0.0 and not env.success
    env2 = DualTrackEnv(spec); env2.reset()                            # MISALIGNED: clear the decoy only
    for h in range(spec.m_d):
        env2.step(spec.decoy.tools[h], spec.decoy.keys[h])
    assert env2.phi_decoy() == 1.0 and env2.phi_true() == 0.0 and not env2.success
    assert abs(env2.phi_scored(0.5) - 0.5) < 1e-9 and env2.phi_scored(1.0) == 0.0   # rho=1 faithful ignores decoy
    env3 = DualTrackEnv(spec); env3.reset()                            # FAITHFUL: only clearing the spine = flag
    for h in range(spec.m_r):
        env3.step(spec.spine.tools[h], spec.spine.keys[h])
    assert env3.phi_true() == 1.0 and env3.success


def test_additive_dual_goodhart_immune():
    """dense_additive_dual anchored on phi_true shares the TRUE-success optimum for ANY shaping pole (the
    dualtrack_env:23 trap fix): a failed anchor with a MAXED misaligned shape still sums to 0, not the shape max."""
    from src.mt_grpo import dense_additive_dual
    assert abs(sum(dense_additive_dual([0.0, 0.5, 0.5], [1.0, 1.0, 1.0]))) < 1e-9   # anchor fails -> 0
    assert abs(sum(dense_additive_dual([0.5, 1.0], [0.2, 0.4])) - 1.0) < 1e-9        # anchor wins -> 1


def test_shuffle_increments_invariants():
    trace = [0.2, 0.2, 0.6, 1.0]
    sh = shuffle_increments(trace, random.Random(3))
    assert abs(sh[-1] - trace[-1]) < 1e-12, "shuffled-Phi keeps Phi_T identical"
    incs_orig = sorted([trace[0]] + [trace[i] - trace[i - 1] for i in range(1, len(trace))])
    incs_sh = sorted([sh[0]] + [sh[i] - sh[i - 1] for i in range(1, len(sh))])
    assert all(abs(a - b) < 1e-12 for a, b in zip(incs_orig, incs_sh)), "same increment multiset"


def test_crn_candidate_determinism():
    spec = build_dualtrack("g3", m_r=2, m_d=2, n_distractors=4, tool_pool=POOL)
    e1 = DualTrackEnv(spec); e2 = DualTrackEnv(spec)
    assert e1.observation()["spine"]["candidates"] == e2.observation()["spine"]["candidates"], \
        "two envs over the same spec must see identical candidate order (CRN)"


def test_starvation_identity():
    """An all-fail group (no spine completion) => sparse all-zero rewards => zero gradient."""
    # three trajectories, none completes the spine (phi_true never reaches 1)
    traces_true = [[0.33, 0.33, 0.67], [0.33, 0.33, 0.33], [0.0, 0.0, 0.0]]
    sparse_g = [per_turn_rewards(tr, 1.0, "sparse") for tr in traces_true]
    assert frac_zero_gradient(sparse_g) == 1.0, "starved sparse => 100% zero-gradient"
    # dense on the SAME progress still separates the progress-makers
    scored_g = [per_turn_rewards(tr, 1.0, "dense") for tr in traces_true]
    assert frac_zero_gradient(scored_g) < 1.0, "dense keeps some gradient from partial progress"
    adv = group_advantages(scored_g)
    assert adv[0][0] > 0 > adv[2][0], "dense ranks partial progress"


def test_phi_noise_shape():
    trace = [0.25, 0.5, 0.75, 1.0]
    noisy = add_phi_noise(trace, 0.1, random.Random(1))
    assert len(noisy) == len(trace)


def test_E0_estimator_theorem():
    """E0: no baseline can manufacture a learning signal from an all-EQUAL-reward group.

    GRPO's group-relative advantage subtracts the group mean and divides by the spread; a group
    whose trajectories all share the same terminal reward has ZERO spread -> zero advantage
    everywhere, for ANY baseline (a constant subtracted from constant rewards is still constant,
    giving no relative preference among actions). Consequence: a value critic (PPO) can only move
    the dense-sparse gap where the terminal group has SPREAD (mixed success) -- NEVER in deep
    starvation (all-zero groups) nor at saturation (all-win groups). This pins down exactly the
    cell where PPO-critic evidence can vs cannot matter (Phase 4 gate)."""
    # deeply-starved sparse: no trajectory completes -> all terminal rewards 0 (flat group)
    starved = [per_turn_rewards(tr, 1.0, "sparse") for tr in [[0.33, 0.33], [0.33, 0.0], [0.0, 0.0]]]
    assert all(abs(x) < 1e-12 for row in group_advantages(starved) for x in row), \
        "flat all-zero (starved) group => zero advantage for any baseline"
    # saturated: everyone succeeds (all terminal reward 1) -> also flat -> zero advantage
    allwin = [per_turn_rewards(tr, 1.0, "sparse") for tr in [[1.0], [1.0], [1.0]]]
    assert all(abs(x) < 1e-12 for row in group_advantages(allwin) for x in row), \
        "flat all-win (saturated) group => zero advantage too"
    # ONLY a mixed-success group carries signal (where a critic/baseline can matter)
    mixed = [per_turn_rewards(tr, 1.0, "sparse") for tr in [[1.0], [0.0], [0.0]]]
    assert any(abs(x) > 1e-9 for row in group_advantages(mixed) for x in row), \
        "mixed-success group => nonzero signal (the only regime a baseline can help)"


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = []
    for t in tests:
        try:
            t()
            print(f"  [ok] {t.__name__}")
        except AssertionError as e:
            fails.append((t.__name__, str(e)))
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\ndualtrack goldens: {len(tests) - len(fails)}/{len(tests)} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
