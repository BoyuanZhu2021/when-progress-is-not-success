"""Multi-turn GRPO core: per-turn potential reward + return-to-go group-relative advantage.

Implements method.md §2 (H1 full-trajectory multi-turn, DRAFT pending PI approval). Pure functions
over Phi traces — NO LLM, NO GPU — so the credit-assignment math is unit-testable on CPU before it
is wired into the training loss. The rollout (policy generates all turns, local vLLM victim responds)
and the PG loss are built separately once the H20 is up + the math is approved.

Arms (matching src/reward.py: dense = potential Phi, sparse = terminal):
  r_dense_t  = Phi_t - Phi_{t-1}                         (per-turn potential gain, >= 0)
  r_sparse_t = 1[Phi_t >= tau and Phi_{t-1} < tau]       (fires once, at the tau crossing)
Telescoping: sum_t r_dense = Phi_T ;  sum_t r_sparse = 1[Phi_T >= tau]  (same terminal return).

Advantage (per goal, over G on-policy trajectories):
  G_{i,t} = sum_{k>=t} r_{i,k}                           (return-to-go)
  b_t, sigma_t = mean/std over trajectories present at turn t
  A_{i,t} = (G_{i,t} - b_t) / (sigma_t + eps)
"""
from __future__ import annotations

import math


def per_turn_rewards(phi_trace: list[float], tau: float, arm: str) -> list[float]:
    """Per-turn reward from a Phi trace [Phi_1, .., Phi_T] (Phi_0 := 0 implied).

    dense          = potential gain ΔΦ_t (>=0 since Phi is monotone); Σ = Phi_T (optimizes a PROXY).
    sparse         = 1 exactly at the turn that first reaches tau, else 0; Σ = 1[success].
    dense_additive = PROPER PBRS (Ng 1999): sparse + terminal-zeroed shaping ΔΦ_t so Σ = 1[success]
                     EXACTLY (the last turn absorbs -Phi_T -> the shaping telescopes to 0). Shares sparse's
                     optimum, so any dense_additive-vs-sparse learning-curve gap is PURE dynamics (not a
                     changed objective) -- the clean policy-invariant control, and the Goodhart detector vs
                     dense(replace): replace wins but additive ties => the win was Phi_T proxy-optimization.
    """
    if arm not in ("dense", "sparse", "dense_additive"):
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "dense_additive":
        sparse = per_turn_rewards(phi_trace, tau, "sparse")
        dense = per_turn_rewards(phi_trace, tau, "dense")          # ΔΦ, sums to Phi_T
        rewards = [s + d for s, d in zip(sparse, dense)]
        if rewards:
            rewards[-1] -= phi_trace[-1]                           # terminal potential := 0 -> Σ shaping = 0
        return rewards
    rewards = []
    prev = 0.0
    crossed = False
    for phi in phi_trace:
        if arm == "dense":
            rewards.append(max(0.0, phi - prev))
        else:
            hit = (not crossed) and (phi >= tau) and (prev < tau)
            rewards.append(1.0 if hit else 0.0)
            crossed = crossed or phi >= tau
        prev = phi
    return rewards


def dense_additive_dual(phi_anchor: list[float], phi_shape: list[float], tau: float = 1.0) -> list[float]:
    """PBRS additive reward with SEPARATE anchor and shaping potentials (DISC-2026W33-001). Anchors on
    ``phi_anchor`` (the TRUE-success potential) and shapes with ``phi_shape`` (ANY potential -- faithful,
    gameable, or misaligned): sparse(phi_anchor) + terminal-zeroed shaping ΔΦ_shape. Σ = 1[phi_anchor reaches
    tau] EXACTLY for ANY phi_shape (the shaping telescopes to 0), so it PROVABLY shares the true-success optimum
    regardless of the shaping potential's fidelity (Ng 1999).

    This is the fix for the dualtrack_env trap ("no mt_grpo change needed"): a SINGLE-trace additive fed
    phi_scored(rho<1) would fire its sparse anchor at phi_scored>=tau (a DECOY-driven crossing, not true
    success) and thereby optimize a DIFFERENT objective. Here the anchor is always the true potential, so the
    Goodhart-immunity of additive holds at every rho / Phi-pole -- the clean control the experiment needs."""
    if len(phi_anchor) != len(phi_shape):
        raise ValueError("anchor and shape traces must have equal length")
    sparse = per_turn_rewards(phi_anchor, tau, "sparse")      # fires on TRUE (anchor) success only
    shape = per_turn_rewards(phi_shape, tau, "dense")         # ΔΦ_shape (>=0, monotone), sums to phi_shape_T
    rewards = [s + d for s, d in zip(sparse, shape)]
    if rewards:
        rewards[-1] -= phi_shape[-1]                          # terminal-zero the shaping -> Σ shaping = 0
    return rewards


def returns_to_go(rewards: list[float]) -> list[float]:
    """G_t = sum_{k>=t} r_k (undiscounted; gamma=1 matches potential telescoping)."""
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc += rewards[t]
        out[t] = acc
    return out


def group_advantages(traj_rewards: list[list[float]], eps: float = 1e-6, *,
                     baseline: str = "position", states: list[list[float]] | None = None) -> list[list[float]]:
    """GRPO group-relative, per-turn-position advantages over G trajectories for ONE goal.

    traj_rewards[i] = per-turn rewards of trajectory i (variable length; episodes end early on
    success). Returns A[i][t] = (G_{i,t} - b_t)/(sigma_t+eps), with b_t/sigma_t computed over the
    trajectories that REACHED turn t. If a position has < 2 trajectories or zero spread, advantages
    there are 0 (no usable gradient) — this is exactly how sparse degenerates when success is rare.

    ``baseline``:
      ``"position"``          — b_t/sigma_t pooled over ALL trajectories at turn position t (the
                                historical default, kept bit-for-bit so prior results reproduce).
      ``"rloo"``              — leave-one-out: A_i = G_i - mean of the OTHER k-1 returns at this
                                position, with NO std normalisation (Ahmadian et al. 2024).
      ``"state_stratified"``  — b/sigma computed within (position, state) cells, where the state is
                                ``states[i][t]``. Cells holding < 2 trajectories yield NO signal
                                (advantage 0) rather than being pooled back together: pooling mixed
                                states would re-introduce the very shift this baseline removes. Cells
                                are arm-independent (they come from ``states``), so the same decisions
                                are zeroed for every arm and the comparison stays fair.

    **Why state stratification exists (DISC-2026W33-003 / F6).** For the PBRS arm,
    ``G_t^additive = 1[S at >= t] - Phi_{t-1}`` EXACTLY, i.e. it is sparse's return-to-go shifted by a
    STATE term. A position-only baseline pools trajectories whose Phi_{t-1} differ, so that shift does
    NOT cancel: it survives into the advantage and inflates sigma_t, attenuating the true signal. The
    consequence is that ``dense_additive`` under ``baseline="position"`` is **not** the policy-invariance
    control it is documented to be. Stratifying on Phi_{t-1} makes the shift constant within a cell, so
    it cancels in (G - b) and A^additive == A^sparse cell-wise — which is what a PBRS control must satisfy.
    Pass ``states[i][t] = Phi_{t-1}`` (the potential BEFORE turn t) to get that property.

    Stratification assumes the state is COARSE enough to populate cells (for the H3 defense potential
    ``Phi = (#conjuncts)/N`` it takes only N+1 values, so at G=6-8 most cells hold >= 2). If a
    fine-grained/continuous potential is ever used, the generalization is an affine state baseline
    ``b_t(s) = alpha_t + beta_t * s`` fitted per position, which absorbs the -Phi_{t-1} term without
    needing populated cells; not implemented here because the defense potential does not need it.
    """
    if baseline not in ("position", "state_stratified", "rloo"):
        raise ValueError(f"unknown baseline {baseline!r}")
    if baseline == "state_stratified" and states is None:
        raise ValueError("baseline='state_stratified' requires `states` (e.g. states[i][t] = Phi_{t-1})")
    return _advantages_and_mask(traj_rewards, eps, baseline=baseline, states=states)[0]


def _advantages_and_mask(traj_rewards, eps=1e-6, *, baseline="position", states=None):
    """``group_advantages`` plus a boolean mask of WHERE a usable relative signal existed.

    ``mask[i][t]`` is True iff position (i,t) sat in a cell with >=2 trajectories AND non-degenerate
    spread -- i.e. exactly the positions that received a non-trivial advantage. Everywhere else the
    advantage is structurally 0 and the arm contributes NO gradient there. ``gated_advantages`` needs
    that distinction and must not re-derive the cell logic (it would drift); this is the single
    source of truth, and ``group_advantages`` is a thin wrapper so its behaviour is bit-identical."""
    rtg = [returns_to_go(r) for r in traj_rewards]
    maxT = max((len(r) for r in rtg), default=0)
    adv = [[0.0] * len(r) for r in rtg]
    mask = [[False] * len(r) for r in rtg]
    for t in range(maxT):
        col = [(i, rtg[i][t]) for i in range(len(rtg)) if t < len(rtg[i])]
        if len(col) < 2:
            continue
        if baseline in ("position", "rloo"):
            cells = [col]                      # rloo pools the whole column; the leave-one-out is per-member
        else:
            # Group by the state at this position. Singleton cells are LEFT AT ZERO, not pooled:
            # pooling different states back together would restore the shift we are removing.
            by_state: dict[float | None, list] = {}
            for i, v in col:
                key = round(states[i][t], 9) if t < len(states[i]) else None
                by_state.setdefault(key, []).append((i, v))
            cells = [cell for cell in by_state.values() if len(cell) >= 2]
        for cell in cells:
            if len(cell) < 2:
                continue
            vals = [v for _, v in cell]
            if baseline == "rloo":
                # REINFORCE Leave-One-Out (Ahmadian et al. 2024): score each trajectory against the mean
                # of the OTHERS, never against a mean that includes itself. Unbiased, and NOT std-normalised.
                #
                # Worth being precise about what this actually changes here, because it is less than the
                # name suggests: algebraically G_i - mean_{j!=i}(G_j) == (k/(k-1)) * (G_i - mean_all), so the
                # leave-one-out part is the position-mean baseline up to a CONSTANT factor that a fixed
                # learning rate largely absorbs. The substantive difference from `position` is dropping the
                # /sigma normalisation -- which is exactly the term implicated when informationless placebos
                # beat sparse in the H3 series (DISC-2026W33-003).
                k = len(cell)
                tot = sum(vals)
                for i, v in cell:
                    adv[i][t] = v - (tot - v) / (k - 1)
                    mask[i][t] = True
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            sigma = math.sqrt(var)
            if sigma < eps:        # no spread -> no relative signal (degenerate group)
                continue
            for i, v in cell:
                adv[i][t] = (v - mean) / (sigma + eps)
                mask[i][t] = True
    return adv, mask


def gated_advantages(sparse_rewards, dense_rewards, *, beta=1.0, eps=1e-6,
                     baseline="position", states=None):
    """STARVATION-GATED shaping (DISC-2026W33-006) -- the DYNAMIC reward assignment.

    Every arm the project has run so far applies its reward rule with a FIXED coefficient, identically,
    everywhere: ``r_t = DeltaPhi_t`` is a pure function of the Phi trace, blind to whether the terminal
    signal already had gradient at that position. That is the mechanism behind the central negative
    ([EXP-2026W32-001](../../LOGS/2026-W32.md#exp-2026w32-001)): dense kept pulling the policy toward
    partial progress even in cells where sparse was perfectly informative, and the damage GREW with
    training (OOD delta -0.104 @step8 -> -0.194 @step12).

    This gates on the group's own state instead::

        A_t = A_t^sparse                       where the sparse group has spread  (sigma_t >= eps)
            = beta * A_t^dense                 where it does not (sparse advantage is structurally 0)

    So shaping can only ever write into cells that were contributing NOTHING, and can never dilute a
    cell where the terminal objective already had signal. It turns the project's central empirical
    finding -- 'dense helps iff sparse is starved' -- from a regime the researcher hand-picks into a
    decision the algorithm makes per cell, online.

    ``beta`` is exposed per call so the caller can ANNEAL it across training steps (beta_k), which PBRS
    policy-invariance and the growing-damage observation above both argue for; beta=0 recovers pure sparse.

    **Honest caveat (do not lose this).** A data-dependent coefficient is NOT potential-based, so
    Ng-1999 policy invariance no longer holds and this arm is not guaranteed to preserve the optimum.
    That is deliberate -- invariance is exactly what pins dense_additive's asymptotic advantage at zero
    -- but it means Goodhart is live again. Keep ``dense_additive`` as the invariant control and
    ``permuted`` as the placebo whenever this arm is run.
    """
    a_sparse, signal = _advantages_and_mask(sparse_rewards, eps, baseline=baseline, states=states)
    a_dense, _ = _advantages_and_mask(dense_rewards, eps, baseline=baseline, states=states)
    out = [row[:] for row in a_sparse]
    for i in range(len(out)):
        for t in range(len(out[i])):
            if not signal[i][t]:                      # sparse gave no usable gradient here
                out[i][t] = beta * (a_dense[i][t] if t < len(a_dense[i]) else 0.0)
    return out


def phi_prev_states(phi_traces: list[list[float]]) -> list[list[float]]:
    """States for ``baseline="state_stratified"``: Phi_{t-1} (the potential BEFORE turn t), with
    Phi_{-1} := 0. This is exactly the term that ``G_t^additive`` is shifted by, so stratifying on it
    is what makes the PBRS arm's advantage match sparse's cell-wise."""
    return [[0.0, *tr[:-1]] for tr in phi_traces]


def advantages(traj_rewards: list[list[float]], estimator: str = "grpo", eps: float = 1e-6, *,
               baseline: str = "position", states: list[list[float]] | None = None) -> list[list[float]]:
    """Per-turn advantages under one of three ESTIMATORS (DISC-2026W32-003 credit-assignment axis), ordered by
    how much temporal-credit / variance-reduction they do BEFORE any reward shaping:
      reinforce          = raw return-to-go G_{i,t} (NO baseline, NO normalization) -- weakest; high variance.
      reinforce_baseline = G_{i,t} - b_t (per-turn-position mean baseline; variance reduction, NO std-norm).
      grpo               = (G_{i,t} - b_t)/(sigma_t+eps) (group-relative, std-normalized) -- project default.
    Hypothesis: dense's advantage over sparse SHRINKS as the estimator's own credit capacity rises
    (REINFORCE > +baseline > GRPO), because PBRS (dense) is a hand-supplied critic (Wiewiora 2003) -- decisive
    for a weak estimator, redundant for a strong one."""
    if estimator == "grpo":
        return group_advantages(traj_rewards, eps, baseline=baseline, states=states)
    rtg = [returns_to_go(r) for r in traj_rewards]
    if estimator == "reinforce":
        return [list(r) for r in rtg]                             # raw return-to-go, no baseline
    if estimator == "reinforce_baseline":
        maxT = max((len(r) for r in rtg), default=0)
        adv = [[0.0] * len(r) for r in rtg]
        for t in range(maxT):
            col = [(i, rtg[i][t]) for i in range(len(rtg)) if t < len(rtg[i])]
            if len(col) < 2:
                continue
            mean = sum(v for _, v in col) / len(col)
            for i, v in col:
                adv[i][t] = v - mean                              # subtract per-position mean, no std-norm
        return adv
    raise ValueError(f"unknown estimator {estimator!r}")


def train_m_threshold(m_of_K: int, K: int, *, step: int = 1, steps: int = 1,
                      hindsight_m: int | None = None, curriculum: bool = False) -> tuple[int, float]:
    """TRAINING-time m-of-K success threshold (fullft-campaign-v1 B2/B3). Returns ``(m_train, tau_train)``
    where ``tau_train = m_train / K`` is the rate the ``count_trace`` anchor is thresholded at.

    Hindsight relabeling and the m-curriculum are the SAME mechanism -- a training threshold that differs
    from the evaluation threshold -- so they share one implementation rather than two parallel paths that
    could drift (AGENTS.md §4.3). The two knobs are mutually exclusive:

      default        m_train = m_of_K                        (byte-identical to every prior campaign)
      hindsight_m=M  m_train = M, constant                   (B2: partial successes count as successes)
      curriculum     m_train ramps 1 -> m_of_K across steps   (B3: start easy, end on the real objective)

    Curriculum schedule, with ``s = step - 1`` (a3's loop is 1-indexed) and ``S = steps``::

        m_train = max(1, round(1 + (m_of_K - 1) * s / (S - 1)))

    **The DV is never touched.** Evaluation always thresholds at ``m_of_K``; only the reward the policy is
    trained against moves. A caller that routes this value into ``evaluate_asr`` has broken the experiment,
    which is why the a3 wiring keeps the eval threshold on a separate variable and the goldens assert it.
    """
    if hindsight_m is not None and curriculum:
        raise ValueError("--hindsight-m and --m-curriculum are mutually exclusive (both retarget the "
                         "training threshold; combining them makes the schedule unidentifiable)")
    if not m_of_K or not K:
        raise ValueError("train_m_threshold requires both m_of_K and K (the m-of-K readout must be active)")
    if hindsight_m is not None:
        if not 1 <= hindsight_m <= m_of_K:
            raise ValueError(f"hindsight_m must be in [1, m_of_K={m_of_K}], got {hindsight_m}")
        m_train = hindsight_m
    elif curriculum:
        s = max(0, step - 1)
        denom = max(1, steps - 1)
        m_train = max(1, min(m_of_K, round(1 + (m_of_K - 1) * s / denom)))
    else:
        m_train = m_of_K
    return m_train, m_train / K


def sft_advantages(successes: list[bool], n_turns: list[int]) -> list[list[float]]:
    """Best-of-N SFT weights (fullft-campaign-v1 B4): 1.0 on every turn of a SUCCESSFUL trajectory, 0 else.

    This is not an advantage in the RL sense and is deliberately not computed from one. Feeding weight 1.0
    into the existing ``grpo_loss_step`` makes its per-example term ``pg = -adv * plogp.sum()`` collapse to
    ``-log p(attacker tokens | prompt)`` -- exactly teacher-forced cross-entropy on the attacker's own
    tokens, masked to them because ``token_logps_batch`` only scores ``resp_ids``. So SFT reuses the one
    loss implementation the project already has instead of adding a second (AGENTS.md §4.3), and the
    caller only has to set ``beta_kl=0`` to drop the KL term.

    A goal whose G attempts all failed contributes an all-zero row, which the a3 example filter drops, so
    it is a genuine no-op rather than a zero-magnitude gradient of arbitrary direction."""
    if len(successes) != len(n_turns):
        raise ValueError(f"successes/n_turns length mismatch: {len(successes)} != {len(n_turns)}")
    return [[1.0 if ok else 0.0] * n for ok, n in zip(successes, n_turns)]


class AdaptiveRolloutAllocator:
    """Variance-informed rollout allocation over goals (Block B6), after VIP, ICLR 2026 (arXiv 2602.01601).

    GRPO gives every goal the same G rollouts, which "implicitly treats all prompts as equally
    informative". For a binary reward the group's gradient signal scales with ``p(1-p)`` and is
    **exactly zero at p=0 and p=1** -- so a uniform budget spends real compute on goals that cannot
    contribute. This campaign measured ``all_zero_group_frac ~ 0.47``: about half of every batch.

    This reallocates the SAME total number of episodes toward goals whose success rate sits near the
    informative band, using each goal's own recent history as the success-probability estimate (an
    EMA rather than VIP's Gaussian process -- with ~24 goals a GP is machinery without benefit).

    Two properties matter for this campaign specifically:

    * **The objective never moves.** Only *where rollouts are spent* changes. dense/hindsight/
      curriculum each lost by redefining success; this does not touch the reward, the threshold, or
      the DV.
    * **No goal is abandoned.** ``floor`` guarantees every goal keeps a minimum allocation, so a
      never-yet-solved goal still gets explored and the held-out compositions stay covered. Dropping
      zero-accuracy goals outright would optimise train-time gradient at the cost of the OOD metric
      that is actually being measured -- the exact trade this campaign exists to avoid.
    """

    def __init__(self, goal_ids, total: int, floor: int = 2, ema: float = 0.5, prior: float = 0.3):
        if floor < 1:
            raise ValueError("floor must be >= 1 (a goal must never be dropped entirely)")
        if total < floor * len(goal_ids):
            raise ValueError(f"total {total} cannot satisfy floor {floor} for {len(goal_ids)} goals")
        self.goal_ids = list(goal_ids)
        self.total = int(total)
        self.floor = int(floor)
        self.ema = float(ema)
        # start at the campaign's observed base rate, so step 1 is near-uniform rather than arbitrary
        self.p = {g: float(prior) for g in self.goal_ids}

    def observe(self, goal_id, successes: int, attempts: int) -> None:
        if attempts <= 0 or goal_id not in self.p:
            return
        rate = successes / attempts
        self.p[goal_id] = (1 - self.ema) * self.p[goal_id] + self.ema * rate

    def allocate(self) -> dict:
        """Episodes per goal. Sums to EXACTLY ``total``; every goal gets at least ``floor``."""
        n = len(self.goal_ids)
        alloc = {g: self.floor for g in self.goal_ids}
        spare = self.total - self.floor * n
        if spare <= 0:
            return alloc
        # weight by sqrt(p(1-p)): the standard deviation of the group's success indicator, i.e. how
        # much spread an extra rollout can buy. sqrt rather than p(1-p) keeps the tail less extreme,
        # so a goal at p=0.1 is de-emphasised but not silenced.
        w = {g: math.sqrt(max(0.0, self.p[g] * (1.0 - self.p[g]))) for g in self.goal_ids}
        tot = sum(w.values())
        if tot <= 1e-12:                      # every goal saturated at p=0 or p=1 -> stay uniform
            for i, g in enumerate(self.goal_ids):
                alloc[g] += spare // n + (1 if i < spare % n else 0)
            return alloc
        # largest-remainder apportionment so the total is exact, never off by rounding
        exact = {g: spare * w[g] / tot for g in self.goal_ids}
        for g in self.goal_ids:
            alloc[g] += int(exact[g])
        left = self.total - sum(alloc.values())
        for g in sorted(self.goal_ids, key=lambda x: exact[x] - int(exact[x]), reverse=True)[:left]:
            alloc[g] += 1
        return alloc


class SelfImitationBuffer:
    """Per-goal store of TRUE successes, replayed into groups where sparse is starved (Block B5).

    This arm is the union of the only two things this campaign found that did not hurt, aimed at the
    one bottleneck it measured:

    * **Keep the objective.** dense (-0.140), hindsight (-0.216) and curriculum (-0.162) each lost in
      proportion to how far their surrogate sat from true m-of-K success. Only genuine successes ever
      enter this buffer, so the thing being optimised is unchanged.
    * **Real successes teach.** `sft` (CE on true successes) and `G12` (more samples of the true
      signal) were the two arms that leaned positive.
    * **Act only where there is nothing.** ~47% of sparse's groups have every attempt fail
      identically and contribute exactly zero gradient. W33's `dense_gated` had this gate right but
      filled the hole with DeltaPhi, whose forecasting AUC is 0.585 -- essentially noise. Replay fills
      the same hole with trajectories that actually reached the goal.

    Unlike hindsight, nothing is relabeled. Unlike G12, no extra rollout is paid for -- the successes
    were already generated and thrown away. A goal never sees another goal's trajectory.

    ``capacity`` bounds memory per goal; the newest successes are kept because the policy moves and
    stale trajectories drift off-policy (this is replay of one's own history, so some staleness is
    inherent and is the honest cost of the method).
    """

    def __init__(self, capacity: int = 4):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._by_goal: dict[object, list] = {}

    def add(self, goal_id, success: bool, turns: list) -> bool:
        """Store a trajectory's turns iff it TRULY succeeded. Returns whether it was stored."""
        if not success or not turns:
            return False
        buf = self._by_goal.setdefault(goal_id, [])
        buf.append(list(turns))
        del buf[:-self.capacity]              # keep the freshest `capacity` entries
        return True

    def sample(self, goal_id, k: int) -> list:
        """Up to k stored successes for THIS goal, newest first. Empty if none seen yet."""
        if k <= 0:
            return []
        return list(reversed(self._by_goal.get(goal_id, [])))[:k]

    def size(self, goal_id=None) -> int:
        if goal_id is not None:
            return len(self._by_goal.get(goal_id, []))
        return sum(len(v) for v in self._by_goal.values())


def group_is_starved(advantages: list[list[float]], eps: float = 1e-9) -> bool:
    """True when a group produced no usable gradient anywhere -- the cell replay targets.

    Same predicate a3 already uses for ``all_zero_group_frac``, factored out so the replay arm and
    the diagnostic can never disagree about what "starved" means."""
    return not any(abs(v) >= eps for row in advantages for v in row)


def clip_negative_advantages(advantages: list[list[float]]) -> list[list[float]]:
    """`rl_pos` (sft-mechanism-4b-9b-v1 §3): keep GRPO's positive advantages and the KL term, drop every
    negative push. After clipping a group's advantages no longer sum to zero -- a deliberately BIASED
    estimator that probes what the negatives do, not a proposed method. Call it BEFORE
    ``group_is_starved`` so a group whose only signal was negative counts as starved."""
    return [[v if v > 0.0 else 0.0 for v in row] for row in advantages]


def frac_zero_gradient(traj_rewards: list[list[float]]) -> float:
    """Diagnostic: fraction of (trajectory,turn) decisions with zero advantage (no learning signal).
    High for sparse when success is rare — the quantity Claim 1 predicts separates the arms."""
    adv = group_advantages(traj_rewards)
    total = sum(len(a) for a in adv) or 1
    zero = sum(1 for a in adv for x in a if abs(x) < 1e-9)
    return zero / total


# --------------------------------------------------------------------------- golden self-test
def _approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def _selftest_gated(fails):
    """Contract goldens for gated_advantages (DISC-2026W33-006 dynamic reward assignment)."""
    # (a) sparse INFORMATIVE everywhere (mixed success) -> gating must be a NO-OP: gated == sparse exactly.
    phis = [[0.5, 1.0], [0.5, 1.0], [0.5, 0.5], [0.0, 0.5]]        # 2 succeed, 2 do not -> spread at every t
    sp = [per_turn_rewards(p, 1.0, "sparse") for p in phis]
    dn = [per_turn_rewards(p, 1.0, "dense") for p in phis]
    g = gated_advantages(sp, dn)
    a_sp = group_advantages(sp)
    if g != a_sp:
        fails.append("gated != sparse when the sparse group has spread everywhere (must be a no-op)")

    # (b) sparse FULLY STARVED (nobody succeeds) -> sparse advantage is all-zero; gated must supply
    #     dense's gradient instead. This is the whole point of the arm.
    phis = [[0.25, 0.25], [0.25, 0.5], [0.5, 0.75], [0.0, 0.25]]   # no trajectory reaches tau=1
    sp = [per_turn_rewards(p, 1.0, "sparse") for p in phis]
    dn = [per_turn_rewards(p, 1.0, "dense") for p in phis]
    a_sp = group_advantages(sp)
    if any(abs(v) > 1e-12 for row in a_sp for v in row):
        fails.append("precondition broken: starved sparse should have all-zero advantage")
    g = gated_advantages(sp, dn)
    if not any(abs(v) > 1e-12 for row in g for v in row):
        fails.append("gated supplied NO gradient in a fully starved group (should fall back to dense)")
    if g != group_advantages(dn):
        fails.append("gated != dense when sparse is starved everywhere")

    # (c) beta scales only the fallback, and beta=0 recovers pure sparse.
    if gated_advantages(sp, dn, beta=0.0) != a_sp:
        fails.append("beta=0 must recover pure sparse")
    half = gated_advantages(sp, dn, beta=0.5)
    if any(abs(h - 0.5 * f) > 1e-12 for hr, fr in zip(half, g) for h, f in zip(hr, fr)):
        fails.append("beta must scale the dense fallback linearly")

    # (d) MIXED: informative at t=0, starved at t=1. Gating must be per-CELL, not per-episode --
    #     sparse's value is preserved where it had signal and dense fills in only where it did not.
    phis = [[1.0, 1.0], [1.0, 1.0], [0.5, 0.75], [0.25, 0.5]]
    sp = [per_turn_rewards(p, 1.0, "sparse") for p in phis]
    dn = [per_turn_rewards(p, 1.0, "dense") for p in phis]
    a_sp = group_advantages(sp)
    g = gated_advantages(sp, dn)
    t0_signal = any(abs(a_sp[i][0]) > 1e-12 for i in range(len(a_sp)))
    if not t0_signal:
        fails.append("precondition broken: t=0 should be informative for sparse here")
    if any(abs(g[i][0] - a_sp[i][0]) > 1e-12 for i in range(len(g))):
        fails.append("gated overwrote a cell where sparse HAD signal (must never dilute)")
    print("  [ok] gated: no-op where sparse has signal; falls back to dense where starved; beta scales fallback")


def _selftest_credit(fails):
    """Goldens for the fullft-campaign B2/B3/B4 credit-assignment methods."""
    K, M = 4, 4

    # (a) DEFAULT is byte-identical to every prior campaign: m_train == m_of_K at every step.
    for st in range(1, 17):
        if train_m_threshold(M, K, step=st, steps=16) != (4, 1.0):
            fails.append(f"default train threshold moved at step {st} (must stay m_of_K)")
            break

    # (b) HINDSIGHT (B2): the plan's worked example -- trajectories scoring {0,1,2,3,4} of K=4 fields are
    #     relabeled {0,0,1,1,1} at m=2. count_trace is a RATE, so compare count/K against tau.
    m_h, tau_h = train_m_threshold(M, K, hindsight_m=2)
    if (m_h, tau_h) != (2, 0.5):
        fails.append(f"hindsight_m=2 gave {(m_h, tau_h)}, expected (2, 0.5)")
    relabeled = [1 if (c / K) >= tau_h - 1e-9 else 0 for c in (0, 1, 2, 3, 4)]
    if relabeled != [0, 0, 1, 1, 1]:
        fails.append(f"hindsight m=2 relabeling {relabeled} != [0,0,1,1,1]")
    # and it must be STRICTLY easier than the eval threshold, else the arm is a no-op
    if not tau_h < M / K:
        fails.append("hindsight threshold must be strictly below the eval threshold")

    # (c) CURRICULUM (B3): ramps 1 -> m_of_K, monotone, hitting both endpoints exactly.
    sched = [train_m_threshold(M, K, step=st, steps=16, curriculum=True)[0] for st in range(1, 17)]
    if sched[0] != 1:
        fails.append(f"curriculum must start at m=1, got {sched[0]}")
    if sched[-1] != M:
        fails.append(f"curriculum must end at m={M}, got {sched[-1]}")
    if any(b < a for a, b in zip(sched, sched[1:])):
        fails.append(f"curriculum must be monotone non-decreasing, got {sched}")
    if set(sched) != {1, 2, 3, 4}:
        fails.append(f"curriculum must visit every m in 1..{M}, got {sorted(set(sched))}")
    # the plan's stated behaviour: at step 1 a count=1 trajectory trains as a SUCCESS, at the last step
    # the SAME trajectory trains as a failure. This is the whole point of the arm.
    if not ((1 / K) >= train_m_threshold(M, K, step=1, steps=16, curriculum=True)[1] - 1e-9):
        fails.append("curriculum step 1: a count=1 trajectory must count as success")
    if (1 / K) >= train_m_threshold(M, K, step=16, steps=16, curriculum=True)[1] - 1e-9:
        fails.append("curriculum last step: a count=1 trajectory must NOT count as success")

    # (d) the DV is never retargeted -- the eval threshold is m_of_K/K no matter what training does.
    for kw in ({"hindsight_m": 1}, {"curriculum": True, "step": 1, "steps": 16}):
        _m, tau_tr = train_m_threshold(M, K, **kw)
        if _approx(tau_tr, M / K):
            fails.append(f"{kw} failed to move the TRAINING threshold off the eval threshold")
    # guards
    for bad in ({"hindsight_m": 0}, {"hindsight_m": M + 1},
                {"hindsight_m": 2, "curriculum": True}):
        try:
            train_m_threshold(M, K, **bad)
            fails.append(f"train_m_threshold{bad} must raise")
        except ValueError:
            pass

    # (g) B6 adaptive rollout allocation: budget is CONSERVED, no goal is dropped, and the mass
    #     moves toward the informative band. Each assert below is a way the arm could silently
    #     become something other than what it claims.
    goals = [f"g{i}" for i in range(6)]
    alloc = AdaptiveRolloutAllocator(goals, total=36, floor=2, ema=1.0)
    a0 = alloc.allocate()
    if sum(a0.values()) != 36:
        fails.append(f"B6 initial allocation does not conserve budget: {sum(a0.values())} != 36")
    if len(set(a0.values())) != 1:
        fails.append("B6 must start near-uniform before any goal has history")
    # p=0 and p=1 goals are saturated (zero signal); mid-p goals should absorb the spare budget
    for g, s, n in (("g0", 0, 10), ("g1", 0, 10), ("g2", 10, 10), ("g3", 5, 10), ("g4", 4, 10), ("g5", 6, 10)):
        alloc.observe(g, s, n)
    a1 = alloc.allocate()
    if sum(a1.values()) != 36:
        fails.append(f"B6 budget not conserved after observe: {sum(a1.values())} != 36")
    if min(a1.values()) < 2:
        fails.append(f"B6 floor violated: {a1} -- a never-solved goal must still be explored")
    if not (a1["g3"] > a1["g0"] and a1["g3"] > a1["g2"]):
        fails.append(f"B6 must favour p~0.5 over saturated p=0 / p=1 goals: {a1}")
    if a1["g0"] != 2 or a1["g2"] != 2:
        fails.append(f"B6 saturated goals should sit at the floor: {a1}")
    # degenerate case: every goal saturated -> fall back to uniform, still exact
    alloc2 = AdaptiveRolloutAllocator([f"h{i}" for i in range(4)], total=20, floor=1, ema=1.0)
    for i in range(4):
        alloc2.observe(f"h{i}", 0, 5)
    a2 = alloc2.allocate()
    if sum(a2.values()) != 20 or max(a2.values()) - min(a2.values()) > 1:
        fails.append(f"B6 all-saturated must degrade to uniform and stay exact: {a2}")
    try:
        AdaptiveRolloutAllocator(goals, total=5, floor=2)
        fails.append("B6 must reject a budget that cannot satisfy the floor")
    except ValueError:
        pass
    try:
        AdaptiveRolloutAllocator(goals, total=36, floor=0)
        fails.append("B6 must reject floor=0 (that would abandon goals and bias the OOD metric)")
    except ValueError:
        pass

    # (f) SIL (B5): replay buffer. The invariants that keep this from becoming another hindsight.
    b = SelfImitationBuffer(capacity=2)
    if b.add("g1", False, [{"t": 1}]):
        fails.append("SIL buffer stored a FAILURE -- the objective must stay true m-of-K")
    if b.add("g1", True, []):
        fails.append("SIL buffer stored an empty trajectory")
    for i in range(3):                                   # capacity=2 -> oldest evicted
        b.add("g1", True, [{"t": i}])
    if b.size("g1") != 2:
        fails.append(f"SIL capacity not enforced: {b.size('g1')} != 2")
    if [t[0]["t"] for t in b.sample("g1", 5)] != [2, 1]:
        fails.append("SIL sample must return NEWEST first (stale trajectories drift off-policy)")
    b.add("g2", True, [{"t": 99}])
    if any(t[0]["t"] == 99 for t in b.sample("g1", 5)):
        fails.append("SIL leaked one goal's success into another goal")
    if b.sample("never-seen", 3) != []:
        fails.append("SIL must return nothing for a goal with no stored success")
    if b.sample("g1", 0) != []:
        fails.append("SIL k=0 must return nothing")
    # the gate: replay fires ONLY where the group produced no gradient at all
    if not group_is_starved([[0.0, 0.0], [0.0]]):
        fails.append("group_is_starved missed an all-zero group")
    if group_is_starved([[0.0, 0.0], [0.0, 1e-3]]):
        fails.append("group_is_starved fired on a group that HAS signal (would dilute sparse)")

    # (e) SFT (B4): weight 1 on successful trajectories' turns, 0 elsewhere; an all-failed goal is a
    #     genuine no-op (all-zero row -> a3 drops the examples) rather than a random-direction gradient.
    w = sft_advantages([True, False, True], [2, 3, 1])
    if w != [[1.0, 1.0], [0.0, 0.0, 0.0], [1.0]]:
        fails.append(f"sft_advantages gave {w}")
    if any(v != 0.0 for row in sft_advantages([False, False], [2, 2]) for v in row):
        fails.append("all-failed SFT group must be an exact no-op")
    try:
        sft_advantages([True], [1, 2])
        fails.append("sft_advantages length mismatch must raise")
    except ValueError:
        pass


def _selftest() -> int:
    fails = []

    _selftest_gated(fails)
    _selftest_credit(fails)

    # 1. dense telescopes to Phi_T; sparse fires once at tau
    tr = [0.2, 0.2, 0.6, 1.0]
    d = per_turn_rewards(tr, tau=1.0, arm="dense")
    s = per_turn_rewards(tr, tau=1.0, arm="sparse")
    if not _approx(sum(d), 1.0):
        fails.append(f"dense sum {sum(d)} != Phi_T 1.0")
    if d != [0.2, 0.0, 0.4, 0.4] and not all(_approx(x, y) for x, y in zip(d, [0.2, 0.0, 0.4, 0.4])):
        fails.append(f"dense per-turn wrong: {d}")
    if s != [0, 0, 0, 1.0]:
        fails.append(f"sparse should fire once at the tau=1 crossing: {s}")
    # tau=0.6 -> sparse fires at turn 3 (Phi 0.2->0.6)
    s06 = per_turn_rewards(tr, tau=0.6, arm="sparse")
    if s06 != [0, 0, 1.0, 0]:
        fails.append(f"sparse tau=0.6 crossing wrong: {s06}")

    # 2. return-to-go
    if returns_to_go([0.2, 0.0, 0.4, 0.4]) != [1.0, 0.8, 0.8, 0.4]:
        fails.append(f"rtg wrong: {returns_to_go([0.2,0.0,0.4,0.4])}")

    # 3. THE mechanism: a group of failed-but-partial trajectories.
    #    3 trajectories, none reaches tau=1.0 (max Phi 0.6/0.4/0.0) -> sparse has NO signal anywhere,
    #    dense still separates the progress-makers from the zero one.
    traces = [[0.2, 0.4, 0.6], [0.2, 0.4, 0.4], [0.0, 0.0, 0.0]]
    dense_g = [per_turn_rewards(tr, 1.0, "dense") for tr in traces]
    sparse_g = [per_turn_rewards(tr, 1.0, "sparse") for tr in traces]
    fz_dense = frac_zero_gradient(dense_g)
    fz_sparse = frac_zero_gradient(sparse_g)
    if fz_sparse != 1.0:
        fails.append(f"sparse should have 100% zero-gradient on all-failed group, got {fz_sparse}")
    if fz_dense >= 1.0:
        fails.append(f"dense should have SOME gradient on partial-progress group, got {fz_dense}")
    adv_dense = group_advantages(dense_g)
    # trajectory 0 (most progress) should get positive advantage at turn 0; trajectory 2 (none) negative
    if not (adv_dense[0][0] > 0 > adv_dense[2][0]):
        fails.append(f"dense advantage should rank progress: t0={adv_dense[0][0]:.2f} t2={adv_dense[2][0]:.2f}")

    # 4. sparse DOES get signal when the group has mixed success (some reach tau)
    mixed = [[0.5, 1.0], [0.5, 0.5], [0.0, 0.0]]  # traj0 succeeds
    sparse_mixed = [per_turn_rewards(tr, 1.0, "sparse") for tr in mixed]
    if frac_zero_gradient(sparse_mixed) >= 1.0:
        fails.append("sparse should have signal when some trajectories succeed")

    # 5. dense_additive (PBRS): shares sparse's terminal return EXACTLY, differs per-turn (shaping)
    for tr in ([0.2, 0.2, 0.6, 1.0], [0.2, 0.4, 0.4], [0.0, 0.0, 0.0]):
        add = per_turn_rewards(tr, 1.0, "dense_additive")
        if not _approx(sum(add), sum(per_turn_rewards(tr, 1.0, "sparse"))):
            fails.append(f"dense_additive sum {sum(add)} != sparse for {tr}")
    if per_turn_rewards([0.2, 0.4, 0.4], 1.0, "dense_additive") == per_turn_rewards([0.2, 0.4, 0.4], 1.0, "sparse"):
        fails.append("dense_additive should differ per-turn from sparse (shaping spreads credit)")

    # 6. estimator ladder: reinforce=raw rtg(>=0), reinforce_baseline mean-zero per position, grpo==group_advantages
    dg = [per_turn_rewards(tr, 1.0, "dense") for tr in [[0.2, 0.4, 0.6], [0.2, 0.4, 0.4], [0.0, 0.0, 0.0]]]
    a_rf, a_rb, a_gr = advantages(dg, "reinforce"), advantages(dg, "reinforce_baseline"), advantages(dg, "grpo")
    if any(x < -1e-9 for a in a_rf for x in a):
        fails.append("reinforce advantages (raw return-to-go) should be >= 0")
    if not _approx(sum(a_rb[i][0] for i in range(3)), 0.0):
        fails.append(f"reinforce_baseline t=0 column not mean-zero: {[a_rb[i][0] for i in range(3)]}")
    if a_gr != group_advantages(dg):
        fails.append("advantages(grpo) must equal group_advantages")

    # 7. dense_additive_dual (DISC-2026W33-001): Σ = 1[ANCHOR success] for ANY shape trace (Goodhart-immune)
    #    -- success anchor + maxed misaligned shape still sums to 1 (true success), not to the shape's max.
    a_win = dense_additive_dual([0.5, 1.0], [0.3, 0.6])       # anchor succeeds, arbitrary shape
    if not _approx(sum(a_win), 1.0):
        fails.append(f"additive_dual should sum to 1[anchor success]=1, got {sum(a_win)}")
    a_fail = dense_additive_dual([0.5, 0.5], [1.0, 1.0])      # anchor FAILS but shape is MAXED
    if not _approx(sum(a_fail), 0.0):
        fails.append(f"additive_dual with failed anchor must sum to 0 regardless of maxed shape, got {sum(a_fail)}")
    # per-turn it differs from bare sparse-on-anchor (the shaping spreads credit early)
    if dense_additive_dual([0.5, 1.0], [0.5, 1.0]) == per_turn_rewards([0.5, 1.0], 1.0, "sparse"):
        fails.append("additive_dual should differ per-turn from sparse (shaping)")
    # matches the single-trace additive when anchor == shape (faithful, rho=1)
    if [round(x, 9) for x in dense_additive_dual([0.2, 0.6, 1.0], [0.2, 0.6, 1.0])] != \
       [round(x, 9) for x in per_turn_rewards([0.2, 0.6, 1.0], 1.0, "dense_additive")]:
        fails.append("additive_dual(faithful) must equal single-trace dense_additive")

    # 8. F6 (DISC-2026W33-003): dense_additive is a PBRS control ONLY under a state-stratified baseline.
    #    G_t^additive = 1[S at >= t] - Phi_{t-1}, so a position-only baseline leaves the state term in.
    phis = [[0.25, 0.50, 0.75, 1.00],    # succeeds
            [0.25, 0.50, 0.50, 0.50],    # stalls mid
            [0.00, 0.25, 0.50, 0.75],    # slower, never succeeds
            [0.25, 0.25, 0.25, 0.25]]    # stuck
    add_g = [per_turn_rewards(p, 1.0, "dense_additive") for p in phis]
    spa_g = [per_turn_rewards(p, 1.0, "sparse") for p in phis]
    states = phi_prev_states(phis)
    # (a) the identity the whole fix rests on: G^additive == G^sparse - Phi_{t-1}
    for i, p in enumerate(phis):
        g_add, g_spa = returns_to_go(add_g[i]), returns_to_go(spa_g[i])
        for t in range(len(p)):
            if not _approx(g_add[t], g_spa[t] - states[i][t]):
                fails.append(f"G^additive[{i}][{t}]={g_add[t]} != G^sparse - Phi_prev = {g_spa[t]-states[i][t]}")
    # (b) position baseline does NOT cancel the state term -> additive != sparse (the bug, asserted)
    a_add_pos = group_advantages(add_g, baseline="position")
    a_spa_pos = group_advantages(spa_g, baseline="position")
    if all(_approx(x, y) for i in range(len(phis)) for x, y in zip(a_add_pos[i], a_spa_pos[i])):
        fails.append("position baseline unexpectedly made additive == sparse (F6 premise broken)")
    # (c) state-stratified baseline DOES cancel it -> additive == sparse cell-wise (the fix)
    a_add_str = group_advantages(add_g, baseline="state_stratified", states=states)
    a_spa_str = group_advantages(spa_g, baseline="state_stratified", states=states)
    for i in range(len(phis)):
        for t, (x, y) in enumerate(zip(a_add_str[i], a_spa_str[i])):
            if not _approx(x, y, tol=1e-6):
                fails.append(f"stratified: A^additive[{i}][{t}]={x:.6f} != A^sparse={y:.6f} (PBRS control broken)")
    # (d) position baseline stays bit-for-bit identical to the historical default (no silent reproduction break)
    if group_advantages(add_g) != a_add_pos:
        fails.append("default baseline must remain 'position' and bit-identical")
    # (e) guards
    try:
        group_advantages(add_g, baseline="state_stratified")
        fails.append("state_stratified without states must raise")
    except ValueError:
        pass
    try:
        group_advantages(add_g, baseline="bogus")
        fails.append("unknown baseline must raise")
    except ValueError:
        pass

    for f in fails:
        print("  [FAIL]", f)
    if not fails:
        print("  [ok] dense telescopes to Phi_T; sparse fires once at tau")
        print("  [ok] return-to-go correct")
        print(f"  [ok] all-failed-partial group: sparse zero-grad=100%, dense zero-grad={fz_dense:.0%} "
              f"(dense still learns, sparse doesn't) <- Claim 1")
        print("  [ok] dense advantage ranks partial progress; sparse gets signal only on mixed-success")
        print("mt_grpo golden: ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
