# Raw traces

Each run directory of the source repository also holds `turns.jsonl` and `eval_turns.jsonl`. They
are the evidence of record for the paper: every attacker generation is stored with the complete
prompt it was produced from and the victim's reply, one JSON object per trajectory:

```
{"ts": ..., "kind": "generation" | "eval", "step": k, "arm": ..., "seed": ..., "goal": <task id>,
 "phi_trace": [Φ_1, ..., Φ_T], "count_trace": [c_1, ..., c_T], "success": bool,
 "response": {"turns": [{"t": 1, "phi": ..., "phi_true": ..., "prompt": ..., "attacker": ..., "victim": ...}, ...]}}
```

The Figure 4 case study and its cohort counts were recomputed from these files with
`code/src/mt_grpo.py` (`per_turn_rewards`, `group_advantages(baseline="state_stratified")`);
the recomputed fraction of zero-advantage decisions equals the `frac_zero_adv` field logged in
`progress.jsonl` at every iteration.

The traces are complete multi-turn attack conversations and are therefore not part of this public
release. They are available to researchers on request for verification and replication.
