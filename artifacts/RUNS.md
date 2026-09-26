# Run records

One directory per run: `run_meta.json` (configuration, models, revisions) and `progress.jsonl`
(one row per training step plus `eval` rows at the evaluation iterations). Raw per-turn traces are
not included; see `RAW_TRACES.md`. `MANIFEST.json` lists every record with its SHA-256, which
matches `paper/results/sources_manifest.json`.

## `artifacts/fullft_campaign/` — Table 1(A), Figures 3 and 5(a), Tables 4-6

| run | arm | seed | steps | eval every | loss | attacker | victim | arm-specific settings |
|---|---|---|---|---|---|---|---|---|
| `s10_sft` | sft | 10 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s10_sparse` | sparse | 10 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s11_sft` | sft | 11 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s11_sparse` | sparse | 11 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s12_sft` | sft | 12 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s12_sparse` | sparse | 12 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s13_sft` | sft | 13 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s13_sparse` | sparse | 13 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s14_sft` | sft | 14 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s14_sparse` | sparse | 14 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s15_sft` | sft | 15 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s15_sparse` | sparse | 15 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s16_sft` | sft | 16 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s16_sparse` | sparse | 16 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s17_sft` | sft | 17 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s17_sparse` | sparse | 17 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_curriculum` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s2_dense` | dense | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_G12` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s2_hindsight` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | hindsight_m=2 |
| `s2_sft` | sft | 2 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_sil` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | self_imitate=2 |
| `s2_sparse` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_vip` | sparse | 2 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | adaptive_rollout=True |
| `s3_curriculum` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s3_dense` | dense | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_G12` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s3_hindsight` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | hindsight_m=2 |
| `s3_sft` | sft | 3 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_sil` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | self_imitate=2 |
| `s3_sparse` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_vip` | sparse | 3 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | adaptive_rollout=True |
| `s4_curriculum` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s4_dense` | dense | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_G12` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s4_hindsight` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | hindsight_m=2 |
| `s4_sft` | sft | 4 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_sil` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | self_imitate=2 |
| `s4_sparse` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_vip` | sparse | 4 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | adaptive_rollout=True |
| `s5_curriculum` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s5_dense` | dense | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_G12` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s5_hindsight` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | hindsight_m=2 |
| `s5_sft` | sft | 5 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_sil` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | self_imitate=2 |
| `s5_sparse` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_vip` | sparse | 5 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | adaptive_rollout=True |
| `s6_curriculum` | sparse | 6 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s6_dense` | dense | 6 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s6_G12` | sparse | 6 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s6_hindsight` | sparse | 6 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | hindsight_m=2 |
| `s6_sft` | sft | 6 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s6_sparse` | sparse | 6 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_curriculum` | sparse | 7 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | m_curriculum=True |
| `s7_dense` | dense | 7 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_G12` | sparse | 7 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s7_sft` | sft | 7 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_sparse` | sparse | 7 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s8_G12` | sparse | 8 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B | G=12 |
| `s8_sft` | sft | 8 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s8_sparse` | sparse | 8 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s9_sft` | sft | 9 | 16 | 2 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s9_sparse` | sparse | 9 | 16 | 2 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |

## `artifacts/sft_dynamics/` — Table 1(B-C), Figure 4 (goal group), Figure 5(b-d)

| run | arm | seed | steps | eval every | loss | attacker | victim | arm-specific settings |
|---|---|---|---|---|---|---|---|---|
| `s2_rl_pos` | rl_pos | 2 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_sft` | sft | 2 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_sft_all` | sft_all | 2 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s2_sparse` | sparse | 2 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_rl_pos` | rl_pos | 3 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_sft` | sft | 3 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s3_sparse` | sparse | 3 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_rl_pos` | rl_pos | 4 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_sft` | sft | 4 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s4_sparse` | sparse | 4 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_rl_pos` | rl_pos | 5 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_sft` | sft | 5 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s5_sparse` | sparse | 5 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s6_rl_pos` | rl_pos | 6 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s6_sft` | sft | 6 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s6_sparse` | sparse | 6 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_rl_pos` | rl_pos | 7 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_sft` | sft | 7 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s7_sparse` | sparse | 7 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s8_sft` | sft | 8 | 32 | 8 | ce | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |
| `s8_sparse` | sparse | 8 | 32 | 8 | pg | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B |  |

## `artifacts/scale_9b_27b38_phase1/` — Table 2 (9B attacker vs 27B-FP8 victim)

| run | arm | seed | steps | eval every | loss | attacker | victim | arm-specific settings |
|---|---|---|---|---|---|---|---|---|
| `s2_dense` | dense | 2 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s2_sft` | sft | 2 | 16 | 8 | ce | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s2_sparse` | sparse | 2 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s3_dense` | dense | 3 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s3_sft` | sft | 3 | 16 | 8 | ce | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s3_sparse` | sparse | 3 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s4_dense` | dense | 4 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s4_sft` | sft | 4 | 16 | 8 | ce | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s4_sparse` | sparse | 4 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s5_dense` | dense | 5 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s5_sft` | sft | 5 | 16 | 8 | ce | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |
| `s5_sparse` | sparse | 5 | 16 | 8 | pg | Qwen/Qwen3.5-9B | Qwen/Qwen3.8-27B-FP8 |  |

## `docs/results/sft-dynamics-4b-9b/`

Saved paired analyses of the mechanism study (`_T12-full`: OFC vs Sparse at iterations 8/16/24/32;
`_side-full`: Positive-only RL vs Sparse; the `-1632` files restrict to iterations 16 and 32), with the
20,000-resample seed-bootstrap intervals used in the paper.
