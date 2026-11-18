# Results: small deep-RL council test (2026-11-18, n = 1 seed per arm, 15M agent steps each)

All numbers are scores (wins + half of draws) over 400 games, with 95% confidence intervals. The agent plays White with the softmax policy. Every game ended by king capture, so there were 0 draws in every arm.

| Arm | vs baseline | vs random | Mean plies vs baseline |
|---|---|---|---|
| W (win advisor only) | **0.638 ± 0.047** | **0.830 ± 0.037** | 21.8 |
| G (egocentric council) | 0.270 ± 0.044 | 0.700 ± 0.045 | 34.6 |
| E (empathic council) | 0.198 ± 0.039 | 0.530 ± 0.049 | 18.3 |
| E-fit (E with NNLS weights) | 0.215 ± 0.040 | 0.653 ± 0.047 | 19.7 |

## Alignment error ε (arm E: win head regressed on the four advisor heads, 3,830 held-out states)

| Fit | NNLS weights (material, safety, scout, pressure) | Test R² | p95 / max abs residual | Std of the win head |
|---|---|---|---|---|
| Global, with intercept | 0, 0.30, 0.30, 0.29 | 0.43 | 0.158 / 1.47 | 0.122 |
| Within each state | 0, 0.35, 0.16, 0.31 | 0.29 | 0.116 / 1.21 | 0.080 |

- The fitted council and the win head pick the same best move in only **45%** of states.
- The win head's mean spread between the best and worst move is 0.29.
- NNLS gives the material advisor **zero** weight.

## Against the pre-registered hypotheses
- **Positive control: passed.** W beats the baseline, with the CI lower bound at 0.59 > 0.5.
- **H1 (the council does not beat a single aligned advisor): supported.** E, E-fit and G are all far below W against both opponents. Fitting the weights to the win head (E-fit) barely helps: 0.198 → 0.215 against the baseline.
- **H2 (egocentric is worse than empathic): not supported, and reversed.** G beat E against both opponents (0.27 vs 0.20 against the baseline, 0.70 vs 0.53 against random). There was no attractor signature: no repetition or ply-limit draws. G's games were longer (34.6 vs 18.3 plies), but they still ended in king capture.
- **H3 (ε is large): supported.** The within-state R² is 0.29, below the 0.5 threshold. The second criterion was not met: the p95 residual (0.116) is smaller than the win-head spread (0.29). Even so, the fitted council matches the win head's best move only 45% of the time.

## Caveats
- One seed per arm. White only. Weak opponents. A small network and budget.
- The reward scales, λ = 10 and the hand-set council weights were chosen once and never tuned. The safety and scout heads ended up dominating the Q-values, with mean |Q| of about 1.3–1.6.
- The legal-move mask comes from the true board. It matches what can be inferred from observations, except that en passant needs the observation from the previous ply.
- The analysis started about 30 minutes late because of a bug in the wait loop (a `pgrep` that matched itself). The results are unaffected.
