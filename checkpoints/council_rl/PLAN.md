# Pre-registration: small deep-RL council test (written 2026-11-18, before the main runs)

## Setup
- Env: `experiments/council_rl/fowenv.py`. The rules come from `starter/referee.py`.
  - The agent plays White only. The opponent is the course baseline (Black), running in-process.
  - Promotions are to queen only. Games are capped at 300 plies, which counts as a draw.
- Input: the current and previous own-turn observations, encoded as 28 planes.
  - The legal mask comes from the true board. It was checked against observation-only `candidates()` on 10k positions, and every mismatch was an en-passant capture, which is inferable from the per-ply observations.
- Advisor rewards (privileged, true board, per agent step):

  | Advisor | Reward |
  |---|---|
  | material | Δ(material balance) / 10 |
  | safety | −(enemy attacks on own king zone) / 10, and −1 on losing by king capture |
  | scout | (number of `?` squares revealed) / 20 |
  | pressure | (own attacks on enemy king zone) / 20 |
  | win | +1 / 0 / −1 at the end of the game |

  Scout and pressure are level rewards, not potential-based. This is intentional: those advisors should be misaligned.
- Network: K separate 4-layer CNNs with 64 channels each, one per advisor. Each outputs a 4096 from-to Q-value per head.
- Training: DQN with 3-step returns, γ = 0.99, Adam at 2.5e-4, batch 512, 2 updates per iteration, hard target update every 2000 updates, replay buffer of 1M.
- Behaviour policy: softmax(λ · Σ wₖQₖ) with λ = 10, plus ε = 0.02 uniform exploration.
- Council weights: material 0.4, safety 0.3, scout 0.1, pressure 0.2.
- Budget: 15M agent steps per arm, seed 0, with identical settings across arms. This is n = 1 per arm, and results are labelled as such.

## Arms

| Arm | Heads | Bootstrap target | Vote |
|---|---|---|---|
| W | win | max | win = 1 |
| G | 4 advisors | each its own max ("egocentric") | council weights |
| E | 4 advisors + win | expected-SARSA under π_w ("empathic") | council weights, win = 0 |
| E-fit | E's checkpoint | — | NNLS weights fitting E's win head on the 4 advisors (eval only) |

## Evaluation
- Final evaluation: 400 games against the baseline and 400 against uniform random, using the softmax policy (the proposal's principal).
- Metrics: score = (W + D/2) / N with a 95% CI, termination reasons, and mean plies.
- ε: NNLS fit of Q_win^π on the four advisor Q^π (arm E), on held-out states from evaluation games. Report R² and the maximum and 95th-percentile absolute residual, both globally (with an intercept) and within each state.

## Hypotheses and decision rules
- **Positive control:** if W does not beat the baseline (score CI lower bound > 0.5), the experiment is **inconclusive**. It is not evidence against the council.
- **H1 ("council does not beat one aligned advisor"):** supported if E-fit and E are both ≤ W, judged by score CI. **Refuted** if E or E-fit is significantly above W. That would be reported as such, since dense rewards may help at small compute.
- **H2 ("egocentric training is worse than empathic"):** supported if G < E significantly, especially with more draws by repetition or the ply limit (the attractor signature).
- **H3 ("alignment ε is large"):** supported if the within-state R² < 0.5, or the 95th-percentile residual is greater than the spread of Q_win across moves.
