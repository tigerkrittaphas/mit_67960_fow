# Practical feasibility of multi-advisor deep RL for Fog-of-War chess (1x RTX A6000 + 32 CPU cores, few weeks; deploy on 2 CPU cores / 4 GiB at 180s+2s)

Notes on local measurements: some numbers below come from benchmarks I ran on the project workstation on 2026-11-18. Their "source" is the exact configuration: Intel Xeon w5-2465X (32 logical CPUs), Python 3.12.3, `chess==1.11.2` (the same pin as `starter/requirements.txt`), onnxruntime 1.32.0 with `intra_op_num_threads=2`, `inter_op_num_threads=1`, and the process pinned with `taskset -c 0,1`. The scripts were in an ephemeral scratchpad, so their key logic is given inline. The course runtime ships ONNX Runtime 1.24.2, not 1.32, and runs on "2 physical cores / 4 hardware threads", which is probably different and slower hardware. CPUs 0 and 1 on this Xeon may be SMT siblings, and it may have AVX-512/AMX. Treat every local latency as an **optimistic bound** for the tournament box.

Course runtime facts used below come from `/mnt/nvme2tb/krittaphas/mit_67960_fow/starter/README.md`:
- CPU only, no GPUs.
- 2 physical cores / 4 hardware threads, 4 GiB per bot. `/tmp` and the workspace count toward that memory.
- 180 s + 2 s per move, fresh process every game, and both bots share the same cores.
- NumPy 2.3.5, PyTorch 2.10.0+cpu, ONNX Runtime 1.24.2 (Python and native C/C++ SDK at `/opt/onnxruntime`), ONNX 1.20.1.
- The upload limit is 128 MiB compressed / 512 MiB expanded.
- The `starters/neural` example's `build.sh` runs `python3 model.py` to export PyTorch to ONNX, and `run.sh` sets `OMP_NUM_THREADS=2`.

## Q1. Sample/compute budgets for RL in chess-like / imperfect-info board games at small scale

### Takeaway
The only published deep-RL result in a chess-like imperfect-information game at single-GPU scale is the RBC agent of Bertram et al. (2022). It used supervised pretraining on human games, then PPO self-play against past snapshots, with no search, over "a few days" on one Tesla M40. That took it from 1118 to 1330 Elo on the RBC leaderboard, a mid-table result (rank 27/84). Search-based AlphaZero-style RL at small scale works on small boards: 9x9 Go with Pgx and Gumbel AlphaZero reached ~105M frames in ~8 h on one A100. Full chess still costs on the order of 10^9 frames even with Gumbel. A few weeks on one A6000 is therefore realistic for "SL then PPO self-play" without search. It is not realistic for tabula-rasa AlphaZero on full 8x8 FoW chess.

### Cited Findings
- **RBC (Reconnaissance Blind Chess) SL+PPO agent.** [Bertram, Fürnkranz & Müller 2022, arXiv:2208.02029](https://arxiv.org/abs/2208.02029)
  - **Pipeline:** SL on human games (the Penumbra dataset), then PPO self-play. The network plays policy-only with no search. All experiments "used only a few days of training on a single Tesla M40".
  - **Input:** 90 planes of 8x8 per observation, with a history of the **last 20 observations**, giving a 1800x8x8 input.
  - **Network:** a shared conv trunk with sense, move and value heads.
  - **Move output:** 73x8x8 plus a "pass" action, for 4673 moves.
  - **SL results:** 5 epochs took ~8 h wall time. Accuracy was 48.34% for moves and 49.71% for senses on held-out data, and the model began to overfit after that.
  - **SL agent strength:** 1118 Elo on the leaderboard, similar to the Stockfish-based "Trout" baseline.
  - **After PPO:** 1330 Elo, rank 27 of 84, with "no indication of convergence". During training the RL agent beat the SL agent >80% of the time.
  - **Reward:** sparse, with +1 given to the winning move and the preceding sense.
- **KataGo.** [Wu 2019, arXiv:1902.10565](https://arxiv.org/abs/1902.10565)
  - Surpassed ELF OpenGo's final model after 19 days on at most 28 V100s (about 1.4 GPU-years), generating 241M training samples across 4.2M games.
  - That is about a **50x** reduction versus ELF (2000 V100s for ~13–14 days, ~74 GPU-years) and at least 10x versus Leela Zero. AlphaZero used ~41 TPU-years.
  - **Techniques:** playout cap randomization, policy target pruning, global pooling, auxiliary policy targets (opponent's reply), and ownership/score targets.
- **Pgx Gumbel AlphaZero baselines.** [Koyamada et al. 2023, arXiv:2303.17503](https://arxiv.org/html/2303.17503v4)
  - **Setup:** 6 ResNet blocks (smaller than AlphaZero), **32 simulations** per move in self-play, self-play batch 1024 x 256 steps per iteration, training minibatch 4096.
  - **Length:** 400 iterations, about **105M frames**. For 9x9 Go this was about **8 h on a single A100**.
  - **Games trained:** Animal Shogi, Gardner (5x5) chess, 9x9 Go, Hex and Othello. **Full chess and 19x19 Go were not trained.**
  - **Strength:** on 9x9 Go, the 200-iteration (~52M frames) model with 800 simulations at evaluation went 62–38 vs Pachi at 10k simulations.
  - Self-play took >90% of wall time. Moving from 1 to 8 A100s reached the same performance about 4x faster.
- **Gumbel MuZero/AlphaZero, few simulations.** [Danihelka et al. ICLR 2022, text copy](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2022/policy_improvement_by_planning_with_gumbel__15d5fc26.txt); [ICLR page](https://iclr.cc/virtual/2022/poster/6418)
  - "Gumbel MuZero learns reliably even with 2 simulations" on 9x9 Go, whereas "MuZero fails to learn from 16 or less simulations".
  - Training speedup versus MuZero at n=200 is 5.9x at n=32, 11.3x at n=16, 16.2x at n=8 and 24.3x at n=4.
  - The chess plots (n=400) run to about **1600M frames**, and the Go plots to 0–800M frames.
- **Obscuro, the current state of the art in FoW chess, uses no learning.** [Zhang & Sandholm, arXiv:2506.01242 / ICLR 2026](https://arxiv.org/html/2506.01242)
  - It is "completely based on real-time search", with learned evaluation left to future work. (Details are under Q6.)

### Inferences
- Budget arithmetic for the stated machine: our local python-chess FoW-style pseudo-legal playout ran at about 52k plies/s per core (Q2). On 30 worker cores that is about 1.5M plies/s for a raw simulator, or about 5.4B plies/day with no network inference. **The bottleneck will be NN inference and observation encoding, not move generation.** A policy-only PPO actor batched on the A6000 could plausibly reach 10^4–10^5 decisions/s. Over 2 weeks that is about 10^10 to 10^11 decisions, nominally in the range of the Gumbel chess curves (about 1.6x10^9 frames). This is unverified: there is no published PPO-on-FoW-chess learning curve.
- Training five separate per-reward agents (material, king safety, scouting, pressure, win) divides that budget by about 5. The RBC evidence (a few M40-days for +212 Elo, starting from SL) suggests that each advisor would need SL or heuristic pretraining to become competent in a few A6000-days. A shared trunk with multiple value/policy heads would save about 5x compute versus five independent networks. That is an inference, not a cited result.

### Gaps
- No PPO-on-standard-chess or PPO-on-dark-chess learning curve with sample counts was found.
- I did not obtain the exact Elo values in the Gumbel chess figure (the figures are not readable in the text copy).
- I did not find the number of self-play games or PPO hyperparameters for Bertram et al. The paper does not report them in the text I extracted.

## Q2. Simulator throughput: python-chess vs C++/Rust generators vs vectorized envs

### Takeaway
python-chess handles about 28k random-playout plies/s per core with full legal-move generation and game-over checks, and about 52k plies/s with pseudo-legal generation, which is what FoW chess uses since it is king-capture based. C++/Rust generators are about 3–4 orders of magnitude faster on perft. With 30 cores, python-chess alone is enough for policy-only RL. It becomes the bottleneck only for search-heavy (MCTS/determinization) data generation.

### Cited Findings
- **Local python-chess benchmark** (config above; 5 s loops, random moves, reset at 300 plies or game end):
  - `list(board.legal_moves)` + `push` + `is_game_over()`: **27,793 plies/s** on one core.
  - `list(board.pseudo_legal_moves)` + `push` (king-capture termination): **51,908 plies/s** on one core.
  - perft(4) from the start position (197,281 leaves) took 0.69 s, or **~285k leaf nodes/s**, using legal generation plus push/pop.
- **chess-library (C++, Disservin):** perft on a Ryzen 9 5950X with movelist preallocation reaches about **270M–450M nodes/s** on one thread for standard chess, for example depth 5 in 430 ms at 449M NPS. It has no Python bindings. — [GitHub Disservin/chess-library](https://github.com/Disservin/chess-library)
- **shakmaty (Rust):** perft 4 ≈ 1.0 ms and perft 5 ≈ 24.1 ms on an i7-6850K, against 0.8 ms / 18.6 ms for the `chess` crate. perft(5) = 4,865,609 leaves, which works out to about 200M nodes/s (my calculation). — [docs.rs shakmaty](https://docs.rs/crate/shakmaty/0.29.3)
  - Perft is unfair when comparing legal and pseudo-legal generators. — [chess_bench](https://github.com/alex65536/chess_bench)
- **Pgx (JAX):**
  - It is "at least 10x faster" than PettingZoo/OpenSpiel Python loops on 1 A100 at batch 1024, and about 100x faster on 8 A100s (batch 8192).
  - The slowest Pgx env still reaches about 10^5 samples/s on one A100.
  - Its chess env is **perfect-information**: observation 8x8x119, actions 4672. A FoW version would need custom masking of the observation.
  - Source: [Koyamada et al. 2023](https://arxiv.org/html/2303.17503v4)
- **EnvPool (C++ batched envs):** about 1M Atari frames/s and 3M MuJoCo steps/s on a DGX-A100, and 2.8x faster than Python subprocess vectorization on a laptop. It does not include chess. — [Weng et al. 2022, arXiv:2206.10558](https://arxiv.org/abs/2206.10558)
- **OpenSpiel:**
  - It lists Kriegspiel ("Chess with opponent's pieces unknown") and Dark Hex. — [OpenSpiel games list](https://openspiel.readthedocs.io/en/latest/games.html)
  - A 2024 PIMC paper says Dark Chess "is implemented in OpenSpiel". That is secondhand, and I did not verify a `dark_chess` entry in the docs. — [arXiv:2408.02380](https://arxiv.org/pdf/2408.02380)

### Inferences
- The python-chess numbers exclude constructing the FoW observation (visible squares from own pieces' attack/move sets) and tensor encoding. Expect these to cut throughput further, plausibly by 2–5x in pure Python (unmeasured). A NumPy/bitboard encoder, or a small Rust/C++ extension (shakmaty / chess-library), would remove this cost.
- For PPO, at roughly 30k decisions/s on a GPU, env stepping across 30 python-chess processes (~0.5–1.5M plies/s raw) is not the bottleneck. For determinized search teachers (Q6), a C++/Rust core would matter.
- Pgx's GPU-vectorized chess is attractive for throughput, but FoW rules would have to be reimplemented in JAX. That is weeks of risk for a course project.

### Gaps
- No published python-chess perft/NPS benchmark exists. Only the local measurement above is available.
- I did not measure the throughput of OpenSpiel's dark chess or kriegspiel.
- I did not measure the cost of FoW observation encoding.

## Q3. Action-space handling (4672 vs 4096, masking)

### Takeaway
Use the AlphaZero-style 73x8x8 = 4672 encoding with a **convolutional** policy head and legal-move masking. Masking is clearly superior to invalid-action penalties, which "do not scale". A dense 4672-way FC head is a large and avoidable latency and parameter cost on CPU.

### Cited Findings
- **4672 encoding:** 64 source squares x 73 move types (56 queen-like, 8 knight, 9 underpromotions), as used by Pgx and AlphaZero. — [Pgx paper](https://arxiv.org/html/2303.17503v4)
- **RBC agent:** used 73x8x8 plus pass (4673) and **did not mask illegal moves** during SL, which "adds significant difficulty". — [Bertram et al. 2022](https://arxiv.org/abs/2208.02029)
- **Masking vs penalty.** Huang & Ontañón (μRTS, maps from 4x4 to 24x24): [arXiv:2006.14171](https://arxiv.org/html/2006.14171v3)
  - With masking, all maps reach return 40 (solved).
  - With invalid-action penalties, only 4x4 is solved. On 10x10 and larger, returns are at most about 1.0, and agents struggle "to even find the very first reward".
  - Removing the mask at evaluation drops return to about 17–34 but still beats the penalty agents.
  - "Naive" masking, which masks during rollout but not in the gradient, still solves the task but with much higher KL divergence.
  - Code: [vwxyzjn/invalid-action-masking](https://github.com/vwxyzjn/invalid-action-masking)
- **Local ONNX benchmark, head-design effect** (2 threads, batch 1; trunk of 3x3 conv residual blocks):
  - 6 blocks x 64 channels with a dense FC to 4672: **19.6M params, 2.20 ms**.
  - The same trunk with a 1x1 conv head to 73 planes: **0.47M params, 0.36 ms**.
  - The dense head adds about 19M params (64 ch x 64 sq x 4672) and about 6x latency.

### Inferences
- In FoW chess the player knows its own legal moves exactly: pseudo-legal moves, since it is king-capture with no check rule. Masking is therefore exact and cheap, unlike RBC, where moves can fail.
- A 4096 from-to encoding with promotions forced to queen loses only underpromotions, which matter little. It allows a simple 64x64 bilinear or conv head. 4672 is the safer standard. (This is a judgment call, not a cited result.)

### Gaps
- I found no paper that directly compares the 4672 and 4096 encodings for learning speed.

## Q4. Opponent pools / league / fictitious self-play for small budgets

### Takeaway
A cheap, proven recipe exists in the closest domain. In RBC, Bertram et al. snapshot the learner whenever it reaches a **65% win rate** against the pool and sample past versions with probability **p_i = 1 − w_i / Σ_j w_j**, where w_i is the win rate over the last **K=500** games. They describe this as approximating "the average strategy of past best-responses" (fictitious-play-like). Full AlphaStar leagues, PSRO and NFSP are much heavier.

### Cited Findings
- **Bertram et al. opponent pool:**
  - Past versions are saved when they reach a 65% win rate.
  - Opponents are sampled as p_i = 1 − w_i/Σw_i, with w_i averaged over the K=500 most recent results.
  - The pool is described as "similar to playing against the average strategy of past best-responses".
  - Argmax play at test time was "slightly better" than sampling, even though theory suggests the optimal FoW policy should be stochastic.
  - Source: [arXiv:2208.02029](https://arxiv.org/abs/2208.02029)
- **KataGo evaluation sampling:** opponents are sampled with frequency proportional to the predicted variance p(1−p) of the outcome, to make each game maximally informative for Elo estimation. — [Wu 2019](https://arxiv.org/abs/1902.10565)
- **Pgx/Gumbel AZ:** pure latest-network self-play was enough in perfect-information games. — [Pgx paper](https://arxiv.org/html/2303.17503v4)

### Inferences
- For five reward-specific advisors, a minimal league could include:
  - the current learner,
  - a prioritized pool of its past snapshots (Bertram's rule),
  - fixed exploiters already available: `starter/baseline.py`, the random bot, and a Stockfish-on-sampled-board bot (like RBC's "Trout").

  This mirrors the main-agent-plus-exploiters idea at negligible cost. It is an inference. I did not fetch the AlphaStar paper.
- Non-stationarity: shaped advisors such as "material" or "scouting" trained against a moving opponent pool will each have different equilibria. Freezing the opponent pool per training phase reduces drift.

### Gaps
- I did not fetch primary sources for the AlphaStar league (Vinyals et al. 2019), PSRO (Lanctot et al. 2017) or NFSP (Heinrich & Silver 2016), so no quantitative claims about them are made here.

## Q5. Partial observability: recurrence vs history stacking; privileged critics / oracle guiding

### Takeaway
In RBC, a **stack of the last 20 observations** fed to a CNN matched LSTM/GRU accuracy with less training time. Using the true board during training is well supported in imperfect-information games. Suphx feeds perfect information to the actor and then drops it out gradually ("oracle guiding"). PerfectDou feeds it only to the critic, so nothing has to be removed at inference.

### Cited Findings
- **History stack vs recurrence:** "a sequence-based architecture, e.g. LSTM or GRU... required more training time without increasing the accuracy of the predictions", so a 20-turn history was used. — [Bertram et al. 2022](https://arxiv.org/abs/2208.02029)
- **Suphx oracle guiding:**
  - The oracle agent sees opponents' private tiles and the wall.
  - Direct knowledge distillation from the oracle to the normal agent "did not work well".
  - Instead, the perfect features are gradually dropped (a decay from 1 to 0) during RL until the agent is a normal agent.
  - Source: [Suphx, arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
- **PerfectDou:** contrasts its approach with Suphx. In PerfectDou, perfect information feeds the **critic** (Perfect-Training-Imperfect-Execution), so it need not be removed before inference. — [PerfectDou supplement, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/e26f31de8b13ec569bf507e6ae2cd952-Supplemental-Conference.pdf)

### Inferences
- For FoW PPO, a natural design is an asymmetric actor-critic. The actor gets the FoW observation plus history. The critic additionally gets the true board, or even a Stockfish evaluation of it. This should cut value-estimate variance substantially. Since the critic is discarded at deploy time, there is no CPU cost.
- Shaped rewards such as material and king safety can be computed from the true board during training.
- Frame stacking of about 8–20 FoW observations (the RBC evidence) is the cheaper first choice. Recurrence complicates PPO batching and ONNX export.

### Gaps
- I did not fetch Pinto et al. 2017 (asymmetric actor-critic), R2D2 or DRQN, so their numbers are not reported.
- I found no FoW-specific ablation of a privileged critic.

## Q6. More sample-efficient alternatives (SL on logs, determinized-search teacher, heuristic imitation, Stockfish bootstrapping)

### Takeaway
The strongest FoW chess agent, Obscuro, uses **no learned model**. It does belief-state sampling plus game-theoretic search with **Stockfish 14 depth-11 leaf evaluations**, and it is superhuman at 3+2 on a 6-core desktop. In RBC, supervised pretraining reached the level of a Stockfish baseline before RL. Bootstrapping from Stockfish evaluations on sampled true boards is therefore the best-evidenced route to sample efficiency, but deployment compute is the constraint.

### Cited Findings
- **Obscuro (Zhang & Sandholm 2025, ICLR 2026):** [arXiv:2506.01242](https://arxiv.org/html/2506.01242)
  - **Method:**
    - Leaf evaluation uses Stockfish 14 in MultiPV at depth 11, clamped to [−1, 1].
    - It keeps the full set of consistent positions, "usually at most about 10^6".
    - It samples "at most a few hundred" positions per move.
    - Search uses 1 equilibrium thread and 2 expansion threads.
  - **Human games:** played at **3 min + 2 s** on a single **6-core Intel i5** desktop.
  - **Results:**
    - 85.1% (+834 =33 −133) over 1,000 games vs the prior state of the art, ZS21 (knowledge-limited subgame solving).
    - 97% vs humans rated 1450–2006.
    - **16–4** vs the #1 chess.com FoW player (rated 2318), about +241 Elo, p=0.0118.
- **RBC SL stage:** SL alone matched the Stockfish-based Trout baseline (1118 Elo), and RL added about 212 Elo. — [Bertram et al. 2022](https://arxiv.org/abs/2208.02029)
- **Suphx:** naive distillation from a perfect-information oracle to an imperfect-information student "did not work well", so they used gradual feature dropout instead. — [Suphx](https://arxiv.org/abs/2003.13590)

### Inferences
- **Cheapest strong teacher.** Sample true boards consistent with the FoW observation, evaluate each with Stockfish (or a small engine), and aggregate. This is a determinized / PIMC teacher.
  - Label FoW observations with it, offline on 32 cores.
  - Train the advisor networks by SL or value regression, then fine-tune with PPO plus a snapshot pool.
  - This mirrors Obscuro's eval and RBC's SL-then-RL pipeline.
- **Deployment.** Obscuro's setup (6 cores, ~10^6-state belief, Stockfish leaf evals) exceeds the 2-core / 4 GiB box, where both bots share cores. A distilled network that needs only a single forward pass or a shallow search is the practical deploy target.
- Whether a Stockfish binary can be shipped and run inside 128 MiB / 4 GiB in the course runtime must be checked against the rules.
- **Time budget.** At 180 s + 2 s with about 40–60 own moves, there are roughly 5–6 s per move. A 2M-param CNN at ~1 ms (Q7) allows thousands of evaluations per move. That makes shallow belief-sampled lookahead with the learned net feasible even on 2 cores.

### Gaps
- No published learned-evaluation FoW chess agent with sample counts was found.
- No numbers exist for the strength of PIMC + Stockfish teachers in FoW chess specifically. ZS21 and Obscuro are search-based, not distillation.
- Legality of shipping Stockfish under the course rules was not checked.

## Q7. ONNX Runtime CPU inference latency for ~2M-param CNNs on 2 threads

### Takeaway
On 2 threads, an AlphaZero-style ResNet of about 0.5–3M params with a conv policy head runs in **0.4–1.1 ms per single position** (about 900–2800 positions/s). This is far below the per-move budget, even allowing a 2–3x slowdown on the tournament CPU. Batching helps little on 2 threads.

### Cited Findings
- **Local ORT benchmark.**
  - **Network:** input 32 or 64 planes x 8x8. Each residual block is two 3x3 conv layers with ReLU and a skip connection. The policy head is a 1x1 conv to 73 planes, i.e. 4672 logits, with no value head.
  - **Method:** random weights; 20 warm-up runs; mean of 200 (batch 1) or 30 (batch 32) runs.

  | Blocks x channels | Params | Batch 1 | Batch 32 |
  |---|---|---|---|
  | 6x64 | 0.47M | 0.36 ms (2,778 pos/s) | 4.44 ms (7,206 pos/s) |
  | 8x96 | 1.36M | 0.79 ms (1,272 pos/s) | 13.4 ms (2,388 pos/s) |
  | 6x128 | 1.82M | 0.67 ms (1,482 pos/s) | 17.1 ms (1,870 pos/s) |
  | 10x128 | 3.03M | 1.12 ms (894 pos/s) | 27.5 ms (1,163 pos/s) |
  | 10x192 | 6.76M | 2.41 ms (415 pos/s) | 63.3 ms (505 pos/s) |

  - With a dense FC policy head (64·ch → 4672), param counts become 19.6M–41M and batch-1 latency 2.2–5.6 ms.
- **Runtime:** the course runtime provides ORT 1.24.2 (Python plus native C/C++ SDK) and PyTorch 2.10.0+cpu. The neural starter exports PyTorch to ONNX at build time and sets `OMP_NUM_THREADS=2`. Each game is a fresh process, so model load and session creation cost is paid per game. — `/mnt/nvme2tb/krittaphas/mit_67960_fow/starter/README.md`, `starter/starters/neural/{build.sh,run.sh}`

### Inferences
- A five-advisor ensemble of about 2M-param nets is about 5x 0.7 ms ≈ 3.5 ms per position if run separately, or about 1 ms with a shared trunk and five heads. Either fits easily in about 5 s per move. The weights (about 8 MB each in fp32) are trivial against 4 GiB and 128 MiB.
- Python-side overhead (python-chess move generation, observation encoding, ORT call overhead of about 50–100 µs) will likely dominate per-node cost in any search. That is an inference and was not measured.

### Gaps
- The benchmarks were not run on the actual tournament CPU or on ORT 1.24.2. Latencies there may be 1.5–3x higher (unverified).
- int8 quantization speedups were not measured.
- Session-creation and model-load time per fresh process was not measured.
