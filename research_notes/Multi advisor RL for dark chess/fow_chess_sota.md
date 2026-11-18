# State of the Art for Fog-of-War (Dark) Chess and Related Imperfect-Information Board Games

Context: these notes check a proposed design against established methods. The design uses separate deep-RL "advisors" with hand-designed rewards (material, king safety, scouting, pressure, win). Their per-move Q-values are combined as a weighted sum, and a move is drawn from a softmax over that sum. Runtime limits: CPU only, 2 cores, 4 GiB, 180 s + 2 s per game. Training budget: one RTX A6000 and 32 CPU cores for a few weeks. Current date: Nov 2026.

## Q1. Zhang & Sandholm 2021, "Subgame solving without common knowledge" (ZS21 dark chess bot)

### Takeaway
ZS21 (NeurIPS 2021) was the first strong dark-chess AI. It uses pure real-time search: depth-limited 1-KLSS subgame solving over sampled positions, with a Stockfish value function at the leaves and no learned blueprint. It runs on a 6-core machine, and the authors estimate its strength at roughly 2000 on chess.com FoW. That estimate is their own rough one.

### Cited Findings
- Problem addressed: standard safe subgame solving needs the full common-knowledge closure of an infoset, which in dark chess is too large to enumerate or store. k-KLSS ("knowledge-limited subgame solving") reasons only about low-order knowledge (k = 1: our infoset plus opponent infosets that touch it) and prunes unreachable nodes. — [arXiv 2106.06068](https://arxiv.org/abs/2106.06068); [NeurIPS 2021 paper](https://proceedings.nips.cc/paper_files/paper/2021/file/c96c08f8bb7960e11a1239352a479053-Paper.pdf)
- The authors prove that naive KLSS can *increase* exploitability relative to the blueprint, and they propose three ways to restore safety (e.g., folding subgame results back into the blueprint, or limiting the infosets where solving happens). — [arXiv 2106.06068](https://arxiv.org/abs/2106.06068)
- Implementation: depth-limited 1-KLSS, re-solved at every turn (nested). It uses reach-maxmargin, falling back to reach-resolve when margins cannot all be made positive. The tree trunk is solved exactly (LP) with iterative deepening. Leaf values come from Stockfish 13 at depth 11, clamped via tanh(x/τ) with τ = 6 pawns. There are at least 200 root particles and an information-discovery penalty (M = 10^7). — [ar5iv 2106.06068](https://ar5iv.labs.arxiv.org/html/2106.06068)
- Hardware: "Our agent runs on a single machine with 6 CPU cores." — [ar5iv 2106.06068](https://ar5iv.labs.arxiv.org/html/2106.06068)
- Strength:
  - It beat an ablation (depth-1 imperfect-information search plus Stockfish perfect-information iterative deepening) 59.5–40.5 over 100 games.
  - One author (about 1700 on chess.com FoW) "has played upwards of 20 games against the agent, winning only two and losing the remainder." In other words, the agent won about 90% of these games.
  - It lost 9–1 to FM Luis Chan (2416, top of the FoW blitz list).
  - The authors estimate about 2000 strength from these games.
  - They note it "threw away advantages in the endgame by playing too pessimistically."
  — [ar5iv 2106.06068](https://ar5iv.labs.arxiv.org/html/2106.06068)
- Noted limitation: nodes fall off the end of the depth-limited tree, so information is lost over time. Using a perfect-information value function at leaves (with no blueprint) is not theoretically sound. — [ar5iv 2106.06068](https://ar5iv.labs.arxiv.org/html/2106.06068)
- No open-source code is referenced on the arXiv page. — [arXiv 2106.06068](https://arxiv.org/abs/2106.06068)

### Inferences
- Even the 2021 "first strong" bot uses no RL at all. Its strength comes from (a) exact tracking of possible positions, (b) a strong perfect-information evaluator (Stockfish), and (c) game-theoretic, mixed-strategy search. The proposed advisor design has none of (a)–(c) explicitly, so its learned Q-values would have to absorb all three.
- ZS21 (about 2000) is a realistic stretch target for a course project. Beating it would be very hard for an RL-only agent.

### Gaps
- ZS21's exact time-per-move setting is not in the material I retrieved.
- No public code release was found for ZS21.

## Q2. Obscuro (Zhang & Sandholm, arXiv 2506.01242, June 2025; ICLR 2026)

### Takeaway
Obscuro is the first superhuman FoW chess AI. It runs entirely on real-time search with no training: exact enumeration of possible positions, KLUSS subgame solving over a few hundred sampled positions, one-sided GT-CFR with PCFR+, Stockfish 14 depth-1 MultiPV leaf evaluation, and *partial* purification of the mixed strategy. Against humans it ran on a 6-core consumer desktop at the same 3+2 time control. It scored 97% against players rated 1450–2006, 80% (16–4) against the #1 human, and 85.1% against ZS21. There is no official code release. A third-party GPL-3.0 reimplementation, misty-chess, appeared in Aug–Sep 2026.

### Cited Findings
- Paper: "General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess," B. H. Zhang & T. Sandholm, arXiv 2506.01242 (June 2025), ICLR 2026. — [arXiv PDF](https://arxiv.org/pdf/2506.01242v2); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9705eec26e511d7844a591ce10c33a3f-Abstract-Conference.html)
- The paper names four challenges of FoW chess, all quoted or paraphrased from §1:
  1. It is highly tactical, so it needs lookahead.
  2. Infoset size can swing from hundreds to millions and back within a few moves.
  3. "A strong agent must at least somewhat play a mixed strategy… Otherwise, an adversary who knows the strategy, or has learned the strategy from past observation, can easily exploit that knowledge."
  4. Common-knowledge sets are too large to reason about.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Algorithm:
  1. Maintain the full set P of positions consistent with observations (usually |P| ≤ 10^6).
  2. Each turn, sample a subset I of "no more than a few hundred positions."
  3. Build an imperfect-information subgame that reuses the previous move's tree.
  4. Solve it with PCFR+ (predictive CFR+) while expanding the tree with "one-sided GT-CFR," a variant of Schmid et al.'s growing-tree CFR.
  5. New nodes are valued by Stockfish 14 run in MultiPV mode at depth 1, which scores all children in one call, clamped to [−1, +1].
  6. Threads: one runs CFR and two expand the tree.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- KLUSS (vs 1-KLSS): opponent infosets that do not touch our infoset are not frozen but optimized jointly. This lets Obscuro keep nodes outside the current infoset alive for many moves, so it can "exploit long bluff opportunities." The authors argue that the safe variant of KLSS from Liu et al. 2023 constrains the solver too much. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Gadgets: it switches between Maxmargin and Resolve depending on whether the current strategy is "safe." Its Resolve uses a non-uniform root distribution rather than the uniform one. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Move selection, i.e. purification: rather than sampling the full CFR strategy, it samples only from the top m actions, 1 ≤ m ≤ 3. Mixing (m > 1) is allowed only when the strategy is judged safe, meaning its expected value is at least what was expected before the turn. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Results vs humans (3+2 time control, chess.com's most popular FoW control):
  - 117 games were played; 17 were excluded for disconnects or obvious throwing, leaving 100 completed games.
  - Obscuro scored 97% (+97 =0 −3) against opponents rated 1450–2006.
  - Against the #1 player on the chess.com FoW blitz list (rated 2318 on 16 Aug 2024), it won a 20-game match 16–4 (80%, +241 Elo, exact binomial p = 0.0118). The match ran over two days, so the human could prepare overnight.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Results vs AI:
  - Against ZS21: 85.1% (+834 =33 −133) over 1,000 games.
  - Against a random opponent: 1,000 straight wins.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Ablations at 5 s/move, each measured as full Obscuro's score against a copy with one technique turned off:

  | Technique turned off | Full Obscuro's score | Games |
  |---|---|---|
  | Purification (unrestricted mixing) | 70.2% | 1,000 |
  | KLUSS (reverting to 1-KLSS) | 58.0% | 1,000 |
  | One-sided GT-CFR (using two-sided) | 53.3% | 10,000 |
  | Non-uniform Resolve distribution | 53.3% | 10,000 |

  Two-sided GT-CFR alone, with every new technique off, still beat ZS21 72.6%. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- **Evaluation-function ablation (most relevant to the advisor design):**
  - "Simple-eval Obscuro" replaced Stockfish with a hand-crafted eval counting only material difference and number of visible squares.
  - Full Obscuro beat it 81.9% (+787 =63 −150).
  - Simple-eval Obscuro still beat ZS21 55.0% (+5258 =486 −4256) over 10,000 games.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Time scaling: there were 10,000-game matches between Obscuro at successive doublings of time per move, from fractions of a second up to 16 s/move. Per doubling, the score is about 56.5% at the low end and about 52.3% at the high end (e.g., 8 s→16 s: 52.3%). Note: pdftotext garbled the fractional time labels (they appear as "18 s/move", "14 s/move", …). I read them as 1/8, 1/4, 1/2 s; this is my interpretation of a garbled extraction. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Hardware: human matches ran "on a single desktop machine with a 6-core Intel i5 CPU." Ablations used 5 cores per side on an EPYC server. Statistics at 5 s/move:
  - average game length 116.6 plies
  - average search depth 10.7 plies
  - average search tree 1,070,552 nodes and 14,404 infosets (181k nodes carried over from the previous move)
  - average number of possible positions 17,264

  The stated AI-vs-AI time control excludes the time to enumerate legal positions. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- On mixed strategies:
  - FoW chess has no chance-dealt private information. So "if a player plays a pure strategy that the opponent knows, the opponent would essentially be playing regular chess."
  - Example: White can win almost a pawn in expectation by mixing a low-probability 2.Qa4 with a high-probability 2.Nc3.
  - Obscuro's first moves are mixed: White plays d4 66.4%, c4 29.6%, e4 1.9%, …; Black plays Nc6 32.5%, c6 25.1%, e6 20.7%, Nf6 15.9%, ….
  - Short losses to Qa4+/Qa5+ are described as "bluffs must sometimes get called."
  - In 10,000 self-play games White scored 57.5%.

  — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- The authors' suggested future directions: learn the evaluation with deep RL rather than using Stockfish, use continuation strategies, and sample infosets with an opponent model rather than uniformly. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Code: the paper links only the lichess game collections and gives no code repository. *misty-chess* is a third-party reimplementation:
  - Author Brian Liou; GPL-3.0-or-later; Python 3.11+ with a Rust extension (built via maturin).
  - Versions 0.1.0 (23 Aug 2026) through 0.1.3 (23 Sep 2026); repo github.com/brianhliou/misty.
  - It follows the Obscuro architecture: exact belief enumeration, knowledge-limited subgames, one-sided GT-CFR with PCFR+, Stockfish depth-1 leaves, and purification with a resolve gadget.
  - It makes no strength or Elo claim.
  - The pure-Python fallback is about 500× slower on the belief hot path, and mid-game belief sets can reach millions of positions.
  - Known weaknesses: the belief can lose the true board past its enumeration cap, and it can draw won endgames against a bare king.

  — [PyPI misty-chess](https://pypi.org/project/misty-chess/)
- Sandholm's CMU 15-888 (Fall 2025) Lecture 13 slides cover KLUSS and FoW chess. — [CMU slides](https://www.cs.cmu.edu/~sandholm/cs15-888F25/Lecture_13_KLUSS_fow-chess.pdf)

### Inferences
- **Feasibility on 2 cores:** Obscuro reached superhuman play on a 6-core desktop at exactly our 3+2-style clock (we have 180 + 2). Time-scaling costs only about 16–45 Elo per halving of compute. So a 2-core port of this kind of search should lose only on the order of 50–100 Elo from compute alone. This is an extrapolation, not a measurement.
- **Thread caveat:** Obscuro uses 3 threads (1 CFR, 2 expansion) on 6 cores. On 2 cores we would lose about two-thirds of the search threads, not just one-third of the core count.
- **Enumeration caveat:** the 5 s AI-vs-AI budget excludes belief-set enumeration. On our clock, enumeration time comes out of the 180 + 2 s, which matters when |P| spikes toward 10^6.
- **Memory:** average |P| ≈ 17k but up to about 10^6, plus a tree of about 1M nodes. Both should fit in 4 GiB only with compact board encodings in Rust or C++. The 500× Python slowdown reported by misty means the belief and search core cannot be pure Python.
- **Implication for advisor rewards:** simple-eval Obscuro, with a material-plus-visibility eval much like the proposed "material" and "scouting" advisors, still beat ZS21 (about 2000). So hand-crafted terms of this kind are fine as an evaluator *inside* a belief-aware game-theoretic search. The bottleneck of the proposed design is the missing belief tracking and equilibrium-style search, not the reward shaping. Eval quality still matters on top of search: full Obscuro beat simple-eval Obscuro 81.9%.
- **Implication for softmax mixing:** greedy (argmax) play is exploitable, as the Qa4+/Qa5+ example shows. But unrestricted mixing was also clearly worse: full Obscuro beat the no-purification version 70.2%. A temperature softmax over summed advisor Q-values is neither an equilibrium strategy nor a safety-gated purification. Its randomness is not tied to what makes a strategy hard to exploit, such as indifference among actions or bluff value. It can be both exploitable and needlessly noisy.
- misty-chess, Python-wrapped Stockfish, and a Rust belief core are likely the fastest route to a strong baseline. GPL-3.0 means any derived code shipped must also be GPL.

### Gaps
- The exact lichess URLs for the published game collections were truncated in extraction and are not reproduced here.
- No official Obscuro code was found.
- misty-chess strength is unverified.

## Q3. DeepNash / R-NaD, ReBeL, Student of Games: pure RL vs search; why deterministic play is exploitable

### Takeaway
DeepNash shows that pure model-free RL can reach human-expert level in an imperfect-information game (Stratego) *without* search. It does this only by converging toward a Nash equilibrium through a game-theoretic algorithm (R-NaD), not by maximizing hand-shaped rewards, and its policy still needed thresholding and discretization fixes. ReBeL and Student of Games show that RL combined with search over public belief states is the principled general recipe, with convergence to Nash in two-player zero-sum games. In both families, play is explicitly mixed.

### Cited Findings
- DeepNash (Perolat et al., arXiv June 2022; Science, Dec 2022) uses "a game-theoretic, model-free deep reinforcement learning method, without search" (R-NaD) and learns Stratego from scratch by self-play. — [arXiv 2206.15378](https://www.arxiv.org/abs/2206.15378); [DeepMind blog](https://deepmind.google/discover/blog/mastering-stratego-the-classic-game-of-imperfect-information/)
- Results:
  - Over 97% win rate against existing Stratego bots.
  - Against humans on Gravon (April 2022), 42 of 50 ranked matches won (84%), placing 3rd on both the 2022 and all-time rankings (ratings 1799 and 1778).

  — [arXiv 2206.15378 PDF](https://arxiv.org/pdf/2206.15378v1.pdf)
- R-NaD repeats three steps until a fixed point:
  1. reward transformation, regularizing toward a reference policy;
  2. dynamics, learning to the fixed point;
  3. update of the regularization policy.

  — [arXiv 2206.15378 PDF](https://arxiv.org/pdf/2206.15378v1.pdf)
- Why deterministic play fails: deterministic value-maximizing decisions are sufficient in perfect-information zero-sum games, but "this approach is theoretically unsound when dealing with imperfect information games." DeepNash generates "billions of unique deployments" so it cannot be exploited, and it bluffs both positively and negatively. — [arXiv 2206.15378 PDF](https://arxiv.org/pdf/2206.15378v1.pdf)
- Practical caveat: because the softmax policy "assigns a non-zero probability to every action," DeepNash was "slightly distorted by low-probability mistakes." It needed fine-tuning with "thresholding and discretization" of action probabilities, plus test-time techniques "to remove any remaining obvious mistakes." — [arXiv 2206.15378 PDF](https://arxiv.org/pdf/2206.15378v1.pdf)
- Search in Stratego was avoided because estimating private configurations in a public state makes existing search techniques intractable there. — [arXiv 2206.15378 PDF](https://arxiv.org/pdf/2206.15378v1.pdf)
- ReBeL (Brown, Bakhtin, Lerer, Gong, arXiv 2007.13544, 2020) combines self-play RL with search at both training and test time. It provably converges to a Nash equilibrium in any two-player zero-sum game, reduces to an AlphaZero-like algorithm in perfect-information games, and is superhuman in heads-up no-limit hold'em "using far less domain knowledge than any prior poker AI." — [arXiv 2007.13544](https://arxiv.org/abs/2007.13544)
- Student of Games (Schmid et al., Science Advances, Nov 2023):
  - It unifies search, learning, and game-theoretic reasoning across chess, Go, poker, and Scotland Yard.
  - Its search is growing-tree CFR (GT-CFR), which expands subgames non-uniformly toward relevant states.
  - It reasons over public belief states, i.e. a public state plus ranges over each player's information states.
  - It beats the prior state-of-the-art Scotland Yard agent.
  - Obscuro's one-sided GT-CFR is derived from it.

  — [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10651118); [arXiv 2112.03178](https://ar5iv.labs.arxiv.org/html/2112.03178); [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- The RBC NeurIPS 2022 report sums up the field this way:
  - NFSP, Deep-CFR, DREAM, ESCHER, and R-NaD "have not achieved superhuman play in complex games without search."
  - "It is not clear that these approaches are practical on large games like RBC without extraordinary hardware."
  - Online search "has been shown to greatly improve play strength."

  — [Gardner et al. 2023, PMLR 220](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)

### Inferences
- The only "pure RL" success in this class, DeepNash, relies on (i) a Nash-converging training dynamic and (ii) training compute that I did not verify (see Gaps). The proposed design has neither. Independent advisors trained on shaped rewards and summed at test time optimize a scalarized proxy, not an equilibrium. Nothing in that setup drives the combined policy toward low exploitability.
- A softmax over a weighted sum of Q-values is a Boltzmann (quantal-response-like) policy around a single-agent value estimate. Randomness that comes from a temperature knob is the same thing DeepNash had to remove (low-probability blunders) without the equilibrium structure that makes mixing worthwhile.
- Under a budget of one A6000 for a few weeks, an R-NaD-style FoW agent trained from scratch is very unlikely to approach Obscuro, which needs zero training. A learned value or policy prior used *inside* search (ReBeL / SoG / Obscuro "future work" style) is the better-supported use of RL.

### Gaps
- DeepNash training compute (TPU count, duration) was not in the retrieved main text. It is presumably in the supplement, which I did not retrieve.
- ReBeL and Student of Games compute figures and exact Scotland Yard margins were not retrieved.

## Q4. Reconnaissance Blind Chess (RBC) competition bots

### Takeaway
Across 2019–2022, every RBC winner (StrangeFish, penumbra/Châteaux, Fianchetto, StrangeFish2) used belief tracking of the set of possible boards (or a particle filter) plus engine evaluation (Stockfish or Lc0) over sampled boards, with sensing chosen for its *decision value*. The one RL-centric entrant (uccchess) did not place near the top. Non-transitive results among top bots directly show that non-equilibrium heuristic strategies are exploitable.

### Cited Findings
- Winners by year:
  - NeurIPS 2019: StrangeFish (G. & R. Perrotta) won 464–40, with a winning record against every bot.
  - 2020 JHU APL online competition: penumbra (G. Clark) won.
  - NeurIPS 2021: Fianchetto (Taufeeque, Tongia, Kalyanakrishnan) won 931–89; StrangeFish2 was second at 876–144; penumbra scored 823–197.
  - NeurIPS 2022: StrangeFish2 won and Fianchetto was second.

  — [RBC past competitions](https://rbc.jhuapl.edu/past_competitions); [RBC tournaments](https://rbc.jhuapl.edu/tournaments/33); [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- RBC rules and format, NeurIPS 2022:
  - Before each move a player privately senses a 3×3 region; the opponent is not told where.
  - Time control was 15 min per game + 5 s per turn.
  - Each pair of bots played 60 games.

  This differs from FoW chess, where vision is implicit and there is no sense action. — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- **StrangeFish2 (2022 winner):**
  - Tracks all possible opponent piece locations and estimates position strength and likelihood for each.
  - Chooses both sense and move to maximize the expected outcome of the current turn.
  - Gains came from board-value heuristics layered on Stockfish analysis, and from "prioritized sub-sampling of tracked positions when time limits did not permit evaluation of all hypotheses."

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- StrangeFish (2019) scores moves across the board set with a mix of best-case, worst-case, and average outcomes (Stockfish plus RBC heuristics). Its sensing considers expected board-set reduction, but mainly picks the square with the best expected outcome of the next move choice. — [Stöhr et al. 2024 (SciTePress)](https://www.scitepress.org/Papers/2024/123542/123542.pdf)
- Fianchetto, based on StrangeFish:
  - Uses Lc0 to model a probability distribution over opponent board states (POMDP framing).
  - Puts opponent-specific extra weight on defensive sensing and moves.
  - Uses an opening book that prefers closed positions to keep information sets small.

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- Kevin:
  - Tracks a distribution over full game histories, since opponent senses matter.
  - Predicts opponent senses and moves with a ResNet+LSTM trained by imitation on StrangeFish2 logs.
  - Maximizes expected value with Stockfish.

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- Châteaux / penumbra uses Deep Synoptic Monte Carlo Planning (Clark, NeurIPS 2021):
  - Lossy fixed-size "synopses" of board sets feed a ResNet value/policy network trained on about 600k games.
  - An unweighted particle filter tracks a "second-order" belief of the opponent's information.
  - Planning uses MCTS-style playouts from belief samples.
  - The NeurIPS paper also tests variants with caution, paranoia, and a new bandit algorithm.

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf); [Clark 2021, arXiv 2110.01810](https://www.arxiv.org/abs/2110.01810v1); [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html)
- Oracle: exhaustively tracks board states, senses to spot possible checks or otherwise to minimize the expected number of states, and plays the move Stockfish recommends most often across boards. It was the strongest bot before NeurIPS, 3rd in 2019, and 6th in 2022. — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- Other approaches:
  - ROOKie: boards vote over Stockfish top-5 moves; sensing maximizes expected consensus.
  - Marmot: MC-CFR from mid-game, giving up soundness.
  - JKU-CODA: Siamese-net preference ranking to weight boards, then the best mix of worst-case and weighted-average Stockfish eval.
  - DynamicEntropy: pessimistic single-observer ISMCTS with shallow Stockfish leaves.
  - GarrisonNRL: ISMCTS for large board sets, Maia-1700 voting for small ones.
  - trout baseline: a single-board estimate plus Stockfish.

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- RL entrant: uccchess is StrangeFish with 20 neural networks replacing the engine, plus an RL-trained selector that classifies the opponent. It did not place near the top, and forfeits lowered its rating. — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- Exploitability observed: excluding StrangeFish2, Fianchetto beat Kevin, Kevin beat Châteaux, and Châteaux beat Fianchetto. The organizers call this a cycle that "illustrates the exploitability one can have in a seemingly strong algorithm by an adaptive opponent." — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- Uncertainty metric:
  - Top bots keep the median number of possible boards after sensing under about 35, but below that it barely correlates with rank. StrangeFish2 ranked only 10th on this metric.
  - The organizers conclude that "simply minimizing uncertainty is insufficient. Accounting for the impact of exactly what one will do with an observation appears to be critical."
  - Top bots agree with Stockfish's top-3 move at least about 50% of the time, but this too correlates weakly with rank.

  — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)

### Inferences
- A standalone "scouting" advisor rewarded for information gain resembles the minimize-uncertainty sensing heuristics (Oracle, JKU-CODA). RBC evidence says these are beaten by decision-aware information gathering, i.e. information valued by how it changes the move choice. In FoW, scouting is implicit in move choice, so its value should come out of search over the belief set, not from a separate reward.
- A cheap, proven baseline that fits 2 cores and 4 GiB:
  1. exact or capped board-set filtering;
  2. Stockfish (or a small NNUE) at shallow depth on a time-budgeted sample of boards;
  3. aggregation by expected value mixed with worst-case (king-capture risk);
  4. some randomization among near-equal moves.

  This is roughly the StrangeFish / Oracle recipe adapted to FoW. RBC bots did this at 15 min + 5 s. At 3 min + 2 s, sample sizes must shrink.

### Gaps
- Exact per-bot hardware and per-move time use are not reported in the competition report.
- No head-to-head data exists between RBC bots and FoW bots; the games differ.
- Post-2022 RBC competitions were not checked.

## Q5. Determinization / PIMC failure modes and ISMCTS

### Takeaway
PIMC (sample worlds, solve each as perfect information, average) suffers from strategy fusion and non-locality (Frank & Basin 1998). It works well only in games with certain properties (Long et al. 2010). ISMCTS (Cowling et al. 2012) reduces strategy fusion by searching over information sets. FoW chess has a hidden-information structure where these failures matter, notably the value of hiding and bluffing. Obscuro avoids them through proper imperfect-information subgame solving.

### Cited Findings
- Long, Sturtevant, Buro & Furtak, "Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search," AAAI 2010, pp. 134–140. Using synthetic trees, it identifies game properties that predict whether PIMC is strong or weak relative to optimal play, and shows these properties can be measured in real games. — [mlanthology](https://mlanthology.org/aaai/2010/long2010aaai-understanding); [Sturtevant page](https://www.cs.du.edu/~sturtevant/papers/long2010understanding.html)
- The two errors Frank & Basin identified in PIMC:
  - Strategy fusion: PIMC assumes it can choose a different strategy in each sampled world, although an information set spanning those worlds forces one choice.
  - Non-locality (paraphrased from a search-result snippet): node values in imperfect-information trees create dependencies on distant parts of the tree, which perfect-information solving ignores.

  — [Long et al. 2010 AAAI PDF](https://cdn.aaai.org/ojs/7562/7562-13-11092-1-2-20201228.pdf), as summarized in search results
- ISMCTS (Cowling, Powley & Whitehouse, IEEE TCIAIG 4(2), June 2012): a family of algorithms that "overcome some of the weaknesses" of determinization by searching trees of information sets. Search "is able to exploit moves that are good in many states in the information set and the effects of strategy fusion can be lessened or eliminated." In Phantom (4,4,4), where strategy fusion is especially harmful, ISMCTS beats determinization. — [White Rose eprint](https://eprints.whiterose.ac.uk/75048/1/CowlingPowleyWhitehouse2012.pdf)
- Whitehouse's thesis reports that ISMCTS outperforms PIMC. ISMCTS was deployed commercially in AI Factory's Spades app. — [Whitehouse thesis](https://etheses.whiterose.ac.uk/8117/)
- Practical status in chess-like hidden-information games: single-observer ISMCTS appears in RBC (DynamicEntropy, GarrisonNRL) but not among the top-3 winners. Determinized Stockfish-on-sampled-boards with aggregation is the dominant RBC recipe. — [Gardner et al. 2023](https://proceedings.mlr.press/v220/gardner23a/gardner23a.pdf)
- ZS21's baseline, essentially Stockfish perfect-information evaluation after shallow imperfect-information search, lost 40.5–59.5 to the deeper KLSS agent. That gives a measured cost of leaning on perfect-information reasoning in dark chess. — [ar5iv 2106.06068](https://ar5iv.labs.arxiv.org/html/2106.06068)

### Inferences
- The proposed advisors each produce Q(s, a) from an observation. If trained on full-state rewards (e.g., true material, true king exposure), they inherit PIMC-like optimism: they implicitly assume the opponent's hidden state is known when it is not. Summing such Q-values is a soft form of strategy fusion.
- Determinized evaluation is still a strong *practical* baseline under tight compute, as RBC shows. The fix is to aggregate conservatively (worst-case and king-capture checks), not to drop determinization.

### Gaps
- I could not retrieve the full Long et al. text (the PDF fetch timed out). The names, definitions, and measured effects of their predictive game properties are not verified in these notes.
- Frank & Basin 1998 was not retrieved directly.

## Q6. Course/competition FoW bots, open-source engines, and simple heuristic or small-RL approaches

### Takeaway
I found little evidence of strong heuristic or small-RL FoW chess bots. Open-source FoW code is mostly game UIs. The only serious engine is misty-chess (an Obscuro-style reimplementation, GPL-3.0, Aug–Sep 2026, no strength claim). The best documented "simple" result is Obscuro's ablation: a material-plus-visibility eval inside full KLUSS/CFR search still beat ZS21 (about 2000).

### Cited Findings
- misty-chess: see Q2. — [PyPI misty-chess](https://pypi.org/project/misty-chess/)
- Open-source FoW projects found are games or UIs rather than bots:
  - christian-mann/fog-of-war-chess
  - JackMordaunt/fog-of-chess (Rust prototype using ggez, about 12 stars)
  - AHAPX/dark-chess (Python web game)
  - SlugChess (FoW plus shuffled officers)

  — [christian-mann/fog-of-war-chess](https://github.com/christian-mann/fog-of-war-chess); [fog-of-chess](https://awesome.ecosyste.ms/projects/github.com%2Fjackmordaunt%2Ffog-of-chess); [dark-chess](https://github.imc.re/AHAPX/dark-chess); [SlugChess](https://www.linuxlinks.com/slugchess-chess-fog-war-shuffled-officers/)
- An older chess.com forum thread says no FoW engine existed at the time. — [chess.com forum](https://www.chess.com/forum/view/chess-variants/fog-of-war-chess-engine)
- FoW is playable on pychess.org, and chess.com has a FoW leaderboard (19,150 active players as of April 2025). — [pychess FoW](https://www.pychess.org/variants/fogofwar); [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)
- Related theory: Gehnen & Stannat, "Fog of War Chess," arXiv 2601.18813 (Jan 2026), analyzes endgames:
  - KQ vs K is always a forced win.
  - KR vs K is *not* a guaranteed win, unlike regular chess.
  - KRR vs K is a forced win.

  — [arXiv 2601.18813](https://arxiv.org/abs/2601.18813v1)
- FoW rule details relevant to bots: the king is captured to win and there is no check or checkmate. Castling through check is legal, stalemate is a win for the stalemating side, and there is no draw by insufficient material. Repetition and 50-move draws happen automatically, possibly without either player knowing. — [arXiv 2506.01242v2, App. A](https://arxiv.org/pdf/2506.01242v2)
- A random opponent beats Obscuro only if it fails to cover Qa4+/Qa5+ in the opening, which happens with very small probability. — [arXiv 2506.01242v2](https://arxiv.org/pdf/2506.01242v2)

### Inferences
- **Feasibility ranking under our limits.** Runtime: 2 CPU cores, 4 GiB, 180 + 2 s, so about 5 s per move over Obscuro's average of about 58 moves. That is close to the 5 s/move setting of Obscuro's AI-vs-AI tests. Training: one A6000 and 32 cores for weeks.
  1. **Best expected strength:** an Obscuro/misty-style search, i.e. exact belief set + KLUSS + PCFR+ / one-sided GT-CFR + Stockfish depth-1 MultiPV leaves + safety-gated purification, in Rust or C++. No training is needed. Compute scaling suggests a 2-core version stays strong (an extrapolation). Risk: implementation complexity and memory spikes when |P| reaches about 10^6.
  2. **Simplest strong baseline:** StrangeFish/Oracle-style board-set filtering, shallow Stockfish on a time-budgeted sample, expectation plus worst-case aggregation, and explicit king-capture safety checks. Add randomization among near-equal moves, especially in the opening.
  3. **Where RL fits the budget:** train a small value or policy network (e.g., NNUE-like) on self-play or Obscuro-style games. Use it as the leaf evaluator or opponent model *inside* (1) or (2), as Obscuro's own future-work section suggests. Hand-designed terms such as material, king safety, and visibility are acceptable as features or eval components; simple-eval Obscuro proves they can carry real strength inside search.
  4. **Proposed design as stated:** separate shaped-reward RL advisors with weighted-sum softmax and no belief tracking or search. It is unlikely to be competitive with (1) or (2). It lacks belief modeling, equilibrium reasoning, and safety-gated mixing, which every strong system cited here relies on. It also risks reward-hacking or misweighting among advisors and temperature-induced blunders (the DeepNash "low-probability mistakes" issue). Its main value would be as a research ablation, e.g. "how far can shaped RL get without search?", not as the strongest entry.
- If the advisor design is kept, the minimal fixes are:
  - feed advisors belief features (or the board set), not a single observation;
  - add a hard king-capture safety filter before sampling;
  - replace the global temperature with top-k mixing gated on value margin, as in Obscuro's purification;
  - consider an R-NaD-style regularized self-play objective for the "win" advisor rather than pure Q-learning.

### Gaps
- I found no concrete course-competition FoW chess bots or published results for small-RL or heuristic-only FoW agents. pychess and corCTF "smogofwar" turned up only as mentions.
- I found no published Elo for any open-source FoW engine.
