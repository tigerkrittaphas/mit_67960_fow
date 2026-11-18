# Alignment Theory and Ensemble Combination for a Fog-of-War Chess "Council"

Proposed design under assessment: several advisors, each scoring candidate moves under its own objective. A principal plays the move sampled from softmax(λ · Σ_i w_i · score_i), with w ≥ 0 fit by NNLS or by CMA-ES. The design is motivated by Collina, Goel, Roth, Ryu and Shi, "Emergent Alignment via Competition" (arXiv:2509.15090), and is applied to Fog-of-War (FoW) chess, a two-player, zero-sum, imperfect-information game.

Theorem numbering below follows arXiv v2 (2 Feb 2026), which matches the ICML 2026 / PMLR 306 version. v1 (18 Sep 2025) numbering may differ.

## Q1. Collina et al.: definitions, theorems, assumptions, experiments, follow-ups

### Takeaway
In Collina et al., the weighted sum is an assumption about utilities, not a decision rule. If the user's utility lies (approximately) inside the convex hull of the AIs' utilities, then strategic AIs competing in a Stackelberg/persuasion equilibrium give the user near-first-best utility. The council inverts this: its advisors are non-strategic, and the principal computes the weighted sum itself. No theorem in the paper covers that setup. What carries over directly is the Section 6 experiment, which is in effect constrained least-squares stacking (NNLS or simplex weights) fit against true labels. It beat the best single model by a large margin, and a plain average did badly.

### Cited Findings
- **Bibliographic.** Authors: Natalie Collina, Surbhi Goel, Aaron Roth, Emily Ryu, Mirah Shi. v1 18 Sep 2025; v2 2 Feb 2026. Published at ICML 2026, PMLR 306:21257–21292. — [arXiv abs](https://arxiv.org/abs/2509.15090); [PMLR](https://proceedings.mlr.press/v306/collina26a.html); [ICML poster](https://icml.cc/virtual/2026/poster/62259)
- **Model.**
  - A user ("Alice") picks an action a ∈ 𝒜 under an unknown state y ∈ 𝒴. There are k AI agents ("Bobs"), each with its own utility U_i. All utilities map into [0,1].
  - The setting is a multi-leader Stackelberg game: the Bobs (AI providers) commit to conversation rules first, and Alice best-responds. The paper describes it as extending Bayesian persuasion to multi-round conversation.
  - There is a commonly known prior P(x_A, x_B, y). Alice observes x_A, the Bobs observe x_B, and y is unobserved.
  - Utilities are not assumed linear; linearity enters only through the alignment condition.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Definition 1, (approximate / ε-) market alignment.** There exist weights w_i ≥ 0 with Σ_i w_i = 1, an offset c, and ε ≥ 0 such that sup_{a,y} |Σ_i w_i U_i(a,y) + c − u_A(a,y)| ≤ ε.
  - Remark 2: Σ w_i ≤ 1 can be obtained by adding a dummy Bob with zero utility.
  - The condition is a uniform (sup-norm) bound over all action–state pairs.
  - The authors say it "becomes a weaker assumption the more differently aligned models there are."
  - Their motivating example is drug manufacturers whose misaligned commercial interests are mutually zero-sum.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Theorem 1 (best-responding Alice).**
  - Assumptions: the game satisfies the "identical induced distribution" condition, and the Bobs are ε-market-aligned.
  - Conclusion: in any Nash equilibrium of the Bobs' game, Alice's expected utility is ≥ U_A(C_B*) − 2ε, where C_B* is the optimal conversation rule of a single perfectly aligned Bob.
  - Proposition 1 gives a sufficient condition: the Alice-optimal leader strategy lets Alice learn her Bayes-optimal action, which requires a rich enough message space.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Section 4: bounded-rational Alice using quantal response with parameter λ.**
  - Lemma 1: the quantal-response suboptimality gap is ≤ log|𝒜|/λ.
  - Lemma 2: softmax is multiplicatively stable under ‖u − u′‖∞ ≤ ε.
  - Theorem 2: under ε-market alignment and a (δ, C_B*)-closeness condition, Alice's utility in any Quantal Response Nash Equilibrium is ≥ U_A(C_B*) − 2ε − δ.
  - Proposition 2: δ ≤ e^{4λε_u} − 1, where ε_u is the uniform error of her utility estimates.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Theorem 5 (main result, information substitutes).**
  - Assumptions:
    - The prior satisfies the Information Substitutes Condition (Definition 14, taken from Frongillo et al. 2021).
    - The Bobs have average market-alignment error ε.
    - Alice commits to the "straightforward" conversation rule (she truthfully reports her posterior expected utilities) and to a λ-quantal-response decision rule.
    - The solution concept is Quantal Response Nash Equilibrium.
  - Bound: E[u_A] ≥ OPT − 2ε − (2(10ζ^{1/3} + δ_conv) + e^{4λ·10ζ^{1/3}} − 1 + δ_conv) − log|𝒜|/λ, with ζ = (3|𝒜|/(K·δ_conv))^{1/2} for K conversation rounds.
  - Corollary 3 (when λ·10ζ^{1/3} ≤ 1/4): E[u_A] ≥ OPT − 2ε − (20 + 40eλ)ζ^{1/3} − 3δ_conv − log|𝒜|/λ.
  - Two building blocks:
    - Theorem 3 (from Aaronson 2005 / Collina et al. 2025b): ζ-agreement holds with probability ≥ 1 − δ_conv after K ≥ 3|𝒜|/(ζ²δ_conv) rounds.
    - Theorem 4 (from Frongillo et al. 2021): under information substitutes, ζ-agreement implies an estimation error of at most 10ζ^{1/3}.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Theorem 6 (Best-AI Selection game).**
  - Assumption: the k Bobs satisfy ε-market alignment. No distributional assumptions are needed (no information substitutes).
  - Conclusion: in any Nash equilibrium of the Best-AI Selection game (Definition 17), Alice's expected utility is ≥ U_A(C_B*) − 2ε. In this game, Alice picks one Bob after an evaluation period.
  - Remark 5: the upper-bound side of the alignment condition is needed only on outcomes realized in equilibrium, and the lower-bound side only on outcomes under C_B*.
  - Appendix A: under a random-utility model, k > (ln(2|𝒜||𝒴|) − ln δ)/(2ε²) Bobs suffice for the condition to hold.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Experiments, Section 6 (whether the convex-hull condition holds empirically).**
  - Weights minimize ‖Uw − y‖²₂ on training folds. NNLS constrains w ≥ 0; the "simplex" variant adds 1ᵀw = 1.
  - Evaluation: 5-fold cross-validation, averaged over 100 random permutations of the Bobs. Metric: MSE. A weight counts as nonzero if it exceeds 1e-6.
  - Data: 100 LLM "Bobs" per domain, made by prompt variations. Domains: ETHICS (1,000 scenarios, with an LLM-with-handcrafted-prompt as the "human") and MovieLens (average human ratings, movies with ≥ 20 ratings).
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Results, Section 6.**
  - NNLS cut MSE relative to the best individual Bob by about 52% (ETHICS) and about 75% (MovieLens). Simplex: about 52% and about 71%.
  - "The simple average performs poorly (even worse than the best individual in MovieLens)."
  - NNLS was sparse, with about 18 (ETHICS) and about 26 (MovieLens) nonzero weights out of 100. It was sparser than simplex at similar error.
  - Error fell with diminishing returns as K grew from 1 to 100.
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **OpinionQA (Section 6.2).** Pew American Trends Panel, with 9 LLMs. NNLS "consistently outperforms the best individual model" and typically matches or beats simplex and simple averaging. — [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Equilibrium experiments (Section 7).**
  - Setup: one-shot persuasion in which Alice has no signal; solved by best-response dynamics.
  - Alice's equilibrium utility was at or above the theoretical bound, and lower ε predicted higher utility. Going from 1 to 2 Bobs gave a large gain.
  - The abstract notes that competition improves user utility reliably when the convex-hull assumption holds, "but does not always when it fails."
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2); [search summary of abstract](https://arxiv.org/abs/2509.15090v2)
- **Authors' stated limitations.**
  - "our paper assumes that the AI models are acting in equilibrium of a highly complex game"
  - Alice is assumed to use the models optimally and form correct posteriors.
  - The model assumes commitment rather than cheap talk, and a single user utility.
  - "Our experimental evaluations in this paper are also very simple and stylized."
  - [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)
- **Follow-up: "Personalization Aids Pluralistic Alignment Under Competition"** (Collina, Goel, Roth, Shi; arXiv 2602.13451, Feb 2026).
  - It extends the model to many users.
  - Under a *Weak Market Alignment* condition, and with per-user personalization, every equilibrium gives each user outcomes comparable to a perfectly aligned model.
  - With a single anonymous policy, some equilibria are uninformative; a stronger condition restores the guarantee.
  - Per a search snippet, weak market alignment requires provider utilities to be approximately additively separable across users.
  - [arXiv abs](https://arxiv.org/abs/2602.13451)

### Inferences
- **Roles are inverted.** In the paper, the advisors (Bobs) are *strategic* and the weighted sum is a *hypothesis* about their utilities. Alignment *emerges* because Bobs compete for Alice's attention. In the council:
  - the advisors are fixed scoring functions with no incentive to compete;
  - the principal itself implements Σ w_i s_i.

  So Theorems 1, 2, 5 and 6 give no guarantee for the council. The only part that transfers literally is Section 6: fitting nonnegative weights so that a combination of proxies approximates a target utility. That is classical stacked regression, not an equilibrium result.
- **No adversary.** Alice faces a state of nature y drawn from a common prior. In FoW chess, the "state" includes an opponent who adapts to the principal's policy, so the common-prior, nature-picks-y framing does not apply. Zero-sum play falls outside the framework.
- **What ε means for the council.** ε is a sup-norm bound over every (a, y). For the council, the analog is that the true objective (win probability or game value) is uniformly within ε of an affine combination of advisor scores. Win probability is roughly logistic in a linear eval (see Q3), so the fit should be done in logit/eval space, not probability space, for the affine condition to be plausible.
- **Label problem.** Collina et al. regress on *observed true utility* (ETHICS labels, MovieLens ratings).
  - NNLS for the council needs a target: game outcome, a Texel-style label, or a stronger engine's value.
  - CMA-ES on match results optimizes win rate directly, but its relation to "market alignment" is then only heuristic.
  - The write-up should say which objective the weights are fit to.
- **The softmax cost is quantified.** The −log|𝒜|/λ term in Lemma 1 and Theorem 5 is the paper's own price of acting by quantal response instead of argmax. In chess, |𝒜| ≈ 30–40, so log|𝒜| ≈ 3.5. A small λ (relative to the score scale) costs utility directly.
- **Theorem 6 is the cleanest result, but it selects rather than combines.** It needs only ε-alignment plus equilibrium play. Its mechanism is "pick the best single advisor after evaluation," which argues for including a best-single-advisor baseline in the council.
- **Averaging did poorly, and sparsity emerged.** These findings support learned NNLS weights over uniform weights. They also suggest that many advisors may be zeroed out, which works as an implicit advisor-selection diagnostic.

### Gaps
- I did not read Definition 14 (information substitutes) verbatim. It was characterized only as an MSE-reduction comparison from Frongillo et al. 2021.
- The exact definition of "identical induced distribution" was not extracted.
- I found no published critiques or rebuttals of Collina et al. as of Nov 2026, only the authors' own follow-up and AI-generated summary pages.
- The precise wording of Weak Market Alignment (Definition 2.13 in 2602.13451) was seen only via a search snippet.

## Q2. Related theory: experts/Hedge, ensemble RL, product vs mixture of experts, BMA, options

### Takeaway
softmax(λ Σ w_i s_i) is mathematically a *product of experts*: it equals Π_i softmax(λ w_i s_i) up to normalization. It is not a mixture. With equal weights it matches the usual description of Wiering and van Hasselt's "Boltzmann multiplication" (formula unverified; see Gaps). The closest RL precedent for learned weights is Marivate and Littman (2013), who fit a linear combination of base agents' Q-values with TD. Ensemble RL work reports that voting or multiplicative combination beats single agents in small domains. There are no game-theoretic guarantees for adversarial settings.

### Cited Findings
- **Wiering and van Hasselt (2008), "Ensemble Algorithms in Reinforcement Learning,"** IEEE Trans. SMC-B 38(4):930–936.
  - Combination rules: majority voting, rank voting, Boltzmann multiplication (BM) and Boltzmann addition, each over policies derived from the base agents' value functions.
  - Base agents: Q-learning, Sarsa, Actor-Critic, QV-learning, ACLA.
  - Results: the BM and majority-voting ensembles beat the single algorithms on five maze tasks, four of them dynamic or partially observable.
  - [readings page](https://www.cs.utexas.edu/~shivaram/readings/b2hd-WieringVanHasselt2008.html) (abstract-level via search; full text not retrieved)
- **Marivate and Littman, "An Ensemble of Linearly Combined Reinforcement-Learning Agents"** (AAAI 2013 late-breaking).
  - "learns a weighted linear combination of Q-values from multiple independent learning algorithms," using a linear TD meta-learner.
  - The method "compares favorably to the best tuned algorithm."
  - They contrast it with Wiering and van Hasselt's fixed combination rules: the fusion is "adjusted given the performance of the base agents."
  - [PDF](https://cdn.aaai.org/ocs/ws/ws0960/7025-30741-1-PB.pdf)
- **Hans and Udluft (ICML 2010)** reportedly found majority voting superior to Q-averaging in neural-network ensembles. This is secondhand, from a search summary; not verified. — [PDF](https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Publikationen/conferences_int/2010/Hans-ICML-2010.pdf)
- **Product vs mixture (Hinton 2002, Neural Computation 14:1771–1800).**
  - A mixture of experts is inefficient in high dimensions because the mixture cannot be sharper than its components.
  - A product of experts can be much sharper; every expert's constraints must be roughly satisfied.
  - Training is hard because of the normalizer, hence contrastive divergence.
  - [Hinton PDF](https://www.cs.utoronto.ca/~hinton/absps/nccd.pdf); [mlanthology](https://mlanthology.org/neco/2002/hinton2002neco-training)

### Inferences
- **Algebraic identity:** softmax(λ Σ_i w_i s_i)(a) ∝ Π_i exp(λ w_i s_i(a)) ∝ Π_i softmax(λ w_i s_i)(a). Consequences:
  - The council is a weighted product of experts (a log-linear pool), and w_i act as per-advisor inverse temperatures.
  - Any one advisor can effectively veto a move by giving it a very low score. That is good for blunder-avoidance advisors and bad if an advisor is miscalibrated.
  - A mixture (Σ w_i π_i) would instead let any single advisor make a move likely.
- **Online weighting.** Hedge-style weights are a natural online alternative to batch NNLS or CMA-ES, for example when adapting weights across opponents. Two caveats:
  - Hedge mixes over experts' *recommendations*, not over their score scales.
  - Its regret is measured against the best *single* expert, not the best combination. That matches Theorem 6's "select one" rather than Section 6's "combine."
- **Learned weighting is supported over fixed rules,** by both Marivate and Littman and Collina et al. §6. Note that Marivate and Littman fit the weights by TD toward the environment's return. That is the RL analog of using a true-utility label, and arguably a better fit for the council than NNLS on hand-made labels.
- **Arbitration alternatives.** State-dependent gating (mixture of experts, or an options/hierarchical controller choosing which advisor acts) would let the weights vary with game phase. The engine-tuning evidence in Q3 (separate middlegame and endgame terms) suggests that phase-dependent weights matter.

### Gaps
- (Background, not re-fetched) **Hedge / multiplicative weights (Freund and Schapire 1997, JCSS 55(1)).** This gives the decision-theoretic online-learning framework, where weights over experts are updated multiplicatively by observed loss with regret guarantees against the best single expert in hindsight. Cited from standard knowledge; the paper was not re-fetched in this session. — [DOI](https://doi.org/10.1006/jcss.1997.1504)
- I could not retrieve the exact Boltzmann multiplication formula from Wiering and van Hasselt (the fetch failed). The identification of the council with BM rests on the method name and the standard description, so it should be verified against the paper.
- **Bayesian model averaging** (Hoeting et al. 1999, Statistical Science) and **options** (Sutton, Precup, Singh 1999, AIJ) were not fetched this session. The standard point is that BMA averages *predictive distributions* weighted by posterior model probability, which converges toward a single model as data grow. It is not a learned combination. Treat this as uncited background.
- I found no ensemble-RL paper studying weighted-sum policy combination in adversarial imperfect-information games.

## Q3. Linear evaluation functions in games: how well linear feature combinations track win probability

### Takeaway
A linear eval passed through a logistic link has long been the workhorse of chess engines. Texel tuning, which is literally logistic regression of game result on a linear eval, gave about 100 Elo of gains. KnightCap learned its evaluation function online with TDLeaf(λ). TD-Gammon found that a linear-in-raw-inputs eval is already far better than random. But the strongest evaluations are nonlinear: TD-Gammon's multilayer net, and NNUE, which replaced Stockfish's handcrafted linear eval entirely. A linear council over a few advisors is therefore a reasonable, tunable baseline, but its ceiling is limited.

### Cited Findings
- **Texel's Tuning Method (Peter Österlund).**
  - It is "a concrete instance of a logistic regression."
  - It minimizes the mean squared error between the game result R_i ∈ {0, ½, 1} and a sigmoid of the quiescence-search score q_i scaled by K.
  - K is chosen once to minimize error; the wiki reports K = 1.13 for that engine.
  - Optimization is ±1 local search over integer parameters; Gauss-Newton and conjugate gradient with difference quotients are listed as alternatives.
  - Data: about 64k fast games, giving about 8.8M positions.
  - Cumulative gain was +99.6 Elo, including +39.4 from changing which positions were included and +10.2 from separate middlegame and endgame queen piece-square tables.
  - [chessprogramming wiki](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- **Original forum framing.** Win probability = sigmoid(C·eval); "if your evaluation function is a linear combination of terms … step (4) is logistic regression." There is disagreement on the scale constant:
  - about 0.58 (the original author, from memory);
  - 0.75 (another user);
  - 1.13 (the wiki).
  This is unresolved and engine-specific. — [talkchess thread](https://talkchess.com/viewtopic.php?p=561474); [wiki](https://www.chessprogramming.org/Texel%27s_Tuning_Method)
- **Phase dependence.** Forum users report that the pawn value varies between middlegame and endgame (roughly +40 vs +150 in one report), so a single fixed linear scale distorts the eval-to-win mapping across phases. — [talkchess thread](https://talkchess.com/viewtopic.php?p=561474)
- **KnightCap (Baxter, Tridgell, Weaver 1999).** TDLeaf(λ), "a variation on the TD(lambda) algorithm that enables it to be used in conjunction with game-tree search," raised its FICS rating from 1650 to 2150 in 308 games over 3 days. The authors credit online play against humans, rather than self-play, as a key reason. — [arXiv cs/9901002](https://arxiv.org/abs/cs/9901002)
- **TD-Gammon (Tesauro 1995, CACM 38(3)).**
  - It is a multilayer network. The early learned strategy was "an evaluation function that is linear in the raw input variables," "enormously better than the random initial strategy" and "better than typical human beginner-level play."
  - A raw-input network (40 hidden units, 200k games) reached about Neurogammon level. Adding hand-designed features "greatly surpassed" all previous programs.
  - Tesauro credits the dice for exploration: "the learner explores more of the state space than it would in the absence of such a stochastic noise source."
  - [bkgm.com article](https://bkgm.com/articles/tesauro/tdl.html)
- **NNUE.**
  - Introduced in 2018 by Yu Nasu for shogi. Its first layer (the accumulator) is incrementally updatable.
  - It was responsible for "the huge strength improvement of Stockfish 12," despite roughly halving search speed.
  - [chessprogramming wiki NNUE](https://www.chessprogramming.org/NNUE)
- **Removal of the handcrafted eval.** Stockfish's handcrafted (classical) evaluation was removed: the official 16.1 announcement (Feb 2024) places the removal in 16.1. The chessprogramming wiki attributes it to Stockfish 16 (June 2023). These sources conflict. — [Stockfish 16.1 blog](https://stockfishchess.org/blog/2024/stockfish-16-1/); [cpw Stockfish](https://www.chessprogramming.org/stockfish)

### Inferences
- **NNLS on outcomes is Texel tuning.** If advisor scores are treated as features, NNLS fit to game outcomes is Texel-style tuning with a nonnegativity constraint and an identity link. A logistic link, sigmoid(Σ w_i s_i) fit to results, is the better-founded choice. Its fitted scale also tells you the "natural" λ: Texel's K is exactly a softmax-temperature analog.
- **Phase-dependent weights.** Texel gains from phase-split terms, and the reports of phase-dependent piece values, suggest the council should allow w(phase), or a gating network, rather than one global w.
- **Limited ceiling.** NNUE's displacement of handcrafted linear eval, and TD-Gammon's need for hidden units, show that a fixed linear combination of a few advisors has a limited ceiling. Stacking with a small nonlinear meta-model may beat NNLS once the training data are large.
- **FoW versus perfect information.** Evidence in this section comes from perfect-information chess and backgammon. In FoW chess, advisor scores are computed on a belief or sampled state, so label noise is larger. KnightCap's point about online play against diverse opponents may matter here too.

### Gaps
- I found no quantitative study comparing linear stacking with nonlinear stacking of *separately trained objective-specific advisors* in chess.
- I found no FoW-specific evaluation-tuning study.
- I did not verify an Elo figure for the NNUE jump.

## Q4. Mixed strategies and softmax: QRE, entropy-regularized play, and exploitability

### Takeaway
Sampling from softmax over a *fixed* evaluation is not a quantal response equilibrium and not a Nash equilibrium. A QRE is a mutual fixed point in which each player's logit response is to the *other's* QRE strategy. Theory and practice in imperfect-information games show that equilibrium-seeking dynamics (MMD, R-NaD, NFSP, and search methods like Obscuro's) are needed for low exploitability, while plain RL self-play can diverge. A fixed softmax-over-score bot randomizes, but it does so in proportion to score gaps, not in the proportions that make it unexploitable. A best-responding opponent can exploit it.

### Cited Findings
- **QRE (McKelvey and Palfrey 1995, GEB 10(1):6–38).**
  - Players make noisy, probabilistic best responses "based on relative expected utility," assuming others do the same. A QRE is a *fixed point* of this process.
  - For logit QRE, as noise → 0 the equilibria approach a subset of Nash equilibria, and in generic games they select a unique one.
  - The extensive-form version is McKelvey and Palfrey 1998.
  - [Caltech resolver](https://resolver.caltech.edu/CaltechAUTHORS:20171128-164515991); [RePEc](https://econpapers.repec.org/RePEc:clt:sswopa:883)
- **Magnetic Mirror Descent (Sokota et al.; arXiv 2206.05825, v4 April 2023).**
  - "First quantal response equilibria solver to achieve linear convergence for extensive-form games with first-order feedback."
  - First standard RL algorithm empirically competitive with CFR in tabular settings.
  - Favorable as deep self-play RL in 3x3 Dark Hex and Phantom Tic-Tac-Toe, which are imperfect-information games close in spirit to FoW chess.
  - [arXiv](https://arxiv.org/abs/2206.05825)
- **DeepNash / R-NaD (Perolat et al. 2022, Stratego).** Regularised Nash Dynamics converges to an "approximate Nash equilibrium, instead of 'cycling' around it" by modifying the learning dynamics. It is model-free with no search, and reached top-3 on Gravon. — [arXiv 2206.15378](https://arxiv.org/abs/2206.15378)
- **NFSP (Heinrich and Silver 2016).** In Leduc poker NFSP approached a Nash equilibrium, "whereas common reinforcement learning methods diverged." — [arXiv 1603.01121](https://arxiv.org/abs/1603.01121)
- **Obscuro (Zhang and Sandholm; arXiv 2506.01242, v2 Feb 2026).** The first superhuman FoW chess AI. It uses new imperfect-information search techniques "without common knowledge," and beat the prior state of the art and top humans. FoW chess is "the largest turn-based zero-sum game" (by amount of imperfect information) with superhuman play. Only the abstract was read. — [arXiv](https://arxiv.org/abs/2506.01242)
- **λ cost from Collina et al.** Even in a cooperative, non-adversarial setting, a λ-quantal response loses up to log|𝒜|/λ relative to best response (Lemma 1). — [arXiv v2 HTML](https://arxiv.org/html/2509.15090v2)

### Inferences
- **Not a QRE.** A logit response to a static Σ w_i s_i is not a QRE. The scores are not expected utilities against the opponent's actual (equilibrium) strategy, and nothing enforces the mutual fixed point. The resulting mixed strategy is an artifact of score gaps and λ.
- **How the fixed softmax bot is exploitable in FoW chess:**
  - (a) Its action frequencies are predictable functions of public observations, so an opponent can model and best-respond to them.
  - (b) It does not randomize to protect private information: equal-score moves get equal probability whatever they reveal.
  - (c) A large λ collapses to near-deterministic play, which loses the hiding value of mixing. A small λ pays log|𝒜|/λ-type losses on clear-cut positions.
- **λ does not provide robustness.** Tuning λ or w by CMA-ES against a fixed opponent pool fits weights to *that pool*. This is the "cycling / best-response-to-population" problem that R-NaD, MMD and NFSP are designed to avoid. Report exploitability, or at least performance against an adaptive opponent, not only win rate against fixed opponents.
- **Bridge to MMD.** Treat the council's Σ w_i s_i as a "magnet," or reference policy, inside an MMD/R-NaD-style update. That keeps the advisors' prior knowledge and the softmax form while letting the dynamics move the policy toward a QRE/Nash of the actual game. MMD's objective is described, per the method name, as reward plus entropy (or KL-to-magnet) regularization; this is unverified beyond the abstract.
- **Benchmarks.** Obscuro sets the strength benchmark for FoW chess. A council bot without equilibrium-aware search should be expected to be far weaker against strong opponents.

### Gaps
- I did not extract the text from the MMD and Obscuro papers on how FoW-style information hiding requires mixing. Only abstracts were read.
- I found no published exploitability measurement of a "softmax over a handcrafted eval" bot in FoW chess or a similar game.

## Q5. Reward shaping pitfalls, proxy rewards and Goodhart in game RL

### Takeaway
Potential-based shaping, F = γΦ(s′) − Φ(s), is the *only* additive shaping that guarantees optimal-policy invariance in MDPs (Ng et al. 1999). In stochastic games it preserves the set of Nash equilibria (Devlin and Kudenko 2011). A material advisor *inside the action-selection sum* is not shaping, however. It changes the objective being optimized, so none of these guarantees apply. Skalse et al. show that over stochastic policies, two nontrivial rewards essentially cannot be "unhackable" relative to each other. A dense material proxy should therefore be expected to be gameable.

### Cited Findings
- **Ng, Harada and Russell (ICML 1999), Theorem 1.**
  - F is potential-based if F(s,a,s′) = γΦ(s′) − Φ(s) for a real-valued Φ.
  - Sufficiency: "every optimal policy in M′ will also be an optimal policy in M (and vice versa)."
  - Necessity: if F is not potential-based, "there exist (proper) transition functions T and a reward function R … such that no optimal policy in M′ is optimal in M."
  - In the infinite-state case, Φ should be bounded.
  - [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Shaping bugs reported by Ng et al.**
  - A bicycle agent rewarded for progress toward its goal "learned to ride in tiny circles near the start state."
  - A soccer robot rewarded for touching the ball learned to stay next to it and "vibrate," touching it as often as possible.
  - [PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- **Devlin and Kudenko (AAMAS 2011).**
  - In multi-agent stochastic games, potential-based shaping keeps the equivalence to Q-table initialization, and "the Nash Equilibria of the underlying stochastic game are not modified."
  - Empirically, however, shaping can change *which* joint policy learning converges to.
  - A 2012 follow-up extends the guarantees to dynamic (time-varying) potentials.
  - [White Rose eprint](https://eprints.whiterose.ac.uk/id/eprint/75111/)
- **Skalse et al. (NeurIPS 2022, "Defining and Characterizing Reward Hacking/Gaming").**
  - A proxy is *unhackable* if increasing its expected return can never decrease expected true return.
  - "Linearity of reward (in state-action visit counts)" makes unhackability very strong. Over all stochastic policies, two reward functions are unhackable relative to each other only if one is constant.
  - Simplifying a reward (dropping terms) usually does not give an unhackable proxy.
  - [arXiv 2209.13085](https://arxiv.org/abs/2209.13085v1); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3d719fee332caa23d5038b8a90e81796-Abstract-Conference.html)

### Inferences
- **Two uses of a material advisor, with different guarantees:**
  - (i) As a *potential* Φ(s) = material(s), added as γΦ(s′) − Φ(s) to the terminal win/loss reward during RL training. This is safe in the Ng and Devlin–Kudenko sense: it changes learning speed and exploration, not the optimal or equilibrium policies.
  - (ii) As a *term in Σ w_i s_i* at decision time, scoring moves by immediate material gain. This alters the objective being maximized. That is exactly the non-potential case, so material-grabbing (pawn snatching that exposes the king, especially when the king is invisible in FoW) can be optimal for the council but suboptimal for winning.
- **Plausible failure mode in FoW chess.** In FoW chess the game is won by capturing the king, and there is no check. Under that rule set, a material advisor may undervalue king safety under uncertainty, and an opponent can bait it with material sacrifices. This is an inference and was not tested in any source.
- **Proxy limits apply directly.** Each advisor is a proxy, and a weighted sum of proxies is still a proxy. Skalse et al.'s result implies that unless the weights are fit to the *true* objective (game outcome), optimizing Σ w_i s_i can lower win rate. This links back to Collina et al.: their guarantee depends on ε being small *uniformly over all actions and states*, including those the optimizer is pushed toward.
- **Practical safeguard.** Fit w on outcome labels (or by CMA-ES on win rate), and audit for Goodhart failures. One audit is to check whether the high-w advisors' scores stay correlated with outcome on positions the council itself steers into, not only on positions from a reference dataset.

### Gaps
- I found no primary-source study of reward hacking specifically from material-based dense rewards in chess RL. The material-grabbing failure is an inference, not a cited result.
- I did not retrieve POMDP-specific results on potential-based shaping (e.g., belief-state potentials). Whether Ng's invariance carries over to belief-space potentials in FoW chess needs checking.
