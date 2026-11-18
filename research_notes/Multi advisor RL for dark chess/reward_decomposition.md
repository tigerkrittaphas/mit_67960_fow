# Reward decomposition and multi-advisor RL: what the literature means for the per-advisor-Q weighted-softmax principal in Fog-of-War chess

Proposed design under review: separate deep-RL advisors (material, king safety, scouting/information gain, pressure on the enemy king, terminal win), each trained on its own reward. Their per-move Q-values are combined by a weighted-sum softmax principal, with weights fit by NNLS or CMA-ES.

Primary sources read in full text for this note: Russell & Zimdars (ICML 2003), Laroche et al. "Multi-Advisor RL" (arXiv 1704.00756 v2, 2017; the PDF header reads "Under review as a conference paper at ICLR 2018", so treat it as a preprint, not a peer-reviewed publication), van Seijen et al. HRA (arXiv 1706.04208, NeurIPS 2017), Juozapaitis et al. drQ (2019), Barreto et al. Successor Features (arXiv 1606.05312, NeurIPS 2017), Romoff et al. TD(Δ) (arXiv 1902.01883, ICML 2019), Yang et al. Envelope Q-learning (arXiv 1908.08342, NeurIPS 2019), and Abels et al. Dynamic Weights (arXiv 1809.07803, ICML 2019). For Fatemi & Tavakoli OVM (ICLR 2022) only the abstract was read.

---

## Q1. Russell & Zimdars 2003: Q-decomposition, local Sarsa vs. local Q-learning, and the "illusion of control"

### Takeaway
Summing component Q-functions gives the globally optimal policy only if every component evaluates the same policy, namely the arbitrator's own policy. Local Sarsa does this by bootstrapping on the action the arbitrator actually took. Independent local Q-learning (each subagent bootstrapping on its own greedy max) converges to "selfish," optimistic estimates. Their sum can choose provably bad actions, for example NoOp forever.

### Cited Findings
- Architecture: "Each subagent has its own reward function and runs its own reinforcement learning process. It supplies to a central arbitrator the Q-values (according to its own reward function) for each possible action. The arbitrator selects an action maximizing the sum of Q-values from all the subagents." The arbitrator "receives no reward signals and maintains no Q-functions." — [Russell & Zimdars, ICML 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Core assumption: the global reward decomposes additively, r(s,a,s') = Σ_j r_j(s,a,s'). Then for any fixed policy π, Q^π(s,a) = Σ_j Q^π_j(s,a). The identity holds only because every Q^π_j is defined with respect to the same π. — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Q-decomposition is preferred over "command arbitration" (subagents vote for actions) and "command fusion" (averaging recommended actions). The authors give a chess example: fusion "is often inapplicable—as, for example, when two chess-playing subagents recommend a knight move and a bishop move respectively." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Local Q-learning update (eq. 2): Q_j(s_t,a_t) ← (1−α)Q_j + α[r_j + γ max_a Q_j(s_{t+1}, a)]. Theorem 1 (citing Tsitsiklis 1994): each Q_j converges a.s. to the selfish fixed point Q̃_j(s,a) = Σ_{s'} P(s'|s,a)[R_j + γ max_{a'} Q̃_j(s',a')]. — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- The "illusion of control": "each subagent learns the Q_j values that would result if that subagent were to make all future decisions for the agent… the subagents converge to 'selfish' estimates that overestimate the true values of their own rewards with respect to a globally optimal policy." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Dollar/euro counterexample: from S0 the agent can go Left ($1), Right (€1), Up ($0.60 + €0.60), or take NoOp; Up is optimal. With selfish Q's, Q̃(s0,NoOp) = 2(1−ε)γ/(1−εγ) while Q̃(s0,Left) = 1/(1−εγ). "When the discount factor is sufficiently large (γ > 0.5), the sum of selfish action-value estimates indicate that the agent is better off doing nothing than settling on one of the absorbing states." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- The sum of selfish components is an upper bound: Q̃(s,a) = Σ_j Q̃_j(s,a) ≥ Q*(s,a). "The sum of selfish components is therefore an optimistic estimate of the optimal value over all components." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Local Sarsa (eq. 5): Q_j(s_t,a_t) ← (1−α)Q_j + α[r_j + γ Q_j(s_{t+1}, a_{t+1})], where a_{t+1} is the action the arbitrator executed. Theorem 2: if all (s,a) are visited infinitely often, the arbitrator's policy is GLIE (greedy in the limit with infinite exploration), and S and A are finite, then each Q_j converges a.s. to Q*_j(s,a) = E[r_j + γ Q*_j(s', π*(s'))]. Proof idea: "the local Sarsa update is just the monolithic Sarsa update in algebraic disguise." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Global optimality "is achieved if each subagent's Q_j-function correctly reflects its own future r_j rewards assuming future decisions are made according to the global arbitration policy." This "requires that the arbitrator inform each subagent of the successor action it actually followed." — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Experiments:
  - Racetrack (progress subagent vs. crash-avoidance subagent): local Sarsa beat local Q by about 1 standard deviation, and local Sarsa matched global Sarsa.
  - Fishery with 10 boats: selfish local Q produced a "tragedy of the commons" and the population "crashed within a couple of years." Local Sarsa learned a sustainable policy and improved faster than monolithic learners, thanks to smaller per-subagent input spaces (3-D per subagent vs. (n+1)-D).

  — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)

### Inferences
- For the proposed design, "each advisor trained greedily on its own reward" is the local Q-learning (egocentric) regime that Russell & Zimdars show is unsound. Sound training requires each advisor to bootstrap on the principal's action or action distribution at s'.
- Their guarantees are tabular and assume a single-agent MDP with an additive reward and Σ_j weights equal to 1. With weights w_j, the same algebra holds for r = Σ_j w_j r_j. Chess is adversarial, and fog-of-war makes it partially observable, so neither Theorem 1 nor Theorem 2 transfers directly. The qualitative lesson about the illusion of control is still the central risk.

### Gaps
- Russell & Zimdars treat neither function approximation nor adversarial or partially observable settings.

---

## Q2. van Seijen et al. 2017: Hybrid Reward Architecture (HRA) on Ms. Pac-Man

### Takeaway
HRA trains one head per reward component and acts on the sum. Two training targets are discussed: each head's own greedy max (the egocentric/selfish target), and evaluation of the uniformly random policy (agnostic). The random-policy target performed best in the fruit task, and it was the one used for the Ms. Pac-Man GVFs. The authors admit the greedy-max target is "generally not consistent." The random-policy target is consistent for the random policy and only "semi-consistent" for control. The Ms. Pac-Man success (25,304 points; 999,990 with "executive memory") relied heavily on domain knowledge: about 1,800 GVFs over map locations, hand-set head weights equal to the in-game point values, normalisation, and exploration heads.

### Cited Findings
- Decomposition: R_env = Σ_k R_k, with a separate agent and Q_k per component. Action selection uses Q_HRA(s,a;θ) := Σ_k Q_k(s,a;θ) (eq. 5). The heads can share lower layers. — [van Seijen et al. 2017, HRA](https://arxiv.org/abs/1706.04208)
- Default loss: each head regresses toward y_k = R_k + γ max_{a'} Q_k(s',a'; θ_{i−1}) (eq. 7), so the heads approximate Q*_1…Q*_n. "Note that Q*_HRA is different from Q*_env and generally not consistent." — [HRA](https://arxiv.org/abs/1706.04208)
- Definitions: a training target is consistent if acting greedily on it is optimal for R_env, and semi-consistent if it gives "a good policy—but not an optimal one." Also: "adding a potential-based additional reward signal to R_env can yield an alternative consistent learning objective that is easier to learn." — [HRA](https://arxiv.org/abs/1706.04208)
- Which target the experiments actually used:
  - Fruit-collection experiments: "Training samples are generated by a random policy… For HRA, we performed experiments with Q*_HRA as training target (using Equation 7), as well as Q^υ_HRA (using Equation 8)… for HRA, using Q^υ_HRA resulted in the best performance" (DQN did best with Q*_env).
  - Ms. Pac-Man, supplementary material: "Each GVF is then in charge of determining the value of the random policy of Ms. Pac-Man state for getting the pseudo-reward placed on the GVF's associated position. The GVFs are trained online with off-policy 1-step bootstrapping with α = 1 and γ = 0.99."

  The headline Ms. Pac-Man result therefore used agnostic (random-policy) heads, not selfish greedy-max heads. This is consistent with Laroche et al.'s statement that agnostic planning "has proved its usefulness on Ms. Pac-Man." — [HRA](https://arxiv.org/abs/1706.04208)
- Ms. Pac-Man aggregator weights were hand-set: "pellets' multiplier is 10, power pellets' 50, fruits' 200, and blue and edible ghosts' 1,000. A ghosts' multiplier of -1,000 has demonstrated to be a fair balance between gaining points and not being killed." This is a hand-tuned negative weight on a danger head. — [HRA](https://arxiv.org/abs/1706.04208)
- Alternative target: evaluate the uniformly random policy υ, with y_k = R_k + γ (1/|A|) Σ_{a'} Q_k(s',a') (eq. 8). Then Q^υ_HRA = Q^υ_env exactly, because all heads evaluate the same policy. "Acting greedily with respect to the Q-values of a random policy might appear to yield a policy that is just slightly better than random, but, surprisingly, we found that for many navigation-based domains Q^υ_HRA acts as a semi-consistent training target." — [HRA](https://arxiv.org/abs/1706.04208)
- Ms. Pac-Man setup:
  - One head per object (pellet, fruit, ghost, blue ghost).
  - Head values are "pseudo Q-values" of location GVFs in [0,1], "multiplied with a weight that is set equal to the reward received when the object is eaten."
  - The second aggregator variant normalises the sum of point-producing heads, then adds the ghost heads "multiplied with a weight vector."
  - Two extra exploration heads: "diversification" (random Q's in [0,20] for the first 50 steps) and count-based (UCB-like).

  — [HRA](https://arxiv.org/abs/1706.04208)
- Results: HRA scored 25,304 (fixed start) and 23,770 (random start) after about 150M frames. A3C trained for 800M frames scored 2,168 / 626 (pixels). The human score is 15,693. The best combination used normalisation plus both exploration types, and "the combination with no exploration at all… performed very poorly." With "executive memory" HRA reached the maximum 999,990 points, which exploits the fixed-start evaluation. — [HRA](https://arxiv.org/abs/1706.04208)
- Scale: "HRA solves Ms. Pac-Man by learning close to 1,800 general value functions… each GVF has a state-space in the order of 10^3 states." HRA cites Horde (Sutton et al. 2011), which "consists of a large number of 'demons' that learn in parallel via off-policy learning." — [HRA](https://arxiv.org/abs/1706.04208)
- Critique from the drQ authors: HRA's sum "generally overestimates the true Q-values, and the corresponding greedy policy can perform arbitrarily poorly, even with infinite training… the components Q_c do not reflect the actual Q-function of the greedy HRA policy." — [Juozapaitis et al. 2019](https://web.engr.oregonstate.edu/~afern/papers/reward_decomposition__workshop_final.pdf)
- According to Laroche et al., the random-policy (agnostic) variant "fails at efficiently gathering the fruits near the ghosts" because it is "overwhelmed by the fear of the ghosts." This "led van Seijen et al. (2017b) to use a dynamic normalisation depending on the number of fruits left on the board." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)

### Inferences
- HRA's success says little about whether summing selfish heads is sound in general. It succeeded in a navigation domain with near-independent, mostly positive object rewards, hand-tuned weights, and heavy domain structure. Chess advisors such as material vs. king safety are tightly coupled and often conflict, which is the regime where the failure modes in Q3 bite.
- Agnostic (random-policy) evaluation is a poor fit for chess. Values under random play are meaningless for tactics: a random continuation hangs pieces, so a "material under random play" advisor would be strongly biased.

### Gaps
- HRA reports no ablation in which head weights are learned rather than set from game point values.

---

## Q3. Laroche, Fatemi, Romoff, van Seijen 2017: Multi-Advisor RL (MAd-RL), with egocentric, agnostic and empathic planning

### Takeaway
The three planning methods differ only in which policy each advisor evaluates when it bootstraps:
- Egocentric evaluates its own local greedy policy. It converges, but it overestimates where advisors disagree and creates attractors: states where staying put beats making progress on any single advisor's goal.
- Agnostic evaluates the random policy. It has no attractors but is overly risk-averse near danger.
- Empathic evaluates the aggregator's greedy policy. It converges to the global optimum when every advisor sees the full state, and it was near-optimal and noise-robust empirically.

The authors favour empathic ("supports the superiority of the empathic planning in the general case").

### Cited Findings
- Aggregator: Q_Σ(x,a) = Σ_j w_j Q_j(x_j, a), with f_Σ(x) = argmax_a Q_Σ(x,a). This is the same weighted sum as the proposed principal. — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Convergence vs. optimality: Theorem 1 (from van Seijen et al. 2017b) says that under Assumption 1 (each advisor's environment is Markov in its local state) and "given any fixed aggregator, global convergence occurs if all advisors use off-policy algorithms that converge in the single-agent setting." The authors add: "Although Theorem 1 guarantees convergence, it does not guarantee the optimality of the converged solution." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- The three methods "differ in the policy they evaluate: egocentric planning evaluates the local greedy policy, agnostic planning evaluates the random policy, and empathic planning evaluates the aggregator's greedy policy." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Egocentric planning (used in Singh & Cohn 1998, Russell & Zimdars 2003, Harutyunyan et al. 2015, and van Seijen et al. 2017a) "suffers from an inversion between the max and Σ operators and, as a consequence, it overestimate[s] the state-action values when the advisors disagree on the optimal action." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Attractor example: in x0 the actions are stay (a0), achieve goal 1 (a1, reward r1) and achieve goal 2 (a2, reward r2). Q^ego_Σ(a0) = γr1 + γr2, Q^ego_Σ(a1) = r1 and Q^ego_Σ(a2) = r2. "If γ > r1/(r1+r2) and γ > r2/(r1+r2), the local egocentric planning commands to execute action a0 sine die." Adding randomness does not help. — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Definition 1 (exact): "An attractor x is a state where the following strict inequality holds: max_{a∈A} Σ_j w_j Q^ego_j(x_j,a) < γ Σ_j w_j max_{a∈A} Q^ego_j(x_j,a)." In words, the best joint action now is worth less than every advisor getting its own best action one step later. This is the max/Σ inversion. Theorem 2: "State x is attractor, if and only if the optimal egocentric policy is to stay in x if possible." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Definition 2: an advisor is progressive if ∀x,a: Q^ego_j(x,a) ≥ γ max_{a'} Q^ego_j(x,a'), meaning no action is worse than losing a turn. Theorem 3: "If all the advisors are progressive, there cannot be any attractor." The condition "is very restrictive" and does not apply to navigation or to tasks "with states that terminate the trajectory while some goals are still incomplete." Being attractor-free also "does not guarantee optimality." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Theorem 4 covers deterministic tasks with reversible actions and R ≥ 0: x is guaranteed not to be an attractor if γ ≤ 1/(|A|−1). Larger action sets therefore need a smaller γ. In the 3-fruit example with γ > 0.5, "the aggregator would opt to go South and hit the wall indefinitely." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Agnostic planning: "Local agnostic planning is equivalent to the global agnostic planning" and has no attractors. However, "acting greedily with respect to Q^agn_Σ only guarantees to be better than a random policy and in general may be far from being optimal." In dangerous environments, "crossing a bridge would be avoided." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Empathic planning has the local equation Q_j(x_j,a) = E[r_j + γ Q_j(x'_j, f_Σ(x'))]. Theorem 5: "Assuming that all advisors are defined on the full state space, MAd-RL with empathic planning converges to the global optimal policy." With reduced advisor state spaces it has no convergence guarantee, but "we never encountered a case where the convergence was not obtained." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- Pac-Boy results (one advisor per fruit location): "egocentric-γ=0.9" shows the attractor in practice ("Pac-Boy goes straight to the centre area of the grid and does not move until a ghost comes too close") and "barely performs better than linear Q-learning." Egocentric-γ=0.4 and empathic-γ=0.9 are near-optimal. When Gaussian noise is added to the advisors' rewards, "the empathic planning performs better than the egocentric one even under noise with a 100 times larger variance." Egocentric "incompatible with high γ values, is therefore myopic." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)
- A supervised side experiment found that decomposed egocentric targets are easier for a DNN to fit than the true RL target, "even without any state space reduction." — [Laroche et al. 2017](https://arxiv.org/abs/1704.00756)

### Inferences
- Chess maps closely onto the attractor setting, because there are many quiet "waiting" moves. Theorem 4 does not literally apply: its γ ≤ 1/(|A|−1) is only a sufficient condition, and it requires deterministic, reversible actions and R ≥ 0. Chess breaks all three through captures and pawn moves, the opponent's intervening move, and possibly negative shaping rewards. Read it as an illustration that a large |A| combined with γ near 1 invites attractors, not as a threshold for chess. Advisors also routinely disagree: the material advisor wants to grab, king safety wants to consolidate, and scouting wants to probe. Egocentric advisors with γ near 1 would be expected to favour noncommittal shuffling moves, from which every advisor optimistically believes "its" plan remains available next turn.
- Fog-of-war violates Assumption 1, since an advisor's local observation is not Markov. The opponent's adaptation also makes the environment non-stationary, so even Theorem 1's convergence is not guaranteed.
- Empathic planning implies a concrete design: every advisor's TD target uses the principal's action at s'. For a softmax principal that means the expected-Sarsa form Σ_{a'} π_principal(a'|s') Q_j(s',a'), with the same weights w the principal uses at that time.

### Gaps
- "Oscillation" as a named failure mode does not appear in the v2 text of arXiv 1704.00756 that I read. Only attractors are formally defined, plus the observed instability of egocentric planning under noise. It may be in a different version or in another Laroche paper; I could not verify it.
- Theorem 5 assumes a fixed w. The paper does not analyse what happens when w changes after training.

---

## Q4. drQ (Juozapaitis et al. 2019), TD(Δ) (Romoff et al. 2019), Horde/GVFs, OVM, and Successor Features + GPI (Barreto et al. 2017)

### Takeaway
- drQ is the off-policy fix for decomposed control. Each component bootstraps on the overall greedy action a+ = argmax_a Σ_c Q_c(s',a), which gives both the optimal overall policy and correct per-component values. drSARSA instead learns the components of the exploration policy.
- SF/GPI formalises the linear-weight case: Q^π_w = ψ^π · w, exact for any w as long as ψ is the feature-return vector of one fixed policy π. GPI's guarantee concerns taking the max over policies' Q under a common reward, not summing different components' selfish Q's.

### Cited Findings
- drQ vs. alternatives:
  - HRA "is an off-policy algorithm, but is unsound."
  - drSARSA "is an on-policy algorithm and thus learns the Q-function of an exploration policy instead of the greedy policy."
  - drQ "first computes the overall greedy action, i.e., a+ = argmax_{a'} Σ_c Q_c(s',a'), and then updates each component Q_c(s,a) toward r_c + γ Q_c(s',a+)."

  — [Juozapaitis, Koul, Fern, Erwig, Doshi-Velez 2019](https://web.engr.oregonstate.edu/~afern/papers/reward_decomposition__workshop_final.pdf)
- drQ convergence: it "achieves two notions of convergence: (1) convergence of Q^t(s,a) to Q*(s,a), and (2) convergence of each Q^t_c(s,a) to Q^{π*}_c(s,a)." The setting is tabular, and the authors adapted a prior multi-agent off-policy variant. The paper was evaluated on CliffWorld and other domains and introduced "minimal sufficient explanations" (MSX). — [Juozapaitis et al. 2019](https://web.engr.oregonstate.edu/~afern/papers/reward_decomposition__workshop_final.pdf)
- On drSARSA: "Standard SARSA results imply convergence to an ε-optimal policy, and each Q_c converges to the correct component Q-function of that policy [Russell & Zimdars]." However, "the Q-components Q_c… do not reflect the value of the greedy policy, but rather the value of the exploration policy followed during learning." — [Juozapaitis et al. 2019](https://web.engr.oregonstate.edu/~afern/papers/reward_decomposition__workshop_final.pdf)
- Romoff et al. 2019, TD(Δ): this decomposes a single value function across time-scales rather than across reward types. TD(Δ) "breaks down a value function into a series of components based on the differences between value functions with smaller discount factors," with V_{γZ} = Σ_z W_z. Theorem 1 states equivalence to standard TD(λ) under matching step sizes and λ_zγ_z = λγ. The paper also gives bias-variance analysis for truncated or mismatched settings. — [Romoff et al., ICML 2019](https://arxiv.org/abs/1902.01883)
- Successor features: if r(s,a,s') = φ(s,a,s')ᵀw, then Q^π(s,a) = ψ^π(s,a)ᵀw. A new task w_{n+1} needs only w_{n+1}: Q^{π_i}_{n+1} = ψ^{π_i}ᵀ w_{n+1}, which "reduces the computation… to the much simpler supervised problem of approximating w_{n+1}." — [Barreto et al. 2017](https://arxiv.org/abs/1606.05312)
- GPI, Theorem 1 of Barreto et al.: given policies π_1..π_n with approximations |Q^{π_i} − Q̃^{π_i}| ≤ ε, define π(s) ∈ argmax_a max_i Q̃^{π_i}(s,a). Then Q^π(s,a) ≥ max_i Q^{π_i}(s,a) − 2ε/(1−γ). — [Barreto et al. 2017](https://arxiv.org/abs/1606.05312)
- Barreto et al. Theorem 2 (transfer bound): Q^{π*_i}_i − Q^π_i ≤ (2/(1−γ)) (φ_max min_j ||w_i − w_j|| + ε). — [Barreto et al. 2017](https://arxiv.org/abs/1606.05312)
- Orchestrated Value Mapping (Fatemi & Tavakoli, ICLR 2022) "generalizes and subsumes Q-Learning, Log Q-Learning, and Q-Decomposition." It combines arbitrary value mappings with linear reward-channel decomposition and claims a convergence proof for the general class. I read only the abstract. — [Fatemi & Tavakoli 2022](https://arxiv.org/abs/2203.07171)

### Inferences
- Under SFs, the proposed advisors (material, king safety, scouting, pressure, win), if each is trained to evaluate the same principal policy π, form exactly a successor-feature vector ψ^π with φ = (r_material, r_king, r_scout, r_pressure, r_win). Then Σ_j w_j Q_j = Q^π_{r_w} holds exactly for any w. This is the condition that makes a weighted sum of advisor Q's meaningful, and it is precisely what egocentric training breaks.
- GPI is a different combiner: it takes max over policies of their Q under one reward. A GPI-style principal for this design would evaluate each advisor's greedy policy on the win reward (or on r_w) and take the max. That gives the guarantee "no worse than the best advisor policy, minus 2ε/(1−γ)." Summing selfish Q's carries no such guarantee.
- drQ gives a practical deep-RL recipe for a greedy principal: the advisor target is r_j + γ Q_j(s', argmax_a Σ_k w_k Q_k(s',a)). For a softmax principal the expected-Sarsa analogue is r_j + γ Σ_a π_w(a|s') Q_j(s',a).

### Gaps
- None of drQ, OVM or SF/GPI addresses two-player zero-sum games or imperfect information. In fog-of-war chess, "Q of the principal policy" also depends on the opponent's policy, so all of these results hold only for a fixed opponent.
- I did not find the drQ final venue printed in the PDF; it is commonly cited as an IJCAI 2019 XAI workshop paper (unverified here).

---

## Q5. Multi-objective RL with linear scalarization: Abels et al. 2019 (dynamic weights) and Yang et al. 2019 (envelope Q-learning)

### Takeaway
Linear scalarization can recover only policies on the convex coverage set (CCS). Pareto-optimal policies in concave regions of the front are unreachable for any weight vector. When weights change, Q-functions trained for one w are wrong for another. The literature's fix is to condition a vector-valued Q on w (UVFA-style), not to refit w on top of fixed per-objective Q's.

### Cited Findings
- "An optimal solution for an MOMDP under linear f is a convex coverage set (CCS), i.e., a set of undominated policies containing at least one optimal policy for any linear scalarization." If weights are known and fixed, "the problem can be solved via single-objective RL." Linear scalarization does not apply when "the user utility cannot be expressed with a linear function." — [Abels et al., ICML 2019](https://arxiv.org/abs/1809.07803)
- Abels et al. propose a Conditioned Network (CN), "a UVFA… adapted to output Q-value-vectors conditioned on an input weight vector." They also propose Diverse Experience Replay (DER) "to improve sample efficiency and counter the replay buffer's bias to recent weight vectors." — [Abels et al. 2019](https://arxiv.org/abs/1809.07803)
- Yang et al.: "The Pareto frontier may encapsulate local concave parts (points A-H, plus point K), whereas CCS is a convex subset of Pareto frontier (points A-H)." Point K is Pareto-optimal but not reachable by any linear preference. Scalarized MORL updates "only search along ω1 direction leading to non-optimal L, even if solution D has been seen under ω2." — [Yang, Sun, Narasimhan, NeurIPS 2019](https://arxiv.org/abs/1908.08342)
- Envelope Q-learning theorems:
  - Theorem 1: the preferred optimal multi-objective Q* is a fixed point of the envelope optimality operator T.
  - Theorem 2: T is a γ-contraction in a pseudo-metric, d(TQ,TQ') ≤ γ d(Q,Q').
  - Theorem 3: a multi-objective Banach fixed-point result, lim_n d(TⁿQ, Q*) = 0.

  The update takes sup over actions and over preferences ω' of ωᵀQ(s',a',ω') (the "convex envelope"). — [Yang et al. 2019](https://arxiv.org/abs/1908.08342)

### Inferences
- In the proposed design the components are not true objectives. Only winning matters, and the others are auxiliary or shaping signals. The CCS/convexity limitation therefore matters less than the policy-mismatch problem. However, if the hope is that some w makes Σ w_j Q_j induce a winning policy, that policy must lie on the CCS of the advisor-objective space. A win-optimal policy that trades off material and king safety non-linearly, or in a state-dependent way, may not be expressible with a single global w.
- If w is to be tuned post hoc (CMA-ES), the MORL-sound way is to train w-conditioned advisors Q_j(s,a; w), as in Abels CN or envelope Q-learning, so that the Q's being weighted evaluate the w-principal's own policy.
- State-dependent weights would be a natural extension, for example by game phase or by the uncertainty level of the fog-of-war belief. HRA already used a dynamic normalisation by fruits remaining.

### Gaps
- I did not extract Abels' or Yang's quantitative benchmark numbers. They are not needed for the design argument.

---

## Q6. Learning combination weights post hoc: is NNLS on component Q's to predict win-Q supported?

### Takeaway
I found no RL paper that fits non-negative least-squares weights over separately learned (selfish) component Q's to regress a win value and then acts on the weighted sum. The closest support comes from two places: successor features (fitting w by regression is principled only when the ψ/Q components evaluate a common policy), and chess engine tuning (Texel's method fits evaluation-term weights by logistic regression to game results). Texel fits static features, not policy-dependent Q-values.

### Cited Findings
- Under SFs, transferring to a new reward "reduces the computation of all Q^{π_i}_{n+1} to the much simpler supervised problem of approximating w_{n+1}." This w is a reward-weight regression on φ, and it gives exact values only for policies whose ψ was learned. — [Barreto et al. 2017](https://arxiv.org/abs/1606.05312)
- Texel's tuning method (Peter Österlund) is "a concrete instance of a logistic regression." It fits evaluation weights so that sigmoid(K·eval) predicts game results (0, 0.5, 1). Practitioners note that it cannot learn "things that it does not understand at all," and that tuned weights interact with search parameters such as pruning margins. — [Chessprogramming wiki: Texel's Tuning Method](https://www.chessprogramming.org/Texel%27s_Tuning_Method); [TalkChess thread](https://talkchess.com/forum/viewtopic.php?p=561291)
- HRA set head weights by hand from domain knowledge (object point values) plus normalisation; they were not learned. — [HRA](https://arxiv.org/abs/1706.04208)
- Linear decomposability: Q^π = Σ_j Q^π_j for one π (Russell & Zimdars). With selfish components, Σ_j Q̃_j ≥ Q* is an optimistic upper bound that is not tied to any policy. — [Russell & Zimdars 2003](https://aima.eecs.berkeley.edu/~russell/papers/ml03-qdecomp.pdf)
- Potential-based shaping keeps the training target consistent: "adding a potential-based additional reward signal to R_env can yield an alternative consistent learning objective." — [HRA](https://arxiv.org/abs/1706.04208)

### Inferences
- **A terminal-win regressor would absorb the fit.** If the terminal-win advisor is itself one of the NNLS regressors and the target is win-Q, the fit will put nearly all weight on that advisor. It then says nothing about how to weight the shaping advisors for control.
- **NNLS regressing win-Q on advisor Q's is a value-prediction fit, not a control guarantee.** Even a perfect fit does not imply that argmax Σ w_j Q_j is a good policy. If the advisors are egocentric, the regressors are optimistic, policy-inconsistent quantities, so the regression error is uninterpretable with respect to the deployed policy. NNLS also forces w ≥ 0, ruling out a negative weight on, say, an over-aggressive "pressure" advisor.
- **Recommended consistent alternative.** Make the shaping advisors potential-based: r_material = γΦ_mat(s') − Φ_mat(s), and similarly for king safety and pressure. Train every advisor empathically or drQ-style on the principal's policy. Then Q_win + Σ_j w_j Q^π_{shape,j} differs from Q_win^π only by state-dependent terms (−Σ w_j Φ_j(s) plus the terminal-potential handling). The greedy policy is therefore invariant to w in the exact case, and w only affects learning speed and approximation. CMA-ES on actual win rate, as an outer loop with advisors re-trained or w-conditioned, is then a legitimate hyperparameter search. NNLS on Q-values is not needed. This is inferred from Ng-style shaping as referenced in HRA, not from a source that tests it.
- **Information gain is not potential-based.** A scouting/information-gain reward is not naturally potential-based. Its Q-function under a fixed policy is still well defined, but its weight changes the optimal policy, so tuning it by win rate (CMA-ES) is more appropriate than regressing it.
- **If NNLS is kept, three practices make the fit meaningful:**
  1. Generate the regression data on-policy from the principal itself.
  2. Have the advisors evaluate that principal (empathic/drQ/expected-Sarsa targets).
  3. Iterate: fit w, re-train the advisors under the new principal, refit. This is a policy-iteration-like loop, with no convergence guarantee found in the literature.

### Gaps
- I found no paper evaluating NNLS or CMA-ES over decomposed Q-heads for game control.
- I did not locate a source that formally treats inverse-RL-style linear reward recovery from component Q's in this setting. Apprenticeship learning (Abbeel & Ng 2004) uses linear rewards over feature expectations, which are policy-specific SFs, but I did not fetch it in this session.
- No literature applies reward decomposition to imperfect-information two-player games such as Fog-of-War or Dark Chess. Whether empathic planning's optimality survives adversarial non-stationarity is an open question.
