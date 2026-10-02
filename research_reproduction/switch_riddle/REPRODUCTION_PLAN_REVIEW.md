# 1. Overall Assessment
The `REPRODUCTION_PLAN.md` accurately captures the core mechanics of the Switch Riddle environment and the fundamental network architecture proposed in Foerster et al. 2016. However, a paper-fidelity review reveals several contradictions within the paper itself regarding terminology (epochs vs. episodes), ambiguities around target network updates when using parallel batching, and a few algorithmic details (like hidden state initialization and RIAL's action selection) that were omitted from the plan.

# 2. Correct Parts
The following elements from the plan are directly supported by the paper **[A]**:
*   **Switch Riddle mechanics:** $n=3, 4$, $T=4n-6$. (Sec 6.2)
*   **Agent selection / warden behavior:** Uniform at random with replacement. (Sec 6.2)
*   **Observation definition:** $o_t^a \in \{0, 1\}$. (Sec 6.2)
*   **Action definition:** $u_t^a \in \{\text{"None"}, \text{"Tell"}\}$ in room, else "None". (Sec 6.2)
*   **Message definition:** 1-bit, continuous for DIAL, discrete for evaluation, only available in the room. (Sec 6.2, Sec 5.2)
*   **Reward:** 0 unless "Tell", then $+1$ if all visited, $-1$ otherwise. (Sec 6.2)
*   **Terminal condition:** Ends on "Tell" or max $T$. (Sec 6.2)
*   **DRU mechanics:** $Logistic(\mathcal{N}(m_t^a, \sigma))$ during training, $\mathbb{1}\{m_t^a > 0\}$ during eval. (Sec 5.2)
*   **Gaussian noise:** $\sigma = 2$. (Sec 6)
*   **GRU architecture:** 2-layer GRU with 128 hidden units, embedding concatenations (element-wise sum). (Sec 6.1)
*   **Optimizer:** RMSProp, momentum=0.95, lr=5e-4. (Sec 6)
*   **Epsilon:** $\epsilon = 0.05$. (Sec 6)
*   **Gamma:** $\gamma = 1.0$. (Sec 6)
*   **32 parallel episodes:** Executed in batches of 32. (Sec 6)
*   **NoComm Baseline:** Shared parameter baseline without communication. (Sec 6)
*   **Shared vs Non-shared (-NS):** Shared learns one network, NS learns independent networks. (Sec 5.1, Sec 6)
*   **Evaluation & Fig 4 target:** Compare performance curves for $n=3$ and $n=4$ across the methods. (Sec 6.2)

# 3. Unsupported Assumptions
*   **Batch Normalization:** 
    *   *Current Plan:* "apply a 1D BatchNorm directly to the scalar/vector $m_{t-1}$ before passing it into its 1-layer MLP embedding."
    *   *Paper Says:* "performance and stability improved when a batch normalisation layer was used to preprocess $m_{t-1}$." (Sec 6.1)
    *   *Classification:* **[B] Strongly implied by the paper.**
    *   *Recommended Change:* Keep this assumption as it is standard deep learning practice for preprocessing inputs.
*   **Task-specific network architecture for $o_t^a$:**
    *   *Current Plan:* "$o_t^a$ processed through a task-specific network (size 128)."
    *   *Paper Says:* "$o_t^a$ is processed through a task-specific network that produces an additional embedding of the same size." (Sec 6.1)
    *   *Classification:* **NOT SPECIFIED IN PAPER** (exact layer count/type).
    *   *Recommended Change:* Explicitly assume a 1-layer linear projection (or MLP) from the scalar observation to a 128-dimensional embedding.

# 4. Potentially Incorrect Parts
*   **Epoch vs Episode terminology:**
    *   *Current Plan:* Assumes 1 Epoch = 1 Training Step = 1 Batch of 32 parallel episodes.
    *   *Paper Says:* Text says "All four methods learn an optimal policy in 5k episodes" (Sec 6.2). However, Figure 4's x-axis plots "# Epochs" up to 5k for $n=3$, and up to 40k for $n=4$. If 1 epoch = 1 episode, then 5k episodes / 32 batch size = ~156 weight updates, which is vastly too few for deep RL to converge. 
    *   *Classification:* **[D] Not supported / potentially incorrect (Paper self-contradiction).**
    *   *Recommended Change:* Treat 1 Epoch as 1 training step (which performs a backward pass on a batch of 32 parallel episodes). This means "5k Epochs" equals 160k total episodes, which aligns with standard deep RL convergence times.
*   **Target-network updates:**
    *   *Current Plan:* "Hard reset every 100 episodes."
    *   *Paper Says:* "the target network is reset every 100 episodes." (Sec 6).
    *   *Classification:* **[D] Potentially incorrect practice.** Since episodes are executed in batches of 32, updating exactly at "100 episodes" is not cleanly divisible (100 / 32 = 3.125). 
    *   *Recommended Change:* Clarify this to mean the target network is reset every 100 *epochs* (training steps) to align with standard batched RL training, or every 3 batches.

# 5. Missing Paper Details
*   **RIAL Message Selection:** 
    *   *Current Plan:* Mentions "discrete Q-learning" for RIAL.
    *   *Paper Says:* "The action selector separately picks $u_t^a$ and $m_t^a$ from $Q_u$ and $Q_m$, using an $\epsilon$-greedy policy." (Sec 5.1).
    *   *Classification:* **[A] Directly supported.**
    *   *Recommended Change:* Explicitly state that RIAL uses two independent $\epsilon$-greedy selectors for environment actions and messages.
*   **Hidden State and Message Initialization:**
    *   *Current Plan:* Omitted.
    *   *Paper Says:* $h_0^a = \mathbf{0}$ for each agent. Algorithm 1 also implicitly passes $m_{-1}^a$, which needs an initial zero-state. (Appendix A, Alg 1).
    *   *Classification:* **[A] Directly supported.**
    *   *Recommended Change:* Add initialization of $h_0^a$ and $m_{-1}^a$ to zeros.
*   **DIAL Message Gradient Calculation:**
    *   *Current Plan:* "DIAL continuous message flow and gradient passing".
    *   *Paper Says:* The message gradient $\mu_t^a$ is driven explicitly by the downstream Q-network error gradient (Appendix A, Alg 1, Eq 4). 
    *   *Classification:* **[A] Directly supported.**
    *   *Recommended Change:* Note that the message gradient must explicitly incorporate the backpropagated Q-loss from the receiving agent.

# 6. Exact Changes Required Before Coding
1.  **Epoch Definition:** Define `1 Epoch = 1 Training Step (batch of 32 parallel episodes)`. Train for 5k epochs for $n=3$ and 40k epochs for $n=4$.
2.  **Target Network:** Reset the target network every `100 Epochs` (training steps), not 100 individual episodes.
3.  **TaskMLP:** Define as a 1-layer linear projection to size 128 (since it is NOT SPECIFIED IN PAPER).
4.  **RIAL Action Selection:** Explicitly define independent $\epsilon$-greedy sampling for $u_t^a$ and $m_t^a$.
5.  **Initialization:** Explicitly set $h_0^a = \mathbf{0}$ and $m_{-1} = \mathbf{0}$ at the start of each episode.
6.  **DIAL Gradients:** Confirm that DIAL explicitly passes gradients from the recipient's $Q$-loss back to the sender's message output.

# 7. Final Approved Experiment Specification
*(Pending resolution of the items above, this section will contain the fully corrected plan.)*
