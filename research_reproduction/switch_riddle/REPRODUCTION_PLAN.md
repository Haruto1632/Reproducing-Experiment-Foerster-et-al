# Final Reproduction Specification

This document provides the final, explicit specification for reproducing the Foerster et al. 2016 Switch Riddle experiment. Every item is classified according to its source fidelity:
*   **[A]** Directly specified by paper
*   **[B]** Strongly implied
*   **[C]** Necessary implementation assumption
*   **[D]** Unknown / unresolved

## 1. Environment & Mechanics
*   **Target Problem:** Switch Riddle for $n = 3$ and $n = 4$ agents. [A]
*   **Time Horizon:** $T = 4n - 6$. [A]
*   **Prisoner Selection:** Uniformly at random with replacement (the warden chooses one agent per day). [A]
*   **Observation ($o_t^a$):** Binary indicator $\in \{0, 1\}$ denoting if agent $a$ is in the interrogation room. [A]
*   **Actions ($u_t^a$):**
    *   If in room ($o_t^a = 1$): $u_t^a \in \{\text{"None"}, \text{"Tell"}\}$. [A]
    *   If outside room ($o_t^a = 0$): $u_t^a \in \{\text{"None"}\}$. [A]
*   **Communication / Message ($m_t^a$):** 1-bit switch/message, left by the agent in the room, and available *only* to the next agent in the room as $m_{t-1}^a$. [A]
*   **Reward ($r_t$):** $0$ at all non-terminal steps. If an agent chooses "Tell": $+1$ if all agents have visited the room, $-1$ otherwise. [A]
*   **Terminal Condition:** Episode ends immediately when an agent chooses "Tell", or when the maximum time-step $T$ is reached. [A]

## 2. Agent Architecture
*   **Inputs:** Tuple $(o_t^a, m_{t-1}^a, u_{t-1}^a, a)$. [A]
*   **Embeddings:**
    *   Agent index $a$ $\rightarrow$ Lookup table (size 128). [A]
    *   Previous action $u_{t-1}^a$ $\rightarrow$ Lookup table (size 128). [A]
    *   Previous message $m_{t-1}^a$ $\rightarrow$ 1-layer MLP (size 128). [A]
    *   Task observation $o_t^a$ $\rightarrow$ Task-specific network (size 128). [A]
*   **State Representation ($z_t^a$):** Element-wise sum of the four embeddings. [A]
*   **RNN:** 2-layer GRU with 128 hidden units. Initial hidden state $h_0^a = \mathbf{0}$. [A]
*   **Output:** 2-layer MLP generating values for actions ($|U|$) and messages ($|M|$). [A]

## 3. RIAL Details
*   **Algorithm:** Deep Recurrent Q-Network (DRQN), independent Q-learning. [A]
*   **Outputs:** Separate $Q_u$ (for actions) and $Q_m$ (for messages). [A]
*   **Selection:** Separate $\epsilon$-greedy selection for the environment action $u_t^a$ and the communication action $m_t^a$. [A]
*   **Replay:** No experience replay. [A]
*   **Recurrency:** Previous action and previous message fed in as recurrent inputs. [A]
*   **Variants:** Parameter-sharing and non-sharing (RIAL-NS) variants. [A]

## 4. DIAL Details
*   **Algorithm:** C-Net. [A]
*   **Communication Flow:** Real-valued communication passed continuously during training. [A]
*   **Discretise/Regularise Unit (DRU):**
    *   Training: Logistic regularization with Gaussian noise $\sigma=2$ ($Logistic(\mathcal{N}(m_t^a, \sigma))$). [A]
    *   Execution/Eval: Binary discretization ($\mathbb{1}\{m_t^a > 0\}$). [A]
*   **Gradients:** Gradient flows from the receiving agent's Q-loss backward through the communication channel to the sender's message output. [A]

## 5. Paper-Specified Parameters
*   $\epsilon = 0.05$ [A]
*   $\gamma = 1.0$ [A]
*   Optimizer = RMSProp [A]
*   Momentum = $0.95$ [A]
*   Learning rate = $5 \times 10^{-4}$ [A]
*   Target network reset = **every 100 episodes** [A]
*   Parallel episodes batch = 32 [A]
*   Gaussian noise $\sigma = 2$ [A]

## 6. Implementation Assumptions
*   **TaskMLP Architecture:** The paper dictates a "task-specific network" producing a 128-size embedding but does not define its layers.
    *   *Assumption [C]:* A 1-layer linear projection mapping the scalar $o_t^a$ to a 128-dimensional vector.
*   **Batch Normalization:** The paper states batch normalization was used to preprocess $m_{t-1}$.
    *   *Assumption [C]:* A standard 1D BatchNorm is applied to the 1-dimensional $m_{t-1}$ input directly before it passes into the 1-layer MLP embedding.
*   **NoComm Baseline:** The paper describes a "shared parameters baseline without communication".
    *   *Assumption [C]:* The exact same architecture as RIAL (shared), but the received message is hardcoded to $0$, and any output message is ignored/discarded. No gradients flow for messages.
*   **Initial Message:** Algorithm 1 directly supports $h_0^a = \mathbf{0}$ [A], but does not explicitly specify the initial previous-message value $m_{-1}$.
    *   *Assumption [C]:* The initial message $m_{-1}$ fed at $t=0$ is $0$.

## 7. Paper Ambiguities
*   **Epoch vs Episode Terminology:** The paper's text states n=3 methods learn an optimal policy in "5k episodes", while Figure 4's x-axis plots "# Epochs" (reaching 5k for n=3, 40k for n=4). It is utterly ambiguous whether 1 Epoch = 1 Episode, 1 Epoch = 1 Batch of 32 episodes, or something else.
    *   *Classification [D]:* Unknown/unresolved. We will not invent an equivalence. We will track both distinct metrics and plot the "epoch" counter to match the visual scale of Figure 4.
*   **Target Network Reset Frequency:** The paper commands "reset every 100 episodes", while parallelizing into batches of 32. 100 is not evenly divisible by 32.
    *   *Classification [D]:* Unknown interaction.
    *   *Resolution [C]:* The scientific specification is every 100 completed episodes. The implementation tracks individual completed episodes. If target updates can only occur at batch boundaries, the update occurs at the first batch boundary after each 100-episode threshold. This batch-boundary behavior is an implementation detail, not a claim about the paper.

## 8. Metrics and Counters
To ensure transparent mapping between our reproduction and the paper, our logs must retain and strictly separate the following distinct counters:
1.  **`completed_episodes`**: The total count of individual episodes finished.
2.  **`update_steps`**: The number of parameter updates / backwards passes executed.
3.  **`env_timesteps`**: The total number of state transitions sampled across all environments.
4.  **`plotted_x_axis`**: The explicit "epoch" counter intended to match Figure 4's scaling.
5.  **`target_update_count`**: The number of times the target network has been reset.
6.  **`last_target_update_episode`**: The exact episode count at which the last target network reset occurred.

## 9. Figure 4 Reproduction Procedure
*   **Visual Reference:** Produce curves matching Figure 4 from the PDF up to ~5k on the plotted scale for $n=3$, and ~40k on the plotted scale for $n=4$.
*   **Algorithms:** DIAL, DIAL-NS, RIAL, RIAL-NS, NoComm.
*   **Normalization:** Rewards must be normalized by the "highest average reward achievable given access to the true state (Oracle)". Since true optimal reward is 1.0 (always Tell when correct), the optimal normalized reward is 1.0.
*   **Averaging:** Results will be averaged across several runs (independent seeds).

## 10. Reproducibility Requirements
*   Every independent run must have its exact random seed recorded.
*   We will strictly record the final configuration, python version, and package versions used.
*   We will strictly use the algorithm structure prescribed in 2016. No modern abstractions (PPO, GAE) will be substituted.

## 11. Acceptance Criteria
1.  The environment logic perfectly handles the $4n-6$ horizon and $+1/-1/0$ reward structure.
2.  The implementations for RIAL and DIAL correctly model the discrete vs. continuous (and differentiable) message spaces.
3.  No "silently equated" metrics (e.g. epochs vs episodes).
4.  Reproduction is evaluated using asymptotic performance, approximate learning speed, qualitative curve shape, relative behavior, and variance across seeds, rather than requiring exact numerical reproduction or exact algorithm ordering as a binary pass/fail criterion. Unspecified implementation details prevent claiming exact numerical reproduction.
