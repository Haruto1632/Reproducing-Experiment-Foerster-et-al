# Switch Riddle n=3 fidelity audit

## Audit basis

Compared the current `switch_riddle` environment, agent, communication controllers, and fast training path with Foerster et al. (2016), Sections 6.1–6.2 and Appendix A Algorithm 1. The requested authors' repository is configured as `origin`, but fetching it failed because this workspace cannot write `.git/FETCH_HEAD`; the checkout does not include its source or `switch_3_dial` configuration. Therefore repository-specific details below remain unverified. This audit does not treat paper-only claims as if they came from the released config.

## Verified discrepancy before code changes

1. **Target recurrent state (confirmed against paper Algorithm 1):** `run_batch_fast` evaluates target-network Q-values using the online network's saved `h_next`. Algorithm 1 maintains each target C-Net's own recurrent state during the forward episode unroll. For a 2-layer GRU, target values computed from online hidden states do not match the specified target network. Fix: independently advance target-network hidden state on each input during the rollout, and use the resulting target hidden states for the next-state values.

## Compared and consistent with paper text

- n=3 has horizon `T=4n−6=6`; room selection is uniform with replacement.
- Only the room occupant can choose None/Tell, and the message is read only by the room occupant; reward is zero except Tell (+1 if all visited, −1 otherwise), and Tell or horizon terminates.
- Inputs include observation, previous message, previous action, and agent ID; the agent uses 128-wide embeddings, a 2-layer GRU with hidden size 128, and a two-layer output MLP. Message BatchNorm is present. Exact BN placement/config and task observation network remain assumptions absent the released source.
- Shared and -NS parameter variants are represented; RIAL has separate action/message Q heads and DIAL has a continuous output with DRU sigma=2 and differentiable message routing.
- epsilon=.05, gamma=1, RMSProp learning rate 5e-4/momentum=.95, batch size 32, zero initial hidden state, and target hard copy threshold of 100 completed episodes are configured.

## Unresolved without released source/config

- Exact `switch_3_dial` settings and original implementation behavior, including message/action sampling details, BN axes/momentum/epsilon, target update accounting, epoch/update mapping, RMSProp remaining defaults, and evaluation cadence/seed count.
- Oracle denominator/normalization implementation: paper says highest average reward with true state; local plotting code uses a theoretical expected reward, but this has not been compared to the released config or figure-generation code.
- The paper defines timing order only broadly. Current environment writes the room occupant's message and advances the room before the next observation; any more specific implementation discrepancy awaits source inspection.

## Change scope

Only the target recurrent-state discrepancy above is being fixed. No environment, architecture, reward, hyperparameter, or algorithm change is intended.
