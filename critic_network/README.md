# Global MEO–LEO Action Critic

This package implements an action-conditioned global critic. It always trains
from observational samples and can optionally rerank greedy LEO actions after
its sample/update gate is ready. MEO actions, epsilon exploration, PPO, and
intra-domain LEO decisions are never changed by the critic.

For every LEO forward/compute step executed under an active cross-domain MEO
trace, the trainer records:

1. the latest global snapshot and the executed joint action;
2. the first global snapshot whose simulation time is later than the action;
3. the packet's terminal reward, success flag, and end-to-end delay.

The model predicts per-node queue/compute deltas, per-link load deltas, packet
success probability, delay, and a scalar global impact. Runtime shadow
predictions are appended to `packet.meo_decision_trace["global_critic_predictions"]`.

Configuration lives at `transformer.critic`. Omitting the block, or setting
`enabled: false`, preserves the previous behavior. Action reranking is also
off by default and requires `selection_enabled: true`. For legal candidates it
uses `zscore(Q) - selection_weight * zscore(-impact)`; any unavailable or
non-finite critic result falls back to the original Q argmax.
