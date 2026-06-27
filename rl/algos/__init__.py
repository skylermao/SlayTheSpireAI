"""Learning algorithm: AlphaZero-style MCTS, the network, self-play, and the reward.

Depends on `rl.core` (the environment). Imported by `rl.train` (orchestration) and
`rl.tune`.
"""

from .mcts import (
    MCTS,
    MCTSConfig,
    StubEvaluator,
    DecisionNode,
    ChanceNode,
    play_combat,
)
from .net import CombatNet, NeuralEvaluator
from . import rewards

__all__ = [
    "MCTS", "MCTSConfig", "StubEvaluator", "DecisionNode", "ChanceNode", "play_combat",
    "CombatNet", "NeuralEvaluator", "rewards",
]
