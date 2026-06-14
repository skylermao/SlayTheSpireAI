"""Combat-only RL interface for sts_lightspeed.

The single class you drive is `CombatSession` (see session.py). It owns one live
BattleContext and exposes reset / legal_actions / step plus a clone/afterstate/peek
model interface for afterstate-style (sasRL) search. No gym dependency.
"""

from .session import (
    CombatSession,
    LegalAction,
    Transition,
    END_TURN,
    PLAY_CARD,
    USE_POTION,
    SELECT_CARD,
    SKIP_SELECT,
)
from .scenario import (
    CombatConfig,
    RewardConfig,
    DatasetSampler,
    resolve_card,
    resolve_relic,
    resolve_encounter,
    resolve_potion,
)
from .mcts import (
    MCTS,
    MCTSConfig,
    StubEvaluator,
    DecisionNode,
    ChanceNode,
    play_combat,
)
from . import encoding

__all__ = [
    "CombatSession",
    "LegalAction",
    "Transition",
    "MCTS",
    "MCTSConfig",
    "StubEvaluator",
    "DecisionNode",
    "ChanceNode",
    "play_combat",
    "END_TURN",
    "PLAY_CARD",
    "USE_POTION",
    "SELECT_CARD",
    "SKIP_SELECT",
    "CombatConfig",
    "RewardConfig",
    "DatasetSampler",
    "resolve_card",
    "resolve_relic",
    "resolve_encounter",
    "resolve_potion",
    "encoding",
]
