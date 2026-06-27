"""Core combat environment: the sim interface, observation encoding, and scenario data.

This layer talks to the C++ simulator and owns the MDP-style interface (`CombatSession`),
the observation encoding, and the scenario/dataset plumbing. It has no dependency on the
learning algorithm, training, or tuning layers.
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
    NON_NORMAL_ENCOUNTERS,
    ELITE_ENCOUNTERS,
    BOSS_ENCOUNTERS,
    resolve_card,
    resolve_relic,
    resolve_encounter,
    resolve_potion,
)
from . import encoding

__all__ = [
    "CombatSession", "LegalAction", "Transition",
    "END_TURN", "PLAY_CARD", "USE_POTION", "SELECT_CARD", "SKIP_SELECT",
    "CombatConfig", "RewardConfig", "DatasetSampler",
    "NON_NORMAL_ENCOUNTERS", "ELITE_ENCOUNTERS", "BOSS_ENCOUNTERS",
    "resolve_card", "resolve_relic", "resolve_encounter", "resolve_potion",
    "encoding",
]
