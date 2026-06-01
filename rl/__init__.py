"""
RL module for Slay the Spire training.

Phases:
- neow: Starting bonus selection
- pathing: Map navigation
- combat: Battle actions (TBD)
- card_reward: Post-combat card selection
- rest_site: Campfire actions
- event: Event choices
- shop: Purchase decisions
- boss_relic: Post-boss relic selection
- card_select: Sub-phase for multi-step card selections
"""

from .phases import (
    NeowPhase,
    PathingPhase,
    CombatPhase,
    CardRewardPhase,
    RestSitePhase,
    EventPhase,
    ShopPhase,
    BossRelicPhase,
    CardSelectPhase,
    CardSelectContext,
)
