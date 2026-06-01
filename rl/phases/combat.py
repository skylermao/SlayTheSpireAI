"""Combat phase handler - battle actions (TBD)."""

import numpy as np
from typing import Any
from .base import PhaseHandler


class CombatPhase(PhaseHandler):
    """
    Handles combat decisions.

    This is the most complex phase with:
    - Variable hand size (0-10+ cards)
    - Variable enemies (1-5)
    - Status effects on player and enemies
    - Energy management
    - Potion usage
    - Card targeting

    Observation (TBD):
        - Player state: HP, block, energy, status effects
        - Hand: cards available, costs, types
        - Enemies: HP, block, intent, status effects
        - Draw pile size, discard pile size
        - Orbs (for Defect, not applicable for Ironclad)

    Action space options:
        1. Flat action space with masking
           - All possible (card, target) pairs + end turn + potions
           - Use MaskablePPO

        2. Hierarchical actions
           - First: choose action type (play card, use potion, end turn)
           - Then: choose specific card/target

        3. Delegate to MCTS
           - Use RL for other phases, MCTS for combat
           - Simplest approach that works

    Current implementation: Placeholder that delegates to MCTS agent.
    """

    @property
    def name(self) -> str:
        return "combat"

    @property
    def max_actions(self) -> int:
        # Placeholder - actual combat has variable action space
        # Max hand (10) × max targets (5) + potions (3) × targets (5) + end turn
        return 10 * 5 + 3 * 5 + 1  # = 66

    @property
    def observation_size(self) -> int:
        # Placeholder - needs full battle state encoding
        # Player: HP, block, energy, ~20 status effects
        # Hand: 10 cards × (id, cost, type, upgraded)
        # Enemies: 5 × (HP, block, intent, ~15 status effects)
        # Piles: draw size, discard size
        return 500  # Rough estimate

    def get_observation(self, gc: Any) -> np.ndarray:
        """
        Build combat observation.

        TODO: Implement full battle state encoding.
        """
        raise NotImplementedError(
            "Combat observation not yet implemented. "
            "Consider using MCTS agent for combat decisions."
        )

    def get_action_mask(self, gc: Any) -> np.ndarray:
        """
        Get valid combat actions.

        TODO: Implement action enumeration from BattleContext.
        """
        raise NotImplementedError(
            "Combat action mask not yet implemented. "
            "Consider using MCTS agent for combat decisions."
        )

    def execute_action(self, gc: Any, action: int) -> None:
        """
        Execute combat action.

        TODO: Implement action execution on BattleContext.
        """
        raise NotImplementedError(
            "Combat execution not yet implemented. "
            "Consider using MCTS agent for combat decisions."
        )

    def delegate_to_mcts(self, gc: Any) -> None:
        """
        Delegate entire combat to MCTS agent.

        This is the simplest approach - let MCTS handle combat
        while RL handles strategic decisions (pathing, cards, etc.)
        """
        import slaythespire as sts

        agent = sts.Agent()
        agent.playout(gc)
