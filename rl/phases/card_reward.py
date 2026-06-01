"""Card reward phase handler - post-combat card selection."""

import numpy as np
from typing import Any
from .base import PhaseHandler, Normalize


class CardRewardPhase(PhaseHandler):
    """
    Handles card reward selection after combat.

    Phase-specific observation:
        - 3 cards × (card_id normalized by 400 + upgraded flag 0/1)

    Action:
        0: Skip, 1-3: Pick card
    """

    @property
    def name(self) -> str:
        return "card_reward"

    @property
    def max_actions(self) -> int:
        return 4

    @property
    def phase_observation_size(self) -> int:
        return 6  # 3 cards × 2

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode card rewards (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)

        for i, card in enumerate(gc.get_card_reward()[:3]):
            obs[i * 2] = int(card.id) / Normalize.CARD_ID
            obs[i * 2 + 1] = 1.0 if card.upgraded else 0.0

        return obs

    def get_action_mask(self, gc: Any) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        mask[0] = True  # Skip always valid
        for i in range(min(len(gc.get_card_reward()), 3)):
            mask[i + 1] = True
        return mask

    def execute_action(self, gc: Any, action: int) -> None:
        if action == 0:
            gc.skip_reward_cards()
        elif 1 <= action <= 3:
            rewards = gc.get_card_reward()
            if action - 1 < len(rewards):
                gc.pick_reward_card(rewards[action - 1])
            else:
                raise ValueError(f"Card index {action - 1} out of range")
        else:
            raise ValueError(f"Invalid card reward action: {action}")
