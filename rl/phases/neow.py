"""Neow phase handler - starting bonus selection."""

import numpy as np
from typing import Any
from .base import PhaseHandler


class NeowPhase(PhaseHandler):
    """
    Handles Neow's starting bonus selection.

    Phase-specific observation:
        - 4 options, each encoded as (bonus_type, drawback_type)
        - 19 bonus types, 7 drawback types (one-hot encoded)
        - All values already 0/1 (one-hot)

    Action:
        - Choose option 0-3
    """

    NUM_BONUSES = 19
    NUM_DRAWBACKS = 7

    @property
    def name(self) -> str:
        return "neow"

    @property
    def max_actions(self) -> int:
        return 4

    @property
    def phase_observation_size(self) -> int:
        return 4 * (self.NUM_BONUSES + self.NUM_DRAWBACKS)

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode the 4 Neow options (one-hot, already normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)

        neow_options = gc.neow_rewards
        option_size = self.NUM_BONUSES + self.NUM_DRAWBACKS

        for i, option in enumerate(neow_options):
            offset = i * option_size
            bonus_idx = int(option.bonus)
            if 0 <= bonus_idx < self.NUM_BONUSES:
                obs[offset + bonus_idx] = 1.0

            drawback_idx = int(option.drawback)
            if 0 <= drawback_idx < self.NUM_DRAWBACKS:
                obs[offset + self.NUM_BONUSES + drawback_idx] = 1.0

        return obs

    def get_action_mask(self, gc: Any) -> np.ndarray:
        return np.ones(self.max_actions, dtype=bool)

    def execute_action(self, gc: Any, action: int) -> None:
        if 0 <= action < 4:
            gc.choose_neow_option(action)
        else:
            raise ValueError(f"Invalid Neow action: {action}")
