"""Boss relic phase handler - post-boss relic selection."""

import numpy as np
from typing import Any
from .base import PhaseHandler, Normalize


class BossRelicPhase(PhaseHandler):
    """
    Handles boss relic selection after defeating act boss.

    Phase-specific observation:
        - 3 relic IDs normalized by 200

    Action:
        0-2: Pick relic
    """

    @property
    def name(self) -> str:
        return "boss_relic"

    @property
    def max_actions(self) -> int:
        return 3

    @property
    def phase_observation_size(self) -> int:
        return 3

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode boss relic options (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)
        for i, relic_id in enumerate(gc.boss_relics[:3]):
            obs[i] = int(relic_id) / Normalize.RELIC_ID
        return obs

    def get_action_mask(self, gc: Any) -> np.ndarray:
        return np.ones(self.max_actions, dtype=bool)

    def execute_action(self, gc: Any, action: int) -> None:
        if 0 <= action < 3:
            gc.choose_boss_relic(action)
        else:
            raise ValueError(f"Invalid boss relic action: {action}")
