"""Event phase handler - event choices."""

import numpy as np
from typing import Any, Dict
from .base import PhaseHandler


class EventPhase(PhaseHandler):
    """
    Handles event choice selection.

    Phase-specific observation:
        - Event type one-hot (57 dims, already 0/1)
        - Number of options normalized by 4

    Action:
        - Choose option 0-3
    """

    NUM_EVENT_TYPES = 57

    EVENT_OPTION_COUNTS: Dict[int, int] = {
        6: 2, 7: 3, 8: 2, 9: 2, 10: 3, 11: 1, 12: 2, 13: 3, 14: 3, 15: 4,
        16: 2, 17: 2, 18: 2, 19: 3, 20: 3, 21: 2, 22: 2, 23: 3, 24: 2, 25: 2,
        26: 4, 27: 2, 28: 2, 29: 3, 30: 2, 31: 2, 32: 3, 33: 3, 34: 2, 35: 2,
        36: 3, 37: 2, 38: 2, 39: 2, 40: 2, 41: 2, 42: 2, 43: 3, 44: 2, 45: 2,
        46: 2, 47: 3, 48: 3, 49: 3, 50: 2, 51: 2, 52: 2, 53: 3, 54: 2, 55: 3, 56: 2,
    }

    @property
    def name(self) -> str:
        return "event"

    @property
    def max_actions(self) -> int:
        return 4

    @property
    def phase_observation_size(self) -> int:
        return self.NUM_EVENT_TYPES + 1

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode event state (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)

        # Event type one-hot (already 0/1)
        event_id = int(gc.cur_event)
        if 0 <= event_id < self.NUM_EVENT_TYPES:
            obs[event_id] = 1.0

        # Num options normalized to [0, 1]
        obs[self.NUM_EVENT_TYPES] = self._get_num_options(gc) / 4.0

        return obs

    def _get_num_options(self, gc: Any) -> int:
        return self.EVENT_OPTION_COUNTS.get(int(gc.cur_event), 2)

    def get_action_mask(self, gc: Any) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        for i in range(min(self._get_num_options(gc), self.max_actions)):
            mask[i] = True
        return mask

    def execute_action(self, gc: Any, action: int) -> None:
        if 0 <= action < self.max_actions:
            gc.choose_event_option(action)
        else:
            raise ValueError(f"Invalid event action: {action}")
