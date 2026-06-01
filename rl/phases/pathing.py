"""Pathing phase handler - map navigation."""

import numpy as np
from typing import Any
from .base import PhaseHandler, Normalize


class PathingPhase(PhaseHandler):
    """
    Handles map navigation decisions.

    Phase-specific observation:
        - Current position (x, y) normalized by map dimensions
        - Available next nodes with room types (one-hot)
        - 3-floor lookahead room types (one-hot)

    Action:
        - Choose next node (0-6 for x position)
    """

    MAP_WIDTH = 7
    MAP_HEIGHT = 15
    NUM_ROOM_TYPES = 10

    @property
    def name(self) -> str:
        return "pathing"

    @property
    def max_actions(self) -> int:
        return self.MAP_WIDTH

    @property
    def phase_observation_size(self) -> int:
        # Position (2) + next floor rooms (70) + 3-floor lookahead (210)
        return 2 + (self.MAP_WIDTH * self.NUM_ROOM_TYPES) * 4

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode map position and paths (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)
        offset = 0

        # Position normalized to [0, 1]
        obs[offset] = gc.cur_map_node_x / Normalize.MAP_WIDTH
        obs[offset + 1] = gc.cur_map_node_y / Normalize.MAP_HEIGHT
        offset += 2

        next_y = gc.cur_map_node_y + 1
        spire_map = gc.map

        # Next floor room types (one-hot, already 0/1)
        for x in range(self.MAP_WIDTH):
            if self._has_edge(gc, x):
                room_type = self._get_room_type(spire_map, x, next_y)
                if 0 <= room_type < self.NUM_ROOM_TYPES:
                    obs[offset + x * self.NUM_ROOM_TYPES + room_type] = 1.0
        offset += self.MAP_WIDTH * self.NUM_ROOM_TYPES

        # 3-floor lookahead (one-hot, already 0/1)
        for lookahead in range(3):
            y = next_y + lookahead + 1
            if y < self.MAP_HEIGHT:
                for x in range(self.MAP_WIDTH):
                    room_type = self._get_room_type(spire_map, x, y)
                    if 0 <= room_type < self.NUM_ROOM_TYPES:
                        obs[offset + x * self.NUM_ROOM_TYPES + room_type] = 1.0
            offset += self.MAP_WIDTH * self.NUM_ROOM_TYPES

        return obs

    def _has_edge(self, gc: Any, target_x: int) -> bool:
        import slaythespire as sts
        return sts.has_edge(gc.map, gc.cur_map_node_x, gc.cur_map_node_y, target_x)

    def _get_room_type(self, spire_map: Any, x: int, y: int) -> int:
        import slaythespire as sts
        return int(sts.get_room_type(spire_map, x, y))

    def get_action_mask(self, gc: Any) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        for x in range(self.MAP_WIDTH):
            if self._has_edge(gc, x):
                mask[x] = True
        return mask

    def execute_action(self, gc: Any, action: int) -> None:
        if 0 <= action < self.MAP_WIDTH:
            gc.transition_to_map_node(action)
        else:
            raise ValueError(f"Invalid pathing action: {action}")
