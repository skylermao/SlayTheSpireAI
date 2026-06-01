"""Rest site phase handler - campfire actions."""

import numpy as np
from typing import Any
from .base import PhaseHandler


class RestSitePhase(PhaseHandler):
    """
    Handles rest site (campfire) decisions.

    Actions (matching C++ GameContext::chooseCampfireOption):
        0: Rest, 1: Smith, 2: Recall, 3: Lift, 4: Toke, 5: Dig, 6: Skip

    Phase-specific observation:
        - Action availability mask (7 dims, binary 0/1)
        - Girya uses remaining (1 dim, normalized by 3)
    """

    REST, SMITH, RECALL, LIFT, TOKE, DIG, SKIP = range(7)

    @property
    def name(self) -> str:
        return "rest_site"

    @property
    def max_actions(self) -> int:
        return 7

    @property
    def phase_observation_size(self) -> int:
        return 8  # 7 action flags + 1 girya uses

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode rest site state (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)

        # Action mask (binary, already 0/1)
        mask = self.get_action_mask(gc)
        obs[0:7] = mask.astype(np.float32)

        # Girya uses normalized to [0, 1]
        obs[7] = self._get_girya_uses(gc) / 3.0

        return obs

    def _has_relic(self, gc: Any, relic_name: str) -> bool:
        import slaythespire as sts
        relic_id = getattr(sts.RelicId, relic_name, None)
        return gc.has_relic(relic_id) if relic_id else False

    def _has_key(self, gc: Any, key_name: str) -> bool:
        import slaythespire as sts
        key_id = getattr(sts.Key, key_name, None)
        return gc.has_key(key_id) if key_id else False

    def _get_girya_uses(self, gc: Any) -> int:
        import slaythespire as sts
        if not self._has_relic(gc, "GIRYA"):
            return 0
        for relic in gc.relics:
            if relic.id == sts.RelicId.GIRYA:
                return 3 - relic.data
        return 0

    def _has_upgradeable_card(self, gc: Any) -> bool:
        try:
            return gc.deck.get_upgradeable_count() > 0
        except:
            return True

    def get_action_mask(self, gc: Any) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        mask[self.REST] = not self._has_relic(gc, "COFFEE_DRIPPER")
        mask[self.SMITH] = not self._has_relic(gc, "FUSION_HAMMER") and self._has_upgradeable_card(gc)
        mask[self.RECALL] = not self._has_key(gc, "RUBY_KEY")
        mask[self.LIFT] = self._has_relic(gc, "GIRYA") and self._get_girya_uses(gc) > 0
        mask[self.TOKE] = self._has_relic(gc, "PEACE_PIPE")
        mask[self.DIG] = self._has_relic(gc, "SHOVEL")
        mask[self.SKIP] = True
        return mask

    def execute_action(self, gc: Any, action: int) -> None:
        if 0 <= action <= self.SKIP:
            gc.choose_campfire_option(action)
        else:
            raise ValueError(f"Invalid rest site action: {action}")
