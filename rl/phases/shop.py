"""Shop phase handler - purchase decisions."""

import numpy as np
from typing import Any
from .base import PhaseHandler, Normalize


class ShopPhase(PhaseHandler):
    """
    Handles shop purchase decisions.

    Phase-specific observation:
        - Card prices (7 dims, normalized by 500)
        - Relic prices (3 dims, normalized by 500)
        - Potion prices (3 dims, normalized by 200)
        - Remove price (1 dim, normalized by 200)
        - Availability flags (15 dims, binary 0/1)

    Action:
        0: Leave, 1-7: Buy cards, 8-10: Buy relics, 11-13: Buy potions, 14: Remove
    """

    LEAVE = 0
    CARD_START, CARD_END = 1, 7
    RELIC_START, RELIC_END = 8, 10
    POTION_START, POTION_END = 11, 13
    REMOVE_CARD = 14

    @property
    def name(self) -> str:
        return "shop"

    @property
    def max_actions(self) -> int:
        return 15

    @property
    def phase_observation_size(self) -> int:
        return 14 + 15  # prices + availability

    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """Encode shop state (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)
        offset = 0
        shop = gc.shop

        # Card prices normalized
        for i in range(7):
            obs[offset + i] = shop.card_price(i) / Normalize.PRICE
        offset += 7

        # Relic prices normalized
        for i in range(3):
            obs[offset + i] = shop.relic_price(i) / Normalize.PRICE
        offset += 3

        # Potion prices normalized
        for i in range(3):
            obs[offset + i] = shop.potion_price(i) / Normalize.POTION_PRICE
        offset += 3

        # Remove price normalized
        obs[offset] = shop.remove_cost / Normalize.POTION_PRICE
        offset += 1

        # Availability (binary, already 0/1)
        obs[offset:offset + 15] = self.get_action_mask(gc).astype(np.float32)

        return obs

    def get_action_mask(self, gc: Any) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        mask[self.LEAVE] = True

        shop = gc.shop
        gold = gc.gold

        for i in range(7):
            price = shop.card_price(i)
            if price > 0 and gold >= price:
                mask[self.CARD_START + i] = True

        for i in range(3):
            price = shop.relic_price(i)
            if price > 0 and gold >= price:
                mask[self.RELIC_START + i] = True

        has_slot = self._has_potion_slot(gc)
        for i in range(3):
            price = shop.potion_price(i)
            if price > 0 and gold >= price and has_slot:
                mask[self.POTION_START + i] = True

        if gold >= shop.remove_cost and self._has_removable_card(gc):
            mask[self.REMOVE_CARD] = True

        return mask

    def _has_potion_slot(self, gc: Any) -> bool:
        try:
            return any(gc.potions[i] is None or gc.potions[i] == 0 for i in range(3))
        except:
            return True

    def _has_removable_card(self, gc: Any) -> bool:
        import slaythespire as sts
        return any(
            (card.id if hasattr(card, 'id') else card) != sts.CardId.ASCENDERS_BANE
            for card in gc.deck
        )

    def execute_action(self, gc: Any, action: int) -> None:
        import slaythespire as sts
        shop = gc.shop

        if action == self.LEAVE:
            gc.screen_state = sts.ScreenState.MAP_SCREEN
        elif self.CARD_START <= action <= self.CARD_END:
            shop.buy_card(gc, action - self.CARD_START)
        elif self.RELIC_START <= action <= self.RELIC_END:
            shop.buy_relic(gc, action - self.RELIC_START)
        elif self.POTION_START <= action <= self.POTION_END:
            shop.buy_potion(gc, action - self.POTION_START)
        elif action == self.REMOVE_CARD:
            shop.buy_card_remove(gc)
        else:
            raise ValueError(f"Invalid shop action: {action}")
