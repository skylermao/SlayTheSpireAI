"""Card selection sub-phase handler - for multi-step actions."""

import numpy as np
from typing import Any, Optional
from enum import IntEnum
from .base import PhaseHandler, Normalize


class CardSelectContext(IntEnum):
    NONE = 0
    COMBAT = 1
    OUT_OF_COMBAT = 2


class CardSelectPhase(PhaseHandler):
    """
    Handles card selection sub-phases (combat and out-of-combat).

    Phase-specific observation:
        - Context one-hot (2 dims, 0/1)
        - Task type one-hot (21 dims, 0/1)
        - Can skip (1 dim, 0/1)
        - Pick count (1 dim, normalized by 10)
        - Cards: MAX × (card_id/400 + upgraded 0/1 + available 0/1)

    Action:
        0-29: Select card, 30: Skip
    """

    MAX_SELECTABLE_CARDS = 30
    NUM_COMBAT_TASKS = 21
    NUM_SCREEN_TYPES = 9

    @property
    def name(self) -> str:
        return "card_select"

    @property
    def max_actions(self) -> int:
        return self.MAX_SELECTABLE_CARDS + 1

    @property
    def phase_observation_size(self) -> int:
        task_size = max(self.NUM_COMBAT_TASKS, self.NUM_SCREEN_TYPES)
        return 2 + task_size + 2 + (self.MAX_SELECTABLE_CARDS * 3)

    def get_context(self, gc: Any, bc: Optional[Any] = None) -> CardSelectContext:
        import slaythespire as sts
        if bc is not None and bc.input_state == sts.InputState.CARD_SELECT:
            return CardSelectContext.COMBAT
        if gc.screen_state == sts.ScreenState.CARD_SELECT:
            return CardSelectContext.OUT_OF_COMBAT
        return CardSelectContext.NONE

    def get_phase_observation(self, gc: Any, bc: Optional[Any] = None) -> np.ndarray:
        """Build phase-specific observation (normalized)."""
        obs = np.zeros(self.phase_observation_size, dtype=np.float32)
        offset = 0

        context = self.get_context(gc, bc)

        # Context one-hot (already 0/1)
        if context == CardSelectContext.COMBAT:
            obs[0] = 1.0
        elif context == CardSelectContext.OUT_OF_COMBAT:
            obs[1] = 1.0
        offset += 2

        # Task type one-hot (already 0/1)
        task_size = max(self.NUM_COMBAT_TASKS, self.NUM_SCREEN_TYPES)
        if context == CardSelectContext.COMBAT and bc is not None:
            task_idx = int(bc.card_select_info.card_select_task)
            if 0 <= task_idx < self.NUM_COMBAT_TASKS:
                obs[offset + task_idx] = 1.0
        elif context == CardSelectContext.OUT_OF_COMBAT:
            screen_idx = int(gc.card_select_screen_type)
            if 0 <= screen_idx < self.NUM_SCREEN_TYPES:
                obs[offset + screen_idx] = 1.0
        offset += task_size

        # Skip and pick count
        can_skip, pick_count, cards = self._get_selection_info(gc, bc, context)
        obs[offset] = 1.0 if can_skip else 0.0
        obs[offset + 1] = min(pick_count / 10.0, 1.0)
        offset += 2

        # Cards (normalized)
        for i, card in enumerate(cards[:self.MAX_SELECTABLE_CARDS]):
            obs[offset + i * 3] = card['id'] / Normalize.CARD_ID
            obs[offset + i * 3 + 1] = 1.0 if card['upgraded'] else 0.0
            obs[offset + i * 3 + 2] = 1.0

        return obs

    def _get_selection_info(self, gc: Any, bc: Optional[Any], context: CardSelectContext) -> tuple:
        can_skip, pick_count, cards = False, 1, []

        if context == CardSelectContext.COMBAT and bc is not None:
            info = bc.card_select_info
            can_skip = info.can_pick_zero
            pick_count = info.pick_count
            cards = self._get_combat_cards(bc)
        elif context == CardSelectContext.OUT_OF_COMBAT:
            pick_count = gc.to_select_count
            cards = [{'id': int(c.id), 'upgraded': c.upgraded} for c in gc.to_select_cards]

        return can_skip, pick_count, cards

    def _get_combat_cards(self, bc: Any) -> list:
        import slaythespire as sts
        task = bc.card_select_info.card_select_task

        if task in [sts.CardSelectTask.ARMAMENTS, sts.CardSelectTask.DUAL_WIELD,
                    sts.CardSelectTask.FORETHOUGHT, sts.CardSelectTask.WARCRY,
                    sts.CardSelectTask.SETUP, sts.CardSelectTask.NIGHTMARE,
                    sts.CardSelectTask.GAMBLE]:
            source = bc.cards.hand
        elif task in [sts.CardSelectTask.HEADBUTT, sts.CardSelectTask.HOLOGRAM]:
            source = bc.cards.discard_pile
        elif task == sts.CardSelectTask.EXHUME:
            source = bc.cards.exhaust_pile
        elif task in [sts.CardSelectTask.SEEK, sts.CardSelectTask.SECRET_WEAPON,
                      sts.CardSelectTask.SECRET_TECHNIQUE]:
            source = bc.cards.draw_pile
        else:
            source = []

        return [{'id': int(c.id), 'upgraded': c.upgraded} for c in source]

    def get_action_mask(self, gc: Any, bc: Optional[Any] = None) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=bool)
        context = self.get_context(gc, bc)
        can_skip, _, cards = self._get_selection_info(gc, bc, context)

        for i in range(min(len(cards), self.MAX_SELECTABLE_CARDS)):
            mask[i] = True
        if can_skip:
            mask[self.MAX_SELECTABLE_CARDS] = True

        return mask

    def execute_action(self, gc: Any, action: int, bc: Optional[Any] = None) -> None:
        import slaythespire as sts
        context = self.get_context(gc, bc)

        if action == self.MAX_SELECTABLE_CARDS:
            if context == CardSelectContext.COMBAT and bc is not None:
                sts.Action(sts.ActionType.SINGLE_CARD_SELECT, -1).execute(bc)
            return

        if context == CardSelectContext.COMBAT and bc is not None:
            sts.Action(sts.ActionType.SINGLE_CARD_SELECT, action).execute(bc)
        elif context == CardSelectContext.OUT_OF_COMBAT:
            gc.choose_select_card_screen_option(action)

    def is_active(self, gc: Any, bc: Optional[Any] = None) -> bool:
        return self.get_context(gc, bc) != CardSelectContext.NONE

    def get_observation(self, gc: Any, bc: Optional[Any] = None) -> np.ndarray:
        nn_obs = self.get_nn_observation(gc)
        phase_obs = self.get_phase_observation(gc, bc)
        return np.concatenate([nn_obs, phase_obs])
