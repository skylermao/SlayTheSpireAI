"""Base class for phase handlers."""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


# NNInterface observation layout (412 dims total):
# [0]: cur_hp (max 200)
# [1]: max_hp (max 200)
# [2]: gold (max 1800)
# [3]: floor_num (max 60)
# [4-13]: boss one-hot (10 dims)
# [14-233]: cards in deck (220 dims, count per card type, max 7)
# [234-411]: relics owned (178 dims, binary)
NN_INTERFACE_SIZE = 412


# Normalization constants (consistent across all phases)
class Normalize:
    """Consistent normalization constants for all observations."""
    HP = 200.0
    GOLD = 1800.0
    FLOOR = 60.0
    CARD_COUNT = 7.0
    CARD_ID = 400.0      # Max card ID
    RELIC_ID = 200.0     # Max relic ID
    PRICE = 500.0        # Max item price (cards, relics)
    POTION_PRICE = 200.0 # Max potion price
    MAP_WIDTH = 6.0      # Max x index (0-6)
    MAP_HEIGHT = 14.0    # Max y index (0-14)


class PhaseHandler(ABC):
    """
    Abstract base class for game phase handlers.

    Each phase handler is responsible for:
    1. Building observations for its phase (NNInterface + phase-specific)
    2. Defining the action space for its phase
    3. Computing valid action masks
    4. Executing actions

    All observations are normalized to [0, 1] range for consistent training.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this phase."""
        pass

    @property
    @abstractmethod
    def max_actions(self) -> int:
        """Maximum number of possible actions in this phase."""
        pass

    @property
    @abstractmethod
    def phase_observation_size(self) -> int:
        """Size of phase-specific observation (excluding NNInterface)."""
        pass

    @property
    def observation_size(self) -> int:
        """Total observation size including NNInterface context."""
        return NN_INTERFACE_SIZE + self.phase_observation_size

    def get_nn_observation(self, gc: Any) -> np.ndarray:
        """
        Get NNInterface observation, normalized to [0, 1].

        Normalization scheme:
        - cur_hp, max_hp: / 200
        - gold: / 1800
        - floor_num: / 60
        - boss one-hot: already 0/1
        - card counts: / 7
        - relics: already 0/1

        Args:
            gc: GameContext from slaythespire module

        Returns:
            Normalized observation array (412 dims)
        """
        import slaythespire as sts

        nn = sts.getNNInterface()
        raw_obs = nn.getObservation(gc)

        obs = np.array(raw_obs, dtype=np.float32)

        # Normalize non-binary values
        obs[0] /= Normalize.HP          # cur_hp
        obs[1] /= Normalize.HP          # max_hp
        obs[2] /= Normalize.GOLD        # gold
        obs[3] /= Normalize.FLOOR       # floor_num
        # [4-13] boss one-hot: already 0/1
        obs[14:234] /= Normalize.CARD_COUNT  # card counts
        # [234-411] relics: already 0/1

        return obs

    def get_observation(self, gc: Any) -> np.ndarray:
        """
        Build full observation array for this phase.

        Concatenates normalized NNInterface context with phase-specific observation.

        Args:
            gc: GameContext from slaythespire module

        Returns:
            Full observation array for this phase (all values in [0, 1])
        """
        nn_obs = self.get_nn_observation(gc)
        phase_obs = self.get_phase_observation(gc)
        return np.concatenate([nn_obs, phase_obs])

    @abstractmethod
    def get_phase_observation(self, gc: Any) -> np.ndarray:
        """
        Build phase-specific observation array.

        All values should be normalized to [0, 1] using constants from Normalize class.

        Args:
            gc: GameContext from slaythespire module

        Returns:
            Phase-specific observation array (normalized)
        """
        pass

    @abstractmethod
    def get_action_mask(self, gc: Any) -> np.ndarray:
        """
        Get mask of valid actions.

        Args:
            gc: GameContext from slaythespire module

        Returns:
            Boolean array where True = valid action
        """
        pass

    @abstractmethod
    def execute_action(self, gc: Any, action: int) -> None:
        """
        Execute the chosen action.

        Args:
            gc: GameContext from slaythespire module
            action: Action index to execute
        """
        pass

    def get_valid_actions(self, gc: Any) -> list[int]:
        """Get list of valid action indices."""
        mask = self.get_action_mask(gc)
        return [i for i, valid in enumerate(mask) if valid]
