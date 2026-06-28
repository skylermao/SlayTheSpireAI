"""Scenario construction: turn run-data / names into sim-ready objects.

This module is pure data plumbing -- it knows how to map human/run-data strings to
`slaythespire` enums and bundles a combat's starting conditions (`CombatConfig`) and
reward shaping (`RewardConfig`). `DatasetSampler` draws scenarios from extracted fight
logs. None of this touches a live BattleContext; that is `CombatSession`'s job.
"""

import gzip
import json
import random
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from ._sts import sts


# =============================================================================
# Name -> enum helpers (accept enums directly, or run-data strings)
# =============================================================================

_CARD_NAME_OVERRIDES = {
    "Strike_R": "STRIKE_RED", "Defend_R": "DEFEND_RED",
    "Strike_G": "STRIKE_GREEN", "Defend_G": "DEFEND_GREEN",
    "Strike_B": "STRIKE_BLUE", "Defend_B": "DEFEND_BLUE",
    "Strike_P": "STRIKE_PURPLE", "Defend_P": "DEFEND_PURPLE",
    "AscendersBane": "ASCENDERS_BANE", "Ghostly": "GHOSTLY_ARMOR",
}

_RELIC_NAME_OVERRIDES = {
    "Du-Vu Doll": "DU_VU_DOLL", "StrikeDummy": "STRIKE_DUMMY",
    "SsserpentHead": "SSSERPENT_HEAD", "Lee's Waffle": "LEES_WAFFLE",
    "Snecko Skull": "SNECKO_SKULL", "MealTicket": "MEAL_TICKET",
    "CaptainsWheel": "CAPTAINS_WHEEL", "TungstenRod": "TUNGSTEN_ROD",
    "CloakClasp": "CLOAK_CLASP", "DataDisk": "DATA_DISK", "PureWater": "PURE_WATER",
    "MawBank": "MAW_BANK", "Old Coin": "OLD_COIN", "SlaversCollar": "SLAVERS_COLLAR",
    "TinyHouse": "TINY_HOUSE", "WingedGreaves": "WING_BOOTS", "HolyWater": "HOLY_WATER",
    "FossilizedHelix": "FOSSILIZED_HELIX", "GamblingChip": "GAMBLING_CHIP",
    "TheSpecimen": "THE_SPECIMEN", "BottledFlame": "BOTTLED_FLAME",
    "BottledLightning": "BOTTLED_LIGHTNING", "BottledTornado": "BOTTLED_TORNADO",
    "WristBlade": "WRIST_BLADE", "DarkstonePeriapt": "DARKSTONE_PERIAPT",
    "Courier": "THE_COURIER", "CharonsAshes": "CHARONS_ASHES", "PaperKrane": "PAPER_KRANE",
    "PaperPhrog": "PAPER_PHROG", "Paper Frog": "PAPER_PHROG",
    "ClockworkSouvenir": "CLOCKWORK_SOUVENIR", "BustedCrown": "BUSTED_CROWN",
    "DeadBranch": "DEAD_BRANCH", "RunicCube": "RUNIC_CUBE", "HornCleat": "HORN_CLEAT",
    "InkBottle": "INK_BOTTLE", "CeramicFish": "CERAMIC_FISH", "WarpedTongs": "WARPED_TONGS",
    "StoneCalendar": "STONE_CALENDAR", "PreservedInsect": "PRESERVED_INSECT",
    "FaceOfCleric": "FACE_OF_CLERIC", "Toxic Egg 2": "TOXIC_EGG",
    "Molten Egg 2": "MOLTEN_EGG", "Frozen Egg 2": "FROZEN_EGG",
    "NlothsMask": "NLOTHS_HUNGRY_FACE", "GremlinMask": "GREMLIN_VISAGE",
    "CultistMask": "CULTIST_HEADPIECE", "NeowsBlessing": "NEOWS_LAMENT",
    "Boot": "THE_BOOT", "MutagenicStrength": "MUTAGENIC_STRENGTH",
    "Blue Doll": "BLUE_CANDLE", "LetterFromLob": "LETTER_OPENER",
}

_ENCOUNTER_NAME_OVERRIDES = {
    "Jaw Worm": "JAW_WORM", "Cultist": "CULTIST", "2 Louse": "TWO_LOUSE",
    "3 Louse": "THREE_LOUSE", "Small Slimes": "SMALL_SLIMES", "Blue Slaver": "BLUE_SLAVER",
    "Red Slaver": "RED_SLAVER", "Gremlin Gang": "GREMLIN_GANG", "Looter": "LOOTER",
    "Large Slime": "LARGE_SLIME", "Lots of Slimes": "LOTS_OF_SLIMES",
    "Exordium Thugs": "EXORDIUM_THUGS", "Exordium Wildlife": "EXORDIUM_WILDLIFE",
    "2 Fungi Beasts": "TWO_FUNGI_BEASTS", "Gremlin Nob": "GREMLIN_NOB",
    "Lagavulin": "LAGAVULIN", "3 Sentries": "THREE_SENTRIES", "Slime Boss": "SLIME_BOSS",
    "The Guardian": "THE_GUARDIAN", "Hexaghost": "HEXAGHOST", "Chosen": "CHOSEN",
    "Shell Parasite": "SHELL_PARASITE", "Spheric Guardian": "SPHERIC_GUARDIAN",
    "3 Byrds": "THREE_BYRDS", "Chosen and Byrds": "CHOSEN_AND_BYRDS",
    "Centurion and Healer": "CENTURION_AND_HEALER", "Cultist and Chosen": "CULTIST_AND_CHOSEN",
    "Snake Plant": "SNAKE_PLANT", "Snecko": "SNECKO",
    "Shelled Parasite and Fungi": "SHELLED_PARASITE_AND_FUNGI", "2 Thieves": "TWO_THIEVES",
    "Gremlin Leader": "GREMLIN_LEADER", "Slavers": "SLAVERS",
    "Book of Stabbing": "BOOK_OF_STABBING", "Automaton": "AUTOMATON",
    "Collector": "COLLECTOR", "Champ": "CHAMP", "3 Darklings": "THREE_DARKLINGS",
    "Orb Walker": "ORB_WALKER", "3 Shapes": "THREE_SHAPES", "4 Shapes": "FOUR_SHAPES",
    "Maw": "MAW", "Spire Growth": "SPIRE_GROWTH", "Transient": "TRANSIENT",
    "Writhing Mass": "WRITHING_MASS", "Giant Head": "GIANT_HEAD", "Nemesis": "NEMESIS",
    "Reptomancer": "REPTOMANCER", "3 Cultists": "THREE_CULTIST",
    "Jaw Worm Horde": "JAW_WORM_HORDE", "Awakened One": "AWAKENED_ONE",
    "Time Eater": "TIME_EATER", "Donu and Deca": "DONU_AND_DECA",
    "Spire Shield and Spire Spear": "SHIELD_AND_SPEAR", "The Heart": "THE_HEART",
}


# Encounter difficulty tiers (names exactly as they appear in the fight data "enemies").
# Note: "Blue Slaver"/"Red Slaver" are A1 normal single-slaver fights and are kept; the
# A2 elite trio "Slavers" is excluded.
BOSS_ENCOUNTERS = {
    "The Guardian", "Hexaghost", "Slime Boss", "Automaton", "Collector", "Champ",
    "Awakened One", "Time Eater", "Donu and Deca", "The Heart",
}
ELITE_ENCOUNTERS = {
    "Gremlin Nob", "Lagavulin", "Lagavulin Event", "3 Sentries", "Gremlin Leader",
    "Slavers", "Book of Stabbing", "Giant Head", "Nemesis", "Reptomancer",
    "Shield and Spear",
}
NON_NORMAL_ENCOUNTERS = BOSS_ENCOUNTERS | ELITE_ENCOUNTERS


def _enum_from(name: str, enum_cls, overrides: dict):
    key = overrides.get(name, name.upper().replace(" ", "_").replace("'", "").replace("-", "_"))
    return getattr(enum_cls, key, None)


def resolve_card(spec: Union[str, "sts.Card", "sts.CardId"]) -> Optional["sts.Card"]:
    """Build an `sts.Card` from a Card, a CardId, or a run-data name (with '+1')."""
    if isinstance(spec, sts.Card):
        return spec
    if isinstance(spec, sts.CardId):
        return sts.Card(spec)
    upgraded = isinstance(spec, str) and spec.endswith("+1")
    name = spec[:-2] if upgraded else spec
    card_id = _enum_from(name, sts.CardId, _CARD_NAME_OVERRIDES)
    if card_id is None:
        return None
    card = sts.Card(card_id)
    if upgraded:
        card.upgrade()
    return card


def resolve_relic(spec: Union[str, "sts.RelicId"]) -> Optional["sts.RelicId"]:
    if isinstance(spec, sts.RelicId):
        return spec
    return _enum_from(spec, sts.RelicId, _RELIC_NAME_OVERRIDES)


def resolve_encounter(spec: Union[str, "sts.MonsterEncounter"]) -> Optional["sts.MonsterEncounter"]:
    if isinstance(spec, sts.MonsterEncounter):
        return spec
    return _enum_from(spec, sts.MonsterEncounter, _ENCOUNTER_NAME_OVERRIDES)


def resolve_potion(spec: Union[str, "sts.Potion"]) -> Optional["sts.Potion"]:
    if isinstance(spec, sts.Potion):
        return spec
    return _enum_from(spec, sts.Potion, {})


# =============================================================================
# Potion pool (for randomly equipping potions on dataset combats)
# =============================================================================
# Each character has three class-locked potions; an Ironclad can only ever hold
# its own three plus the class-agnostic (common/uncommon/rare) potions. Drop the
# other three classes' exclusives, plus the INVALID / empty-slot sentinels.
_NON_IRONCLAD_POTIONS = {
    "POISON_POTION", "CUNNING_POTION", "GHOST_IN_A_JAR",          # Silent
    "FOCUS_POTION", "POTION_OF_CAPACITY", "ESSENCE_OF_DARKNESS",  # Defect
    "BOTTLED_MIRACLE", "STANCE_POTION", "AMBROSIA",               # Watcher
}


def _ironclad_potion_pool() -> list:
    pool = []
    for name, pot in sts.Potion.__members__.items():
        if name in ("INVALID", "EMPTY_POTION_SLOT") or name in _NON_IRONCLAD_POTIONS:
            continue
        pool.append(pot)
    return pool


IRONCLAD_POTION_POOL = _ironclad_potion_pool()


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class CombatConfig:
    """A fully-specified combat starting state."""
    deck: Sequence[Union[str, "sts.Card", "sts.CardId"]]
    encounter: Union[str, "sts.MonsterEncounter"]
    max_hp: int
    cur_hp: Optional[int] = None                       # defaults to max_hp
    relics: Sequence[Union[str, "sts.RelicId"]] = ()
    potions: Sequence[Union[str, "sts.Potion"]] = ()
    ascension: int = 0
    seed: Optional[int] = None                         # battle RNG seed; None -> session picks
    human_hp_loss: Optional[int] = None                # optional baseline (from dataset)


@dataclass
class RewardConfig:
    """Shaped per-step reward coefficients (all applied to raw HP units)."""
    damage_dealt_coef: float = 1.0      # reward per point of enemy HP removed
    hp_loss_coef: float = 1.0           # penalty per point of player HP lost
    turn_penalty: float = 0.0           # penalty applied once per END_TURN
    win_bonus: float = 10.0             # terminal bonus on victory
    death_penalty: float = 50.0         # terminal penalty on death
    normalize_by_max_hp: bool = False   # divide HP-based terms by max_hp


# =============================================================================
# Dataset sampler
# =============================================================================

class DatasetSampler:
    """Samples `CombatConfig`s from extracted fight data (data/*.json.gz).

    Each fight record carries the deck, relics, enemy, and entering HP for one combat
    of a real run; `sample()` turns one into a `CombatConfig`. The source data has no
    potions, so potions are drawn here: by default 80% no potion, 15% one, 5% two, each
    chosen uniformly at random from the Ironclad-legal pool (`IRONCLAD_POTION_POOL`).

    Validation: the fight set is large (the Ironclad-wins file is ~1.8M combats), so by
    default mappability is checked lazily -- `sample()` resamples until it hits a fight
    whose deck/relics/enemy all resolve -- instead of filtering the whole list up front.
    Pass `require_mappable=True` to eagerly filter (slow, but yields a clean fixed set).

    NOTE on lost fights: this data is extracted from *winning* runs (~21 fights/run), so the
    human survived every combat -- there is no lost (fatal) fight to exclude here. Dropping
    the final lost fight only applies when extracting from all runs (incl. losses) and
    should use the run's `victory` flag, not an HP heuristic.

    `damage_taken` is net HP lost in the fight (post-block) -- verified against the all-runs
    `current_hp_per_floor`: end-of-floor HP = entering - damage_taken + Burning Blood's
    +6 heal (capped at max). It is NOT a reliable death signal here because ~10% of fights
    (clustered at the boss floors) have damage_taken >= the recorded `entering_hp` yet were
    won -- i.e. this extraction's `entering_hp` is unreliable for some boss/elite records,
    so don't filter on damage_taken >= entering_hp.
    """

    def __init__(self, fights: list[dict], rng: Optional[random.Random] = None,
                 require_mappable: bool = False,
                 potion_distribution: "tuple[float, float, float]" = (0.80, 0.15, 0.05),
                 potion_pool: Optional[list] = None):
        self.rng = rng or random.Random()
        self.fights = [f for f in fights if self._mappable(f)] if require_mappable else fights
        if not self.fights:
            raise ValueError("DatasetSampler: no fights")
        self.potion_distribution = potion_distribution
        self.potion_pool = potion_pool if potion_pool is not None else IRONCLAD_POTION_POOL

    @classmethod
    def from_gzip(cls, path: str, **kw) -> "DatasetSampler":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            fights = json.load(f)
        return cls(fights, **kw)

    @classmethod
    def from_json(cls, path: str, **kw) -> "DatasetSampler":
        with open(path, "r", encoding="utf-8") as f:
            fights = json.load(f)
        return cls(fights, **kw)

    @staticmethod
    def _mappable(f: dict) -> bool:
        if resolve_encounter(f["enemies"]) is None:
            return False
        if any(resolve_card(c) is None for c in f["cards"]):
            return False
        if any(resolve_relic(r) is None for r in f["relics"]):
            return False
        return True

    def _sample_potions(self) -> list:
        """Roll a potion count from `potion_distribution`, then draw that many at random
        (with replacement -- duplicates are legal to hold) from the Ironclad pool."""
        r = self.rng.random()
        p0, p1, _ = self.potion_distribution
        n = 0 if r < p0 else (1 if r < p0 + p1 else 2)
        return [self.rng.choice(self.potion_pool) for _ in range(n)]

    def sample(self) -> CombatConfig:
        # Lazy mappability: resample past any fight whose deck/relics/enemy don't resolve.
        for _ in range(64):
            f = self.rng.choice(self.fights)
            if self._mappable(f):
                break
        else:
            raise RuntimeError("DatasetSampler: no mappable fight found in 64 tries")
        return CombatConfig(
            deck=list(f["cards"]),
            relics=list(f["relics"]),
            max_hp=int(f["max_hp"]),
            cur_hp=int(f["entering_hp"]),
            encounter=f["enemies"],
            potions=self._sample_potions(),
            ascension=int(f.get("ascension", 0)),
            human_hp_loss=f.get("damage_taken"),
        )
