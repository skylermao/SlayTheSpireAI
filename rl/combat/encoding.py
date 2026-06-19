"""State encoding: BattleContext -> network-ready tensors.

`encode_observation(bc)` is a *pure read* of a live (or cloned/afterstate)
BattleContext. It never mutates the sim, so it works equally on the current state and
on hypothetical afterstates. The layout targets a permutation-invariant set/graph
encoder, not a position-MLP.

Observation keys (all fixed-shape):
    hand_cards / hand_mask          per-card features for cards in hand (<= MAX_HAND)
    draw_pile/discard_pile/exhaust  card-index count vectors (size NUM_CARDS)
    player                          HP/block/energy + status stacks
    monsters / monster_mask         per-enemy features incl. intent + status stacks
    potions                         count vector over potion ids (held potions)
    scalars                         turn, phase flags, potion/skip availability

CARD_SELECT candidates are deliberately NOT in the observation: they are an unbounded
set (a draw/discard select can be the whole deck), so padding them into a fixed array
is wrong. They are exposed variable-length via `select_candidate_features()` and
chosen through the variable-length action set (see session.legal_actions).
"""

import numpy as np

from ._sts import sts


# =============================================================================
# Sizing / normalization constants
# =============================================================================

MAX_HAND = 10          # game hard-caps hand at 10
MAX_ENEMIES = 5        # widest encounter slot count
MAX_POTIONS = 5        # potion belt capacity ceiling
NUM_POTIONS = max(int(p) for p in sts.Potion.__members__.values()) + 1  # potion vocab (raw id space)
# Note: there is no MAX_SELECT. CARD_SELECT candidates are an unbounded set (a
# draw/discard select can be the whole deck), so they are NOT padded into the fixed
# observation. They live in the variable-length action set (`legal_actions()`) and
# are featurized on demand via `select_candidate_features()`.

# Per-card feature layout (hand and select candidates share it).
CARD_FEATURES = 8
# [0]=card index (dense, for embedding) [1]=cost_for_turn [2]=upgraded [3]=ethereal
# [4]=exhausts [5]=requires_target [6]=type_int [7]=special_data (normalized)
#   special_data: Searing Blow upgrade level / Rampage & Ritual Dagger bonus damage /
#   Genetic Algorithm block; 0 for every other card.

# Normalizers (keep observation roughly in [-2, 2]); raw card ids are left unscaled
# because they are meant to index a learned embedding table.
_HP_NORM = 200.0
_BLOCK_NORM = 50.0
_ENERGY_NORM = 10.0
_DMG_NORM = 50.0
_STACK_NORM = 20.0
_COUNT_NORM = 10.0   # per-turn play counts and relic counters are small integers

# Curated status sets: the buffs/debuffs that matter for Ironclad combat. Order is
# fixed and defines the observation layout.
PLAYER_STATUSES = [
    sts.PlayerStatus.STRENGTH, sts.PlayerStatus.DEXTERITY, sts.PlayerStatus.METALLICIZE,
    sts.PlayerStatus.PLATED_ARMOR, sts.PlayerStatus.THORNS, sts.PlayerStatus.RITUAL,
    sts.PlayerStatus.RAGE, sts.PlayerStatus.COMBUST, sts.PlayerStatus.FIRE_BREATHING,
    sts.PlayerStatus.RUPTURE, sts.PlayerStatus.VIGOR, sts.PlayerStatus.REGEN,
    sts.PlayerStatus.BUFFER, sts.PlayerStatus.DOUBLE_DAMAGE, sts.PlayerStatus.DOUBLE_TAP,
    sts.PlayerStatus.FLAME_BARRIER, sts.PlayerStatus.NEXT_TURN_BLOCK,
    sts.PlayerStatus.DRAW_CARD_NEXT_TURN, sts.PlayerStatus.ENERGIZED,
    sts.PlayerStatus.BARRICADE, sts.PlayerStatus.DEMON_FORM, sts.PlayerStatus.CORRUPTION,
    sts.PlayerStatus.BRUTALITY, sts.PlayerStatus.DARK_EMBRACE, sts.PlayerStatus.EVOLVE,
    sts.PlayerStatus.FEEL_NO_PAIN, sts.PlayerStatus.JUGGERNAUT,
    sts.PlayerStatus.VULNERABLE, sts.PlayerStatus.WEAK, sts.PlayerStatus.FRAIL,
    sts.PlayerStatus.ENTANGLED, sts.PlayerStatus.NO_DRAW, sts.PlayerStatus.CONFUSED,
    sts.PlayerStatus.DRAW_REDUCTION, sts.PlayerStatus.HEX, sts.PlayerStatus.CONSTRICTED,
]

MONSTER_STATUSES = [
    sts.MonsterStatus.STRENGTH, sts.MonsterStatus.ARTIFACT, sts.MonsterStatus.PLATED_ARMOR,
    sts.MonsterStatus.THORNS, sts.MonsterStatus.RITUAL, sts.MonsterStatus.ANGRY,
    sts.MonsterStatus.CURIOSITY, sts.MonsterStatus.METALLICIZE, sts.MonsterStatus.POISON,
    sts.MonsterStatus.REGEN, sts.MonsterStatus.GENERIC_STRENGTH_UP,
    sts.MonsterStatus.MODE_SHIFT, sts.MonsterStatus.TIME_WARP, sts.MonsterStatus.MINION,
    sts.MonsterStatus.MINION_LEADER, sts.MonsterStatus.INTANGIBLE,
    sts.MonsterStatus.INVINCIBLE, sts.MonsterStatus.BARRICADE, sts.MonsterStatus.FLIGHT,
    sts.MonsterStatus.CURL_UP, sts.MonsterStatus.ASLEEP, sts.MonsterStatus.SHIFTING,
    sts.MonsterStatus.REACTIVE, sts.MonsterStatus.SHARP_HIDE, sts.MonsterStatus.VULNERABLE,
    sts.MonsterStatus.WEAK, sts.MonsterStatus.CHOKED, sts.MonsterStatus.LOCK_ON,
    sts.MonsterStatus.MARK, sts.MonsterStatus.SHACKLED, sts.MonsterStatus.CORPSE_EXPLOSION,
]

# cur_hp, max_hp, block, energy, energy_per_turn,
# + per-turn play counts (cards/attacks/skills played, cards discarded),
# + persistent relic counters (happy_flower, incense_burner, ink_bottle, inserter,
#   nunchaku, pen_nib, sundial) -- progress toward each counter relic's next trigger.
PLAYER_BASE_DIM = 16
PLAYER_DIM = PLAYER_BASE_DIM + len(PLAYER_STATUSES)
MONSTER_BASE_DIM = 8  # cur_hp, max_hp, block, intent_dmg, intent_hits, attacking, alive, targetable
MONSTER_DIM = MONSTER_BASE_DIM + len(MONSTER_STATUSES)
SCALAR_DIM = 5  # turn, is_card_select, can_skip_select, num_potions, num_alive_enemies

_CARD_TYPE_INT = {
    sts.CardType.ATTACK: 0, sts.CardType.SKILL: 1, sts.CardType.POWER: 2,
    sts.CardType.STATUS: 3, sts.CardType.CURSE: 4, sts.CardType.INVALID: 5,
}


# =============================================================================
# Card identity space
# =============================================================================
# The compiled enum already exposes only Ironclad (RED), colorless, curse, and
# status cards. We further drop colorless cards an Ironclad can never obtain
# (generated-only tokens), then pack the survivors into a dense [0, NUM_CARDS)
# index used both as the embedding id (card feature [0]) and the pile-count slot.
# Anything outside the set (e.g. an off-pool card spawned mid-combat) folds into a
# single UNKNOWN bucket so encoding never indexes out of range.

_EXCLUDED_CARD_NAMES = {
    "BECOME_ALMIGHTY", "BETA", "EXPUNGER", "FAME_AND_FORTUNE", "INSIGHT",
    "LIVE_FOREVER", "MIRACLE", "OMEGA", "SAFETY", "SHIV", "SMITE", "THROUGH_VIOLENCE",
}


def _build_card_index():
    allowed = []
    for name, cid in sts.CardId.__members__.items():
        if name == "INVALID" or name in _EXCLUDED_CARD_NAMES:
            continue
        allowed.append(cid)
    allowed.sort(key=lambda c: int(c))
    return allowed, {int(c): i for i, c in enumerate(allowed)}


ALLOWED_CARD_IDS, CARD_ID_TO_INDEX = _build_card_index()
UNKNOWN_CARD_INDEX = len(ALLOWED_CARD_IDS)
NUM_CARDS = UNKNOWN_CARD_INDEX + 1   # dense slots + one UNKNOWN bucket

# Materialize the whole raw-id -> dense-index map as a flat lookup table, so the
# per-card mapping in the hot encode path is a single array index rather than a dict
# hash. The engine always hands us raw CardId enums, so this boundary remap is
# unavoidable; the LUT just makes it as cheap as possible. Unmapped ids -> UNKNOWN.
_MAX_RAW_CARD_ID = max(int(c) for c in sts.CardId.__members__.values())
CARD_INDEX_LUT = np.full(_MAX_RAW_CARD_ID + 1, UNKNOWN_CARD_INDEX, dtype=np.int32)
for _cid, _idx in CARD_ID_TO_INDEX.items():
    CARD_INDEX_LUT[_cid] = _idx


def index_for_card_id(card_id: int) -> int:
    """Dense embedding/pile index for a raw CardId int (UNKNOWN if out of set)."""
    cid = int(card_id)
    return int(CARD_INDEX_LUT[cid]) if 0 <= cid <= _MAX_RAW_CARD_ID else UNKNOWN_CARD_INDEX


# =============================================================================
# Relic identity space + card-select task
# =============================================================================
# Relics are static for a combat, so the observation carries a multi-hot presence
# vector over a dense relic index (the BattleContext doesn't expose a relic list, so
# the session builds this once at reset from its known relics). This lets the net
# condition play on relic synergies (Pen Nib, Kunai, Velvet Choker, Snecko Eye, ...).

def _build_relic_index():
    allowed = [r for n, r in sts.RelicId.__members__.items() if n != "INVALID"]
    allowed.sort(key=lambda r: int(r))
    return allowed, {int(r): i for i, r in enumerate(allowed)}


ALLOWED_RELIC_IDS, RELIC_ID_TO_INDEX = _build_relic_index()
NUM_RELICS = len(ALLOWED_RELIC_IDS)


def relic_vector(relic_id_ints) -> np.ndarray:
    """Multi-hot presence vector over the relic index for a list of raw RelicId ints."""
    v = np.zeros(NUM_RELICS, np.float32)
    for rid in relic_id_ints:
        i = RELIC_ID_TO_INDEX.get(int(rid))
        if i is not None:
            v[i] = 1.0
    return v


# Card-select task id (which select the agent is making: ARMAMENTS upgrade vs
# EXHAUST_ONE vs WARCRY top-of-draw, ...). Encoded so the net knows whether to pick a
# good or bad card. Tasks are 0..NUM_SELECT_TASKS-1; SELECT_TASK_NONE marks "not selecting".
NUM_SELECT_TASKS = max(int(v) for v in sts.CardSelectTask.__members__.values()) + 1
SELECT_TASK_NONE = NUM_SELECT_TASKS


# =============================================================================
# Small read-only helpers (shared by the session)
# =============================================================================

def enemy_total_hp(bc) -> int:
    return sum(max(0, mon.cur_hp) for mon in bc.monsters)


def card_features(card) -> np.ndarray:
    return np.array([
        float(index_for_card_id(int(card.id))),
        float(card.cost_for_turn),
        1.0 if card.upgraded else 0.0,
        1.0 if card.ethereal else 0.0,
        1.0 if card.exhausts else 0.0,
        1.0 if card.requires_target else 0.0,
        float(_CARD_TYPE_INT.get(card.type, 5)),
        card.special_data / _DMG_NORM,
    ], dtype=np.float32)


def select_candidate_cards(bc, select_actions) -> list:
    """Card referred to by each enumerated CARD_SELECT action (None for a skip).

    The returned list is index-aligned with `select_actions`, so candidate k lines up
    with select action k. Two kinds of source:

      * pile tasks    -- the select index points into hand / draw / discard / exhaust
      * generated tasks (Discovery, Codex) -- candidates are 3 freshly-rolled CardIds
        held in `card_select_info.cards`, not in any pile. Codex adds a 4th "skip"
        option (index 3) with no card.
    """
    task = bc.card_select_info.card_select_task
    T = sts.CardSelectTask

    # --- generated-card tasks: candidates come from card_select_info.cards ---
    if task in (T.DISCOVERY, T.CODEX):
        gen = []
        for cid in bc.card_select_info.cards:
            gen.append(None if cid == sts.CardId.INVALID else sts.CardInstance(cid))
        return [gen[si] if 0 <= (si := a.get_source_idx()) < len(gen) else None
                for a in select_actions]

    # --- pile tasks: the select index points into one specific pile ---
    # Defect/Silent/Watcher tasks (Seek, Recycle, Setup, Nightmare, Meditate) can
    # never fire for an Ironclad/colorless deck and are intentionally omitted.
    hand, draw = bc.cards.hand, bc.cards.draw_pile
    discard, exhaust = bc.cards.discard_pile, bc.cards.exhaust_pile
    if task in (T.ARMAMENTS,      # Armaments  -> upgrade a card in hand
                T.DUAL_WIELD,     # Dual Wield -> copy an Attack/Power in hand
                T.WARCRY,         # Warcry     -> put a hand card on top of draw
                T.EXHAUST_ONE,    # True Grit+ / Burning Pact -> exhaust a hand card
                T.FORETHOUGHT):   # Forethought (colorless)   -> hand card to bottom of draw
        source = hand
    elif task in (T.HEADBUTT,                 # Headbutt        -> from discard pile
                  T.LIQUID_MEMORIES_POTION):  # Liquid Memories -> from discard pile
        source = discard
    elif task == T.EXHUME:        # Exhume   -> a card in the exhaust pile
        source = exhaust
    elif task in (T.SECRET_WEAPON, T.SECRET_TECHNIQUE):  # colorless -> from draw pile
        source = draw
    else:
        source = hand             # safe default; any unlisted select reads from hand
    cards = list(source)
    return [cards[si] if 0 <= (si := a.get_source_idx()) < len(cards) else None
            for a in select_actions]


# =============================================================================
# Observation builder
# =============================================================================

def empty_observation() -> dict:
    return {
        "hand_cards": np.zeros((MAX_HAND, CARD_FEATURES), np.float32),
        "hand_mask": np.zeros(MAX_HAND, np.int8),
        "draw_pile": np.zeros(NUM_CARDS, np.float32),
        "discard_pile": np.zeros(NUM_CARDS, np.float32),
        "exhaust_pile": np.zeros(NUM_CARDS, np.float32),
        "player": np.zeros(PLAYER_DIM, np.float32),
        "monsters": np.zeros((MAX_ENEMIES, MONSTER_DIM), np.float32),
        "monster_mask": np.zeros(MAX_ENEMIES, np.int8),
        "potions": np.zeros(NUM_POTIONS, np.float32),
        "relics": np.zeros(NUM_RELICS, np.float32),
        "select_task": np.array([SELECT_TASK_NONE], np.float32),
        "scalars": np.zeros(SCALAR_DIM, np.float32),
    }


def select_candidate_features(bc, select_actions) -> np.ndarray:
    """Variable-length per-candidate features for the current CARD_SELECT decision.

    Returns an (n_candidates, CARD_FEATURES) array index-aligned with the SELECT_CARD
    actions in `select_actions` (zeros row for a skip slot). `n_candidates` is whatever
    the game offers -- 3 for Discovery, up to the whole draw pile for Secret Weapon --
    so this is never capped. The policy's select/pointer head attends over these rows
    and pads per minibatch; nothing global needs sizing.
    """
    cands = select_candidate_cards(bc, select_actions)
    feats = np.zeros((len(cands), CARD_FEATURES), np.float32)
    for k, card in enumerate(cands):
        if card is not None:
            feats[k] = card_features(card)
    return feats


def encode_observation(bc, relic_vec=None) -> dict:
    """Encode a (possibly hypothetical) BattleContext into observation arrays.

    This is the fixed-shape board state only. CARD_SELECT candidates are NOT here --
    see `select_candidate_features()` for the variable-length candidate set. `relic_vec`
    is the combat's static relic multi-hot (the session supplies it; bc has no list).
    """
    obs = empty_observation()

    # Hand
    for i, card in enumerate(bc.cards.hand):
        if i >= MAX_HAND:
            break
        obs["hand_cards"][i] = card_features(card)
        obs["hand_mask"][i] = 1

    # Piles as card-id counts
    for pile_key, pile in (("draw_pile", bc.cards.draw_pile),
                           ("discard_pile", bc.cards.discard_pile),
                           ("exhaust_pile", bc.cards.exhaust_pile)):
        for card in pile:
            obs[pile_key][index_for_card_id(int(card.id))] += 1.0

    # Player
    p = bc.player
    obs["player"][0] = p.cur_hp / _HP_NORM
    obs["player"][1] = p.max_hp / _HP_NORM
    obs["player"][2] = p.block / _BLOCK_NORM
    obs["player"][3] = p.energy / _ENERGY_NORM
    obs["player"][4] = p.energy_per_turn / _ENERGY_NORM
    # per-turn play counts (drive Velvet Choker / Kunai / Shuriken / per-turn relics)
    obs["player"][5] = p.cards_played_this_turn / _COUNT_NORM
    obs["player"][6] = p.attacks_played_this_turn / _COUNT_NORM
    obs["player"][7] = p.skills_played_this_turn / _COUNT_NORM
    obs["player"][8] = p.cards_discarded_this_turn / _COUNT_NORM
    # persistent relic counters (progress toward Pen Nib / Nunchaku / Ink Bottle / etc.)
    obs["player"][9] = p.happy_flower_counter / _COUNT_NORM
    obs["player"][10] = p.incense_burner_counter / _COUNT_NORM
    obs["player"][11] = p.ink_bottle_counter / _COUNT_NORM
    obs["player"][12] = p.inserter_counter / _COUNT_NORM
    obs["player"][13] = p.nunchaku_counter / _COUNT_NORM
    obs["player"][14] = p.pen_nib_counter / _COUNT_NORM
    obs["player"][15] = p.sundial_counter / _COUNT_NORM
    idx = PLAYER_BASE_DIM
    for status in PLAYER_STATUSES:
        try:
            if p.has_status(status):
                obs["player"][idx] = p.get_status(status) / _STACK_NORM
        except (KeyError, IndexError):
            pass
        idx += 1

    # Monsters
    for i, mon in enumerate(bc.monsters):
        if i >= MAX_ENEMIES:
            break
        obs["monsters"][i, 0] = max(0, mon.cur_hp) / _HP_NORM
        obs["monsters"][i, 1] = mon.max_hp / _HP_NORM
        obs["monsters"][i, 2] = mon.block / _BLOCK_NORM
        try:
            di = mon.get_move_damage(bc)
            obs["monsters"][i, 3] = di.damage / _DMG_NORM
            obs["monsters"][i, 4] = di.attack_count / 5.0
        except Exception:
            pass
        obs["monsters"][i, 5] = 1.0 if mon.is_attacking() else 0.0
        alive = mon.is_alive()
        obs["monsters"][i, 6] = 1.0 if alive else 0.0
        obs["monsters"][i, 7] = 1.0 if mon.is_targetable() else 0.0
        j = MONSTER_BASE_DIM
        for status in MONSTER_STATUSES:
            try:
                if mon.has_status(status):
                    obs["monsters"][i, j] = mon.get_status(status) / _STACK_NORM
            except (KeyError, IndexError):
                pass
            j += 1
        if alive and mon.is_targetable():
            obs["monster_mask"][i] = 1

    # Potions held, as a count vector over potion ids (skip empty/invalid slots).
    for pot in bc.potions:
        pid = int(pot)
        if pot not in (sts.Potion.EMPTY_POTION_SLOT, sts.Potion.INVALID) and 0 <= pid < NUM_POTIONS:
            obs["potions"][pid] += 1.0

    # Whether a CARD_SELECT decision is pending (the candidates themselves are
    # exposed separately, variable-length, via select_candidate_features()).
    selecting = (bc.outcome == sts.BattleOutcome.UNDECIDED
                 and bc.input_state == sts.InputState.CARD_SELECT)

    # Scalars
    obs["scalars"][0] = bc.turn / 25.0
    obs["scalars"][1] = 1.0 if selecting else 0.0
    obs["scalars"][2] = 1.0 if (selecting and bc.card_select_info.can_pick_zero) else 0.0
    obs["scalars"][3] = bc.potion_count / float(MAX_POTIONS)
    obs["scalars"][4] = bc.monsters.get_alive_count() / float(MAX_ENEMIES)

    if relic_vec is not None:
        obs["relics"] = relic_vec
    obs["select_task"][0] = float(int(bc.card_select_info.card_select_task)
                                  if selecting else SELECT_TASK_NONE)
    return obs
