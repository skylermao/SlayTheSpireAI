"""CombatSession: the single class your RL code talks to.

This is the whole interface between a hand-written training loop and the C++ sim. It
owns one live `slaythespire.BattleContext`, drives it to the next decision point, and
exposes the state/actions/transitions an agent needs -- without any gym dependency.

-----------------------------------------------------------------------------
Data flow (everything to/from sts_lightspeed goes through this class):

    config ──reset()──▶ build GameContext, enter_battle, BattleContext.init
                        │
                        ▼  drain queued sim actions to a decision (`_settle`)
    ┌──────────────── decision point ────────────────┐
    │  observe()        -> obs tensors (read-only)    │   ◀── encoding.encode_observation
    │  legal_actions()  -> [LegalAction]  (concrete)  │   ◀── is_valid / enumerate_actions
    │  step(action)     -> Transition(obs, r, done)   │   ──▶ Action.execute + execute_actions
    └─────────────────────────────────────────────────┘
                        │
                        ▼  back to settle / next decision, until outcome != UNDECIDED

    Model interface for afterstate / sasRL search (no mutation of `self`):
        clone()         -> deep copy via BattleContext copy-ctor
        afterstate(a)   -> clone, apply a, settle  (a fresh CombatSession)
        peek(a)         -> Transition you'd get from step(a), without taking it
-----------------------------------------------------------------------------

The session reimplements no game logic: legality comes from `Action.is_valid`, card-
select options from `Action.enumerate_actions`, and every transition from the sim's
own `execute`. The only Python-side bookkeeping is shaped reward and termination.
"""

from dataclasses import dataclass
from typing import Optional, Union

from ._sts import sts
from . import encoding as enc
from .scenario import CombatConfig, RewardConfig, DatasetSampler
from .scenario import resolve_card, resolve_relic, resolve_encounter, resolve_potion


# Action kinds (the discrete "what kind of move" label on a LegalAction).
END_TURN = "end_turn"
PLAY_CARD = "play_card"
USE_POTION = "use_potion"
SELECT_CARD = "select_card"
SKIP_SELECT = "skip_select"


@dataclass(frozen=True)
class LegalAction:
    """One fully-concrete legal move (target already resolved).

    The position of a LegalAction in `legal_actions()` is its action index -- that is
    the only "action space" a hand-written policy needs. `action` is the sim object to
    execute; the rest is metadata for masking heads, embeddings, and logging.
    """
    kind: str
    action: "sts.Action"
    source_idx: int = -1   # hand index (play), potion slot (potion), or select source
    target_idx: int = -1   # monster idx for targeted moves, else -1
    card_id: int = -1      # raw CardId int (true identity), else -1
    card_index: int = -1   # dense embedding index (matches obs card features), else -1
    potion_id: int = -1    # raw Potion int for USE_POTION actions, else -1
    calc_damage: int = -1  # actual damage a PLAY_CARD would deal to its target, else -1
    label: str = ""        # human-readable, for render/logging

    def __repr__(self) -> str:
        return f"LegalAction({self.label or self.kind})"


@dataclass
class Transition:
    """Result of taking (or peeking) one action."""
    obs: dict
    reward: float
    done: bool
    truncated: bool
    info: dict


class CombatSession:
    """Drives a single Slay the Spire combat (Ironclad) over the C++ sim."""

    def __init__(
        self,
        config: Optional[CombatConfig] = None,
        sampler: Optional[DatasetSampler] = None,
        reward_config: Optional[RewardConfig] = None,
        max_turns: int = 60,
    ):
        if config is None and sampler is None:
            raise ValueError("Provide a CombatConfig and/or a DatasetSampler")
        self._config = config
        self._sampler = sampler
        self.reward_cfg = reward_config or RewardConfig()
        self.max_turns = max_turns

        self.gc: Optional[sts.GameContext] = None
        self.bc: Optional[sts.BattleContext] = None
        self._select_actions: list = []   # cached enumerate during CARD_SELECT
        self._seed_counter = 0
        self._start_cur_hp = 0
        self._relic_vec = None             # static relic multi-hot for the live combat
        self.cfg: Optional[CombatConfig] = None   # the config of the live combat

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def reset(self, config: Optional[CombatConfig] = None, seed: Optional[int] = None) -> dict:
        """Start a fresh combat and return the first observation."""
        cfg = config or (self._sampler.sample() if self._sampler else self._config)
        self.cfg = cfg

        if cfg.seed is not None:
            battle_seed = cfg.seed
        elif seed is not None:
            battle_seed = seed
        else:
            battle_seed = self._seed_counter
            self._seed_counter += 1

        self._build_battle(cfg, battle_seed)
        return self.observe()

    def _build_battle(self, cfg: CombatConfig, seed: int) -> None:
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, cfg.ascension)

        # Replace the default starter deck with the scenario deck.
        while len(gc.deck) > 0:
            gc.remove_card(0)
        for spec in cfg.deck:
            card = resolve_card(spec)
            if card is None:
                raise ValueError(f"Unmappable card in deck: {spec!r}")
            gc.obtain_card(card)

        # Relics first (pickup effects fire), then potions.
        relic_ids = []
        for spec in cfg.relics:
            relic_id = resolve_relic(spec)
            if relic_id is None:
                raise ValueError(f"Unmappable relic: {spec!r}")
            relic_ids.append(int(relic_id))
            if not gc.has_relic(relic_id):
                gc.obtain_relic(relic_id)
        self._relic_vec = enc.relic_vector(relic_ids)   # static for the combat
        for spec in cfg.potions:
            potion = resolve_potion(spec)
            if potion is not None and potion != sts.Potion.EMPTY_POTION_SLOT:
                gc.obtain_potion(potion)

        # Force HP to the snapshot AFTER relics, overriding pickup HP side effects.
        max_diff = cfg.max_hp - gc.max_hp
        if max_diff > 0:
            gc.player_increase_max_hp(max_diff)
        elif max_diff < 0:
            gc.lose_max_hp(-max_diff)
        target_hp = cfg.cur_hp if cfg.cur_hp is not None else cfg.max_hp
        hp_diff = gc.cur_hp - target_hp
        if hp_diff > 0:
            gc.damage_player(hp_diff)
        elif hp_diff < 0:
            gc.player_heal(-hp_diff)

        encounter = resolve_encounter(cfg.encounter)
        if encounter is None:
            raise ValueError(f"Unmappable encounter: {cfg.encounter!r}")
        gc.enter_battle(encounter)

        bc = sts.BattleContext()
        bc.init(gc)

        self.gc, self.bc = gc, bc
        self._start_cur_hp = bc.player.cur_hp
        self._settle()

    # ========================================================================
    # Sim driving (drain to a decision; the one place we touch execute_actions)
    # ========================================================================

    def _settle(self) -> None:
        """Flush queued sim actions until a decision is pending or the fight ends."""
        guard = 0
        while (not self.done
               and self.bc.input_state == sts.InputState.EXECUTING_ACTIONS):
            self.bc.execute_actions()
            guard += 1
            if guard > 10000:
                raise RuntimeError("settle loop did not converge")
        self._refresh_select_actions()

    def _refresh_select_actions(self) -> None:
        if not self.done and self.bc.input_state == sts.InputState.CARD_SELECT:
            self._select_actions = list(sts.Action.enumerate_actions(self.bc))
        else:
            self._select_actions = []

    # ========================================================================
    # Status queries
    # ========================================================================

    @property
    def done(self) -> bool:
        """True once the fight is decided.

        `is_battle_over` can lag (stays False with input_state == EXECUTING_ACTIONS
        when the player dies on the enemy turn), so `outcome` is authoritative.
        """
        return self.bc is not None and self.bc.outcome != sts.BattleOutcome.UNDECIDED

    @property
    def truncated(self) -> bool:
        return (not self.done) and self.bc is not None and self.bc.turn >= self.max_turns

    @property
    def outcome(self) -> str:
        return self.bc.outcome.name if self.bc is not None else "UNDECIDED"

    @property
    def won(self) -> bool:
        return self.bc is not None and self.bc.outcome == sts.BattleOutcome.PLAYER_VICTORY

    # ========================================================================
    # The decision interface
    # ========================================================================

    def observe(self) -> dict:
        """Read-only encode of the fixed-shape board state into observation tensors."""
        return enc.encode_observation(self.bc, self._relic_vec)

    def select_candidate_features(self) -> "np.ndarray":
        """Variable-length (n_candidates, CARD_FEATURES) for the pending CARD_SELECT.

        Index-aligned with the SELECT_CARD entries of `legal_actions()`. Empty (0 rows)
        when no card-select is pending. The select/pointer head consumes these and pads
        per minibatch -- there is no global candidate cap.
        """
        return enc.select_candidate_features(self.bc, self._select_actions)

    def reseed(self, seed: int) -> None:
        """Re-seed the battle RNGs so this (cloned) session samples a fresh stochastic
        outcome on the next action -- reshuffles, random card effects, monster AI. Does
        not reorder the current draw pile (its order is part of state).
        """
        self.bc.reseed(int(seed) & 0x7FFFFFFFFFFFFFFF)

    def determinize(self, seed: int) -> None:
        """Full search-time determinization of the hidden state with an independent
        seed: reseed the battle RNGs *and* re-shuffle the draw-pile order. This makes a
        cloned session a *truly random* draw from the agent's information set -- the
        draw it produces is no longer the real game's privileged next draw. Used by
        MCTS chance nodes; the real session (reality) is never determinized, so a
        committed move still resolves to the true simulator output.
        """
        s = int(seed) & 0x7FFFFFFFFFFFFFFF
        self.bc.reseed(s)
        self.bc.shuffle_draw_pile(s)

    def state_signature(self):
        """Hashable key over the *observable* (information-set) combat state, for MCTS
        chance-node dedup. Hidden draw-pile ORDER is excluded -- piles are multisets --
        so states the policy cannot tell apart merge into one node.
        """
        bc = self.bc
        if self.done:
            return ("terminal", bc.outcome.name, max(0, bc.player.cur_hp))

        def ck(c):  # card identity that matters for play, order-independent
            return (enc.index_for_card_id(int(c.id)), int(c.cost_for_turn),
                    int(c.special_data), 1 if c.upgraded else 0)

        hand = tuple(sorted(ck(c) for c in bc.cards.hand))
        draw = tuple(sorted(ck(c) for c in bc.cards.draw_pile))
        discard = tuple(sorted(ck(c) for c in bc.cards.discard_pile))
        exhaust = tuple(sorted(ck(c) for c in bc.cards.exhaust_pile))

        p = bc.player
        pst = []
        for s in enc.PLAYER_STATUSES:
            try:
                pst.append(p.get_status(s) if p.has_status(s) else 0)
            except (KeyError, IndexError):
                pst.append(0)
        player = (p.cur_hp, p.block, p.energy, tuple(pst))

        mons = []
        for m in bc.monsters:
            mst = []
            for s in enc.MONSTER_STATUSES:
                try:
                    mst.append(m.get_status(s) if m.has_status(s) else 0)
                except (KeyError, IndexError):
                    mst.append(0)
            try:
                di = m.get_move_damage(bc)
                intent = (di.damage, di.attack_count)
            except Exception:
                intent = (0, 0)
            mons.append((max(0, m.cur_hp), m.block, bool(m.is_alive()), intent, tuple(mst)))

        select = ()
        if bc.input_state == sts.InputState.CARD_SELECT:
            csi = bc.card_select_info
            cands = tuple(sorted(
                enc.index_for_card_id(int(c.id))
                for c in enc.select_candidate_cards(bc, self._select_actions) if c is not None))
            select = (int(csi.card_select_task), bool(csi.can_pick_zero), cands)

        return (int(bc.input_state), bc.turn, player, tuple(mons),
                hand, draw, discard, exhaust, select)

    def legal_actions(self) -> list[LegalAction]:
        """Every concrete legal move right now, as a flat list.

        The list *is* the action mask: index k is action k. Empty if the fight is
        over or the sim is mid-execution (which `_settle` should have drained).
        """
        bc = self.bc
        if bc is None or self.done:
            return []
        out: list[LegalAction] = []

        if bc.input_state == sts.InputState.PLAYER_NORMAL:
            if sts.Action(sts.ActionType.END_TURN).is_valid(bc):
                out.append(LegalAction(END_TURN, sts.Action(sts.ActionType.END_TURN),
                                       label="end turn"))

            for i in range(bc.cards.cards_in_hand):
                card = bc.cards.hand[i]
                name = f"{card.name}{'+' if card.upgraded else ''}"
                cid = int(card.id)
                cidx = enc.index_for_card_id(cid)
                if card.requires_target:
                    for mon in bc.monsters:
                        t = mon.idx
                        a = sts.Action(sts.ActionType.CARD, i, t)
                        if a.is_valid(bc):
                            out.append(LegalAction(PLAY_CARD, a, source_idx=i, target_idx=t,
                                                   card_id=cid, card_index=cidx,
                                                   calc_damage=self._card_damage(i, t),
                                                   label=f"play {name} -> {mon.get_name()}"))
                else:
                    a = sts.Action(sts.ActionType.CARD, i, 0)
                    if a.is_valid(bc):
                        out.append(LegalAction(PLAY_CARD, a, source_idx=i,
                                               card_id=cid, card_index=cidx,
                                               calc_damage=self._card_damage(i, 0),
                                               label=f"play {name}"))

            for p in range(min(bc.potion_capacity, enc.MAX_POTIONS)):
                playable, needs_target = self._potion_status(p)
                if not playable:
                    continue
                pid = int(bc.potions[p])
                if needs_target:
                    for mon in bc.monsters:
                        a = sts.Action(sts.ActionType.POTION, p, mon.idx)
                        if a.is_valid(bc):
                            out.append(LegalAction(USE_POTION, a, source_idx=p,
                                                   target_idx=mon.idx, potion_id=pid,
                                                   label=f"potion {p} -> {mon.get_name()}"))
                else:
                    out.append(LegalAction(USE_POTION, sts.Action(sts.ActionType.POTION, p),
                                           source_idx=p, potion_id=pid,
                                           label=f"potion {p}"))

        elif bc.input_state == sts.InputState.CARD_SELECT:
            candidates = enc.select_candidate_cards(bc, self._select_actions)
            for a, card in zip(self._select_actions, candidates):
                if card is None:   # e.g. Codex's index-3 skip option
                    out.append(LegalAction(SELECT_CARD, a, source_idx=a.get_source_idx(),
                                           label="select (skip)"))
                    continue
                cid = int(card.id)
                name = f"{card.name}{'+' if card.upgraded else ''}"
                out.append(LegalAction(SELECT_CARD, a, source_idx=a.get_source_idx(),
                                       card_id=cid, card_index=enc.index_for_card_id(cid),
                                       label=f"select {name}"))
            if bc.card_select_info.can_pick_zero:
                # Convention used across this repo: select index -1 ends the pick.
                out.append(LegalAction(SKIP_SELECT,
                                       sts.Action(sts.ActionType.SINGLE_CARD_SELECT, -1),
                                       label="skip select"))

        return out

    def _card_damage(self, hand_idx: int, target_idx: int) -> int:
        """Actual damage the hand card would deal to `target_idx` (post all modifiers,
        incl. the target's Vulnerable); -1 for non-attacks. Per-action feature for the
        policy head -- damage is target-dependent so it can't be a card feature."""
        try:
            return int(self.bc.get_card_damage(hand_idx, target_idx))
        except Exception:
            return -1

    def _potion_status(self, p: int):
        """(playable, needs_target) for potion slot p, by probing sim validity."""
        bc = self.bc
        if p >= bc.potion_capacity:
            return False, False
        potions = bc.potions
        if p >= len(potions) or potions[p] in (sts.Potion.EMPTY_POTION_SLOT, sts.Potion.INVALID):
            return False, False
        if sts.Action(sts.ActionType.POTION, p).is_valid(bc):
            return True, False
        for t in range(bc.monsters.monster_count):
            if sts.Action(sts.ActionType.POTION, p, t).is_valid(bc):
                return True, True
        return False, False

    def step(self, action: Union[LegalAction, int]) -> Transition:
        """Apply an action to the live sim and advance to the next decision."""
        if self.bc is None:
            raise RuntimeError("Call reset() before step()")
        la = self._resolve(action)

        prev_enemy_hp = enc.enemy_total_hp(self.bc)
        prev_player_hp = self.bc.player.cur_hp

        la.action.execute(self.bc)
        self.bc.execute_actions()
        self._settle()

        return self._transition(prev_enemy_hp, prev_player_hp, la.kind == END_TURN)

    # ========================================================================
    # Model interface: clone / afterstate / peek (no mutation of self)
    # ========================================================================

    def clone(self) -> "CombatSession":
        """A detached copy sharing config, with its own BattleContext (copy-ctor)."""
        new = CombatSession.__new__(CombatSession)
        new._config = self._config
        new._sampler = self._sampler
        new.reward_cfg = self.reward_cfg
        new.max_turns = self.max_turns
        new._seed_counter = self._seed_counter
        new._start_cur_hp = self._start_cur_hp
        new._relic_vec = self._relic_vec      # static; safe to share (read-only)
        new.cfg = self.cfg
        new.gc = None  # not needed once combat is running
        new.bc = sts.BattleContext(self.bc)
        new._select_actions = []
        new._refresh_select_actions()
        return new

    def afterstate(self, action: Union[LegalAction, int]) -> "CombatSession":
        """Clone, apply `action`, settle -- the resulting state as a fresh session.

        NOTE: stochastic actions (any card that draws) realize one RNG sample here.
        For an expected-afterstate value, average over several clones.
        """
        child = self.clone()
        la = child._resolve(action)
        la.action.execute(child.bc)
        child.bc.execute_actions()
        child._settle()
        return child

    def peek(self, action: Union[LegalAction, int]) -> Transition:
        """The Transition `step(action)` would return, without taking it."""
        la = self._resolve(action)
        prev_enemy_hp = enc.enemy_total_hp(self.bc)
        prev_player_hp = self.bc.player.cur_hp
        child = self.afterstate(la)
        return child._transition(prev_enemy_hp, prev_player_hp, la.kind == END_TURN)

    # ========================================================================
    # Reward / info / render
    # ========================================================================

    def _resolve(self, action: Union[LegalAction, int]) -> LegalAction:
        if isinstance(action, LegalAction):
            return action
        legal = self.legal_actions()
        if not 0 <= int(action) < len(legal):
            raise IndexError(f"action index {action} out of range (0..{len(legal) - 1})")
        return legal[int(action)]

    def _transition(self, prev_enemy_hp: int, prev_player_hp: int,
                    is_end_turn: bool) -> Transition:
        """Build the Transition for the current (post-apply) state of self.bc."""
        rc = self.reward_cfg
        hp_unit = max(self.cfg.max_hp if rc.normalize_by_max_hp else 1.0, 1.0)
        dmg_dealt = max(0, prev_enemy_hp - enc.enemy_total_hp(self.bc))
        hp_lost = max(0, prev_player_hp - self.bc.player.cur_hp)
        reward = (rc.damage_dealt_coef * dmg_dealt - rc.hp_loss_coef * hp_lost) / hp_unit
        if is_end_turn:
            reward -= rc.turn_penalty

        done = self.done
        truncated = self.truncated
        if done:
            reward += rc.win_bonus if self.won else -rc.death_penalty

        info = self.info()
        if done:
            info["won"] = self.won
            info["agent_hp_loss"] = self._start_cur_hp - max(0, self.bc.player.cur_hp)
        return Transition(self.observe(), float(reward), done, truncated, info)

    def info(self) -> dict:
        bc = self.bc
        return {
            "turn": bc.turn,
            "input_state": bc.input_state.name,
            "player_hp": bc.player.cur_hp,
            "player_max_hp": bc.player.max_hp,
            "player_block": bc.player.block,
            "energy": bc.player.energy,
            "enemy_hp": enc.enemy_total_hp(bc),
            "enemies_alive": bc.monsters.get_alive_count(),
            "outcome": bc.outcome.name,
            "human_hp_loss": self.cfg.human_hp_loss if self.cfg else None,
        }

    def render(self) -> str:
        bc = self.bc
        if bc is None:
            return "<no combat>"
        lines = [f"--- Turn {bc.turn} | {bc.input_state.name} | outcome {bc.outcome.name} ---",
                 f"Player HP {bc.player.cur_hp}/{bc.player.max_hp}  block {bc.player.block}"
                 f"  energy {bc.player.energy}"]
        for mon in bc.monsters:
            if mon.is_alive():
                di = mon.get_move_damage(bc)
                lines.append(f"  [{mon.idx}] {mon.get_name()} HP {mon.cur_hp}/{mon.max_hp}"
                             f" block {mon.block} intent {di.damage}x{di.attack_count}")
        lines.append("Hand: " + ", ".join(
            f"{c.name}{'+' if c.upgraded else ''}({c.cost_for_turn})" for c in bc.cards.hand))
        return "\n".join(lines)

    def close(self) -> None:
        self.gc = None
        self.bc = None
