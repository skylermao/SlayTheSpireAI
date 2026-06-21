"""Verbose greedy playthrough: narrate exactly what the trained agent does.

Plays a few sampled (normal) combats with greedy MCTS and prints a full transcript --
starting setup (relics/deck/potions/enemy/HP), then per player-turn: HP/block/energy,
player & enemy statuses, enemy intent, and each card the agent plays (with target), plus
what the enemy turn did. Intended for inspection, not training.

    PYTHONPATH=. python -m rl.combat.playthrough --ckpt checkpoints_from_ec2/net_v5_final.pt \
        --games 2 --sims 128 --normal-only --seed 0
"""

import argparse
import random
from collections import Counter

import torch

from .scenario import DatasetSampler, NON_NORMAL_ENCOUNTERS
from .session import CombatSession, PLAY_CARD, USE_POTION, END_TURN, SELECT_CARD, SKIP_SELECT
from .mcts import MCTS, MCTSConfig
from .net import CombatNet, NeuralEvaluator
from . import encoding as enc


def _player_statuses(p):
    out = []
    for s in enc.PLAYER_STATUSES:
        try:
            if p.has_status(s):
                out.append(f"{s.name.title()} {p.get_status(s)}")
        except Exception:
            pass
    return out


def _monster_statuses(m):
    out = []
    for s in enc.MONSTER_STATUSES:
        try:
            if m.has_status(s):
                out.append(f"{s.name.title()} {m.get_status(s)}")
        except Exception:
            pass
    return out


def _intent(m, bc):
    try:
        di = m.get_move_damage(bc)
    except Exception:
        return "?"
    if m.is_attacking() and di.damage > 0:
        hits = f"x{di.attack_count}" if di.attack_count > 1 else ""
        total = f" (={di.damage * di.attack_count})" if di.attack_count > 1 else ""
        return f"ATTACK {di.damage}{hits}{total}"
    return "non-attack (buff/debuff/defend)"


def _enemy_lines(bc):
    lines = []
    for m in bc.monsters:
        if not m.is_alive():
            continue
        st = _monster_statuses(m)
        st_s = f"  [{', '.join(st)}]" if st else ""
        blk = f" block {m.block}" if m.block else ""
        lines.append(f"      - {m.get_name()}: HP {max(0,m.cur_hp)}/{m.max_hp}{blk}  "
                     f"intent: {_intent(m, bc)}{st_s}")
    return lines


def _deck_summary(cards):
    """cards: CardInstances (hand) -> use .name; or spec strings (deck) -> use as-is."""
    def label(x):
        if isinstance(x, str):
            return x
        return f"{x.name}{'+' if x.upgraded else ''}"
    c = Counter(label(x) for x in cards)
    return ", ".join(f"{n}x{name}" if n > 1 else name for name, n in sorted(c.items()))


def play_structured(mcts, session) -> dict:
    """Play one combat greedily, returning a structured transcript (for UI/CLI)."""
    s = session
    s.reset()
    bc = s.bc
    cfg = s.cfg

    setup = {
        "encounter": cfg.encounter,
        "player_hp": bc.player.cur_hp, "player_max_hp": bc.player.max_hp,
        "human_entering_hp": cfg.cur_hp, "human_hp_loss": cfg.human_hp_loss,
        "relics": list(cfg.relics), "potions": [p.name for p in cfg.potions],
        "deck": _deck_summary(list(cfg.deck)), "deck_size": len(cfg.deck),
        "energy_per_turn": bc.player.energy_per_turn,
    }
    turns, cur = [], None
    last_turn, move_count = None, 0
    while not (s.done or s.truncated):
        if bc.input_state.name == "PLAYER_NORMAL" and bc.turn != last_turn:
            last_turn = bc.turn
            cur = {
                "turn": bc.turn, "player_hp": bc.player.cur_hp,
                "player_max_hp": bc.player.max_hp, "block": bc.player.block,
                "energy": bc.player.energy, "energy_per_turn": bc.player.energy_per_turn,
                "statuses": _player_statuses(bc.player),
                "hand": _deck_summary(list(bc.cards.hand)),
                "draw": len(list(bc.cards.draw_pile)),
                "discard": len(list(bc.cards.discard_pile)),
                "exhaust": len(list(bc.cards.exhaust_pile)),
                "enemies": [{
                    "name": m.get_name(), "hp": max(0, m.cur_hp), "max_hp": m.max_hp,
                    "block": m.block, "intent": _intent(m, bc),
                    "statuses": _monster_statuses(m),
                } for m in bc.monsters if m.is_alive()],
                "actions": [],
            }
            turns.append(cur)

        action, _, _ = mcts.run(s)
        if action is None:
            break
        if cur is not None:
            if action.kind == END_TURN:
                cur["actions"].append(f"End turn (block {bc.player.block})")
            else:
                cur["actions"].append(action.label)
        s.step(action)
        move_count += 1
        if move_count > 400:
            break

    return {
        "setup": setup, "turns": turns,
        "result": {
            "won": s.won, "done": s.done,
            "final_hp": max(0, bc.player.cur_hp), "max_hp": bc.player.max_hp,
            "hp_lost": s._start_cur_hp - max(0, bc.player.cur_hp),
            "human_hp_loss": cfg.human_hp_loss,
        },
    }


def play_one(mcts, sampler, gidx):
    s = CombatSession(sampler=sampler)
    s.reset()
    bc = s.bc
    cfg = s.cfg

    print("=" * 78)
    print(f"COMBAT {gidx}")
    print("=" * 78)
    print(f"  Enemy encounter : {cfg.encounter}")
    print(f"  Player HP       : {bc.player.cur_hp}/{bc.player.max_hp}"
          f"   (human entered at {cfg.cur_hp}; human lost {cfg.human_hp_loss} HP here)")
    print(f"  Relics          : {', '.join(cfg.relics) if cfg.relics else '(none)'}")
    pots = [p.name for p in cfg.potions] if cfg.potions else []
    print(f"  Potions         : {', '.join(pots) if pots else '(none)'}")
    print(f"  Deck ({len(cfg.deck)} cards) : {_deck_summary(list(cfg.deck))}")
    print(f"  Energy/turn     : {bc.player.energy_per_turn}")

    last_turn = None
    turn_start_hp = bc.player.cur_hp
    move_count = 0
    while not (s.done or s.truncated):
        # New player turn header (only when we're at a normal decision, not mid-select)
        if bc.input_state.name == "PLAYER_NORMAL" and bc.turn != last_turn:
            last_turn = bc.turn
            took = turn_start_hp - bc.player.cur_hp
            print(f"\n  --- TURN {bc.turn} ---")
            pst = _player_statuses(bc.player)
            print(f"    Player: HP {bc.player.cur_hp}/{bc.player.max_hp}  block {bc.player.block}"
                  f"  energy {bc.player.energy}/{bc.player.energy_per_turn}"
                  + (f"  statuses [{', '.join(pst)}]" if pst else ""))
            print(f"    Hand: {_deck_summary(bc.cards.hand)}")
            print(f"    Draw {len(list(bc.cards.draw_pile))} | "
                  f"Discard {len(list(bc.cards.discard_pile))} | Exhaust {len(list(bc.cards.exhaust_pile))}")
            print(f"    Enemies:")
            for ln in _enemy_lines(bc):
                print(ln)
            print(f"    Agent plays:")
            turn_start_hp = bc.player.cur_hp

        action, _, _ = mcts.run(s)
        if action is None:
            break

        # Narrate the action
        if action.kind == PLAY_CARD:
            print(f"      > {action.label}")
        elif action.kind == USE_POTION:
            print(f"      > {action.label}")
        elif action.kind in (SELECT_CARD, SKIP_SELECT):
            print(f"      > (select) {action.label}")
        elif action.kind == END_TURN:
            blk = bc.player.block
            print(f"      > END TURN  (ending with {blk} block)")

        s.step(action)
        move_count += 1
        if move_count > 400:
            print("    [safety cutoff]")
            break

    print(f"\n  RESULT: {'VICTORY' if s.won else ('LOSS' if s.done else 'truncated')}"
          f"  final HP {max(0,bc.player.cur_hp)}/{bc.player.max_hp}"
          f"  (HP lost this combat: {s._start_cur_hp - max(0,bc.player.cur_hp)};"
          f" human lost {cfg.human_hp_loss})")
    print()
    return s.won


def cfg_cards(cfg):
    # fallback: rebuild card list from cfg specs (names only) -- approximate
    from .scenario import resolve_card
    out = []
    for spec in cfg.deck:
        c = resolve_card(spec)
        if c is not None:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/ironclad_a0_fights.json.gz")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--normal-only", action="store_true")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    net = CombatNet(d_model=args.d_model)
    sd = torch.load(args.ckpt, map_location="cpu")
    net.load_state_dict(sd.get("net", sd))
    net.eval()
    evaluator = NeuralEvaluator(net, device=args.device)
    mcts = MCTS(evaluator, MCTSConfig(num_simulations=args.sims, temperature=0.0,
                                      add_root_noise=False, seed=args.seed))

    fights = DatasetSampler.from_gzip(args.data, rng=random.Random(args.seed)).fights
    if args.normal_only:
        fights = [f for f in fights if f["enemies"] not in NON_NORMAL_ENCOUNTERS]
    sampler = DatasetSampler(fights, rng=random.Random(args.seed))

    wins = 0
    for g in range(1, args.games + 1):
        wins += play_one(mcts, sampler, g)
    print("=" * 78)
    print(f"SUMMARY: {wins}/{args.games} won")


if __name__ == "__main__":
    main()
