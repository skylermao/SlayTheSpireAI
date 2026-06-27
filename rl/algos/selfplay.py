"""Concurrent self-play that batches leaf evaluations across games.

Each game plays to completion via MCTS; the search is a generator that *pauses* at
every leaf needing evaluation (`MCTS.search_gen`). The driver runs many game
generators at once, gathers the one pending eval request from each, runs a single
batched forward (`evaluator.evaluate_many`), and resumes them all. So the GPU sees
batches of size ~(#concurrent games) instead of one-at-a-time leaves.

Each real move records a training example: the root observation + its legal actions +
select candidates, the MCTS visit-count policy (the target pi), and -- filled at game
end -- the game value target z (win -> floor+hp, death -> -1, truncated -> 0).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.session import CombatSession, LegalAction
from .mcts import MCTS


@dataclass
class SelfPlayExample:
    """One training datum from a single real move."""
    obs: dict                       # observation arrays (fixed shape)
    actions: list                   # the LegalActions pi/logits are aligned to
    sel_feats: np.ndarray           # (n_select_candidates, CARD_FEATURES)
    policy: np.ndarray              # target pi over `actions` (MCTS visit distribution)
    z: float = 0.0                  # value target (set at game end)


@dataclass
class GameResult:
    examples: list = field(default_factory=list)
    outcome: str = "UNDECIDED"
    won: bool = False
    hp: int = 0
    moves: int = 0


def _play_game_gen(mcts: MCTS, session: CombatSession):
    """Generator: play one full combat, yielding leaf-eval requests, recording data."""
    session.reset()
    examples: list[SelfPlayExample] = []
    hps: list[int] = []                        # player HP at each recorded decision
    while not (session.done or session.truncated):
        action, policy, root = yield from mcts.search_gen(session)
        if action is None:
            break
        # root.session is the (unmutated) cloned root state -> obs/actions/policy align.
        examples.append(SelfPlayExample(
            obs=root.session.observe(),
            actions=root.actions,
            sel_feats=root.session.select_candidate_features(),
            policy=np.asarray(policy, dtype=np.float32),
        ))
        hps.append(root.session.bc.player.cur_hp)
        session.step(action)

    # Value targets are return-to-go: per-step HP-loss shaping + terminal outcome.
    terminal_value = mcts._game_value(session)
    max_hp = session.cfg.max_hp if session.cfg else 1
    targets = mcts.shaped_return_targets(hps, session.bc.player.cur_hp,
                                         terminal_value, max_hp)
    for ex, z in zip(examples, targets):
        ex.z = z
    return GameResult(examples=examples, outcome=session.outcome, won=session.won,
                      hp=session.bc.player.cur_hp, moves=len(examples))


def generate_selfplay(mcts: MCTS, sessions: "list[CombatSession]") -> "list[GameResult]":
    """Play `sessions` concurrently, batching leaf evals across all live games.

    Batch size per forward = number of games currently paused at a leaf (<= len(sessions)),
    so it shrinks as games finish. Returns one GameResult per input session, in order.
    """
    gens = {i: _play_game_gen(mcts, s) for i, s in enumerate(sessions)}
    pending: dict[int, CombatSession] = {}     # game idx -> session awaiting evaluation
    results: dict[int, GameResult] = {}

    def advance(i, send_val):
        try:
            pending[i] = gens[i].send(send_val) if send_val is not None else next(gens[i])
        except StopIteration as done:
            results[i] = done.value
            del gens[i]

    for i in list(gens):                       # prime each game to its first leaf
        advance(i, None)
    while gens:
        idxs = [i for i in pending if i in gens]
        evals = mcts.evaluator.evaluate_many([pending[i] for i in idxs])
        pending.clear()
        for i, ev in zip(idxs, evals):
            advance(i, ev)

    return [results[i] for i in range(len(sessions))]


def collect_selfplay(mcts: MCTS, make_session, n_games: int,
                     max_concurrent: int) -> "list[GameResult]":
    """Play `n_games` total, keeping up to `max_concurrent` running at once.

    Finished games are refilled from `make_session()` so batches stay near full until
    the final tail. Same batched-eval driver as `generate_selfplay`.
    """
    gens: dict[int, object] = {}
    pending: dict[int, CombatSession] = {}
    results: list[GameResult] = []
    started = 0

    def advance(gid, send_val):
        try:
            pending[gid] = gens[gid].send(send_val) if send_val is not None else next(gens[gid])
        except StopIteration as done:
            results.append(done.value)
            del gens[gid]

    def start():
        nonlocal started
        gid = started
        started += 1
        gens[gid] = _play_game_gen(mcts, make_session())
        advance(gid, None)

    while started < n_games and len(gens) < max_concurrent:
        start()
    while gens:
        idxs = [g for g in pending if g in gens]
        evals = mcts.evaluator.evaluate_many([pending[g] for g in idxs])
        pending.clear()
        for g, ev in zip(idxs, evals):
            advance(g, ev)
        while started < n_games and len(gens) < max_concurrent:
            start()
    return results


if __name__ == "__main__":
    import torch
    from ..core.scenario import CombatConfig
    from .mcts import MCTSConfig
    from .net import CombatNet, NeuralEvaluator

    torch.manual_seed(0)
    ev = NeuralEvaluator(CombatNet())
    mcts = MCTS(evaluator=ev, config=MCTSConfig(num_simulations=48, seed=0, temperature=1.0))

    cfg = CombatConfig(deck=["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
                       relics=["Burning Blood"], max_hp=80, cur_hp=80,
                       encounter="Jaw Worm", seed=None)

    n_games = 8
    sessions = [CombatSession(config=cfg) for _ in range(n_games)]
    results = generate_selfplay(mcts, sessions)

    n_ex = sum(len(r.examples) for r in results)
    wins = sum(r.won for r in results)
    print(f"device={ev.device}  games={n_games}  wins={wins}  examples={n_ex}")
    for r in results[:4]:
        print(f"  moves={r.moves} outcome={r.outcome} hp={r.hp} z={r.examples[0].z:.3f}")
