"""AlphaZero-style MCTS over CombatSession (step 1: search machinery + stub evaluator).

Tree structure mirrors the simulator's decision points exactly (see the discussion in
session.py): two node *kinds* interleave.

  DecisionNode  -- a CombatSession parked at PLAYER_NORMAL or CARD_SELECT. Its edges are
                   `legal_actions()` (targets pre-enumerated, selects are their own
                   nodes). Selection among edges is PUCT, seeded by evaluator priors.

  ChanceNode    -- the stochastic result of applying one edge. We *always* create one
                   after an edge; its children are deduped resulting DecisionNodes keyed
                   by `state_signature()`. A deterministic settle yields one child; a
                   stochastic one (reshuffle / Discovery roll / enemy move) yields
                   several, grown by progressive widening. Backup is frequency-weighted
                   (each visit contributes its leaf value to W/N), so Q is the empirical
                   expectation over outcomes -- no explicit probabilities needed.

Outcome sampling: clone the parent session, `determinize()` it with an independent
search seed (reseed battle RNGs + re-shuffle the hidden draw-pile order), then apply the
edge. The search thus draws from the agent's information set rather than peeking the
real game's next draw. Reality is never determinized -- a committed move on the real
session resolves to the true simulator output. A draw branches iff there is genuine
uncertainty (pile larger than the draw count); drawing the whole pile, or any
deterministic settle, dedups to a single outcome.

KNOWN approximation: the re-shuffle is uniform over the whole draw pile, so it forgets
*known* top-of-deck placements (Warcry / Headbutt putting a card on top). Preserving
those is a later refinement.

Value (bounded [-1, 1]): win -> win_value_floor + (1-floor)*hp_fraction; death -> -1;
max_turns truncation -> bootstrap with the evaluator's value.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .session import CombatSession, LegalAction
from . import encoding as enc


# =============================================================================
# Config + evaluator
# =============================================================================

@dataclass
class MCTSConfig:
    num_simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    pw_k: float = 1.0          # progressive widening: max children ~ k * visits^alpha
    pw_alpha: float = 0.5
    temperature: float = 1.0   # move selection from root visit counts (<=1e-3 -> greedy)
    discount: float = 1.0
    win_value_floor: float = 0.2   # win value = floor + (1-floor)*hp_frac
    add_root_noise: bool = True   # Dirichlet exploration at the root (off for eval)
    seed: Optional[int] = None


# An evaluator maps a session -> (priors over its legal_actions, scalar value in [-1,1]).
Evaluator = Callable[[CombatSession], "tuple[np.ndarray, float]"]


class StubEvaluator:
    """Uniform priors + HP-differential heuristic value. No neural net (step 1)."""

    def __call__(self, session: CombatSession):
        n = len(session.legal_actions())
        priors = np.full(n, 1.0 / n, np.float32) if n else np.zeros(0, np.float32)
        bc = session.bc
        php = bc.player.cur_hp / max(1, bc.player.max_hp)
        ehp_max = sum(m.max_hp for m in bc.monsters) or 1
        ehp = enc.enemy_total_hp(bc) / ehp_max
        return priors, float(np.clip(php - ehp, -1.0, 1.0))


# =============================================================================
# Nodes
# =============================================================================

class DecisionNode:
    __slots__ = ("session", "actions", "priors", "children",
                 "N", "expanded", "is_terminal")

    def __init__(self, session: CombatSession):
        self.session = session
        self.actions: list[LegalAction] = []
        self.priors: Optional[np.ndarray] = None
        self.children: list[Optional["ChanceNode"]] = []
        self.N = 0                      # times an edge was selected from here
        self.expanded = False
        self.is_terminal = session.done or session.truncated


class ChanceNode:
    __slots__ = ("N", "W", "outcomes")

    def __init__(self):
        self.N = 0                      # visits through this edge
        self.W = 0.0                    # summed backed-up value
        self.outcomes: dict = {}        # state_signature -> DecisionNode

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


# =============================================================================
# Search
# =============================================================================

class MCTS:
    def __init__(self, evaluator: Optional[Evaluator] = None,
                 config: Optional[MCTSConfig] = None):
        self.evaluator = evaluator or StubEvaluator()
        self.cfg = config or MCTSConfig()
        self.rng = random.Random(self.cfg.seed)
        self.np_rng = np.random.default_rng(self.cfg.seed)

    # ----- public -----------------------------------------------------------

    def run(self, session: CombatSession):
        """Search from `session`, evaluating leaves one at a time (batch-1).

        Returns (chosen_action, visit_policy, root). Drives the same generator that
        concurrent self-play batches across games -- see `search_gen`.
        """
        gen = self.search_gen(session)
        try:
            req = next(gen)
            while True:
                req = gen.send(self.evaluator(req))
        except StopIteration as done:
            return done.value

    # ----- search as a generator (yields a session needing evaluation) ------
    #
    # A leaf evaluation is a `yield session` that resumes with (priors, value). The
    # synchronous `run` feeds those one by one; concurrent self-play collects yields
    # from many games and feeds a single batched forward.

    def search_gen(self, session: CombatSession):
        root = DecisionNode(session.clone())
        yield from self._expand_gen(root, add_noise=self.cfg.add_root_noise)
        for _ in range(self.cfg.num_simulations):
            yield from self._simulate_gen(root)
        visits = np.array([c.N if c is not None else 0 for c in root.children],
                          dtype=np.float64)
        policy = self._policy_from_visits(visits)
        action = root.actions[self._select_move(visits)] if root.actions else None
        return action, policy, root

    def _simulate_gen(self, node: DecisionNode):
        if node.is_terminal:
            if node.session.done:
                return self._terminal_value(node.session)
            _, value = yield node.session          # truncated: bootstrap with evaluator
            return value
        if not node.expanded:
            value = yield from self._expand_gen(node)
            return value

        a = self._select_action(node)
        chance = node.children[a]
        if chance is None:
            chance = node.children[a] = ChanceNode()
        child = self._sample_outcome(chance, node.session, node.actions[a])

        v = self.cfg.discount * (yield from self._simulate_gen(child))
        chance.N += 1
        chance.W += v
        node.N += 1
        return v

    def _expand_gen(self, node: DecisionNode, add_noise: bool = False):
        """First visit: attach legal actions + evaluator priors, return value."""
        node.actions = node.session.legal_actions()
        priors, value = yield node.session         # <-- pause for evaluation
        priors = np.asarray(priors, dtype=np.float32)
        if add_noise and len(node.actions) > 0:
            noise = self.np_rng.dirichlet([self.cfg.dirichlet_alpha] * len(node.actions))
            eps = self.cfg.dirichlet_epsilon
            priors = (1 - eps) * priors + eps * noise.astype(np.float32)
        node.priors = priors
        node.children = [None] * len(node.actions)
        node.expanded = True
        return value

    def _sample_outcome(self, chance: ChanceNode, parent: CombatSession,
                        action: LegalAction) -> DecisionNode:
        """Draw (or reuse) a resulting state for this edge, with dedup + widening."""
        widen = len(chance.outcomes) < self.cfg.pw_k * ((chance.N + 1) ** self.cfg.pw_alpha)
        if widen or not chance.outcomes:
            child = parent.clone()
            child.determinize(self.rng.getrandbits(63))   # independent search RNG
            child.step(action)
            sig = child.state_signature()
            existing = chance.outcomes.get(sig)
            if existing is not None:
                return existing
            node = DecisionNode(child)
            chance.outcomes[sig] = node
            return node
        # widening capped: revisit an existing outcome ~ its frequency (visit count)
        nodes = list(chance.outcomes.values())
        weights = [max(1, nd.N) for nd in nodes]
        return self.rng.choices(nodes, weights=weights, k=1)[0]

    # ----- selection / values ----------------------------------------------

    def _select_action(self, node: DecisionNode) -> int:
        sqrt_total = math.sqrt(max(1, node.N))
        best_i, best_score = 0, -math.inf
        for i, ch in enumerate(node.children):
            q = ch.Q if ch is not None else 0.0
            n = ch.N if ch is not None else 0
            u = self.cfg.c_puct * float(node.priors[i]) * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_i = score, i
        return best_i

    def _terminal_value(self, session: CombatSession) -> float:
        """Value of a *decided* combat: win -> floor + (1-floor)*hp_frac, death -> -1."""
        if session.won:
            hp = max(0, session.bc.player.cur_hp) / max(1, session.cfg.max_hp)
            f = self.cfg.win_value_floor
            return f + (1.0 - f) * hp
        return -1.0

    def _game_value(self, session: CombatSession) -> float:
        """Self-play value target z for a finished game (neutral 0 if only truncated)."""
        return self._terminal_value(session) if session.done else 0.0

    # ----- move selection ---------------------------------------------------

    def _policy_from_visits(self, visits: np.ndarray) -> np.ndarray:
        total = visits.sum()
        n = len(visits)
        if n == 0:
            return visits
        if total == 0:
            return np.full(n, 1.0 / n)
        tau = self.cfg.temperature
        if tau <= 1e-3:
            pi = np.zeros(n)
            pi[int(visits.argmax())] = 1.0
            return pi
        logits = visits ** (1.0 / tau)
        return logits / logits.sum()

    def _select_move(self, visits: np.ndarray) -> int:
        if self.cfg.temperature <= 1e-3:
            return int(visits.argmax())
        pi = self._policy_from_visits(visits)
        return int(self.np_rng.choice(len(pi), p=pi))


# =============================================================================
# Convenience: play a full combat with the search
# =============================================================================

def play_combat(session: CombatSession, mcts: MCTS, verbose: bool = False) -> dict:
    """Drive a combat to completion, choosing each move by a fresh search."""
    session.reset()
    moves = 0
    while not (session.done or session.truncated):
        action, _policy, root = mcts.run(session)
        if action is None:
            break
        if verbose:
            noutcomes = sum(len(c.outcomes) for c in root.children if c is not None)
            print(f"move {moves}: {action.label}  (root visits={root.N}, "
                  f"distinct outcomes across edges={noutcomes})")
        session.step(action)
        moves += 1
    return {"moves": moves, "outcome": session.outcome,
            "won": session.won, "hp": session.bc.player.cur_hp}


if __name__ == "__main__":
    from .scenario import CombatConfig

    cfg = CombatConfig(
        deck=["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
        relics=["Burning Blood"], max_hp=80, cur_hp=80, encounter="Jaw Worm", seed=42,
    )
    mcts = MCTS(config=MCTSConfig(num_simulations=128, seed=0))

    print("== MCTS (stub evaluator) playing Jaw Worm ==")
    for ep in range(3):
        session = CombatSession(config=cfg)
        result = play_combat(session, mcts, verbose=(ep == 0))
        print(f"ep {ep}: moves={result['moves']} outcome={result['outcome']} "
              f"hp={result['hp']}")
