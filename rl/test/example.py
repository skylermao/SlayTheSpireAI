"""Example training/eval loops over CombatSession.

Run:  python -m rl.test.example     (from the repo root)

Shows the two ways a hand-written algorithm talks to the sim:
  1. on-policy rollout  -- reset / legal_actions / step  (what PPO collects)
  2. afterstate greedy  -- peek every legal action, take the best  (sasRL-style)
"""

import random

from ..core.session import CombatSession
from ..core.scenario import CombatConfig


def random_rollout(session: CombatSession, rng: random.Random) -> dict:
    """One episode picking uniformly among legal actions -- the basic step loop."""
    session.reset()
    total_reward = 0.0
    steps = 0
    while not (session.done or session.truncated):
        legal = session.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)            # <-- your policy goes here
        t = session.step(action)              # obs/reward/done flow back out
        total_reward += t.reward
        steps += 1
    return {"steps": steps, "reward": total_reward,
            "outcome": session.outcome, "hp": session.bc.player.cur_hp}


def afterstate_greedy(session: CombatSession) -> dict:
    """One episode scoring each action by its afterstate (sasRL flavour).

    Here the 'value' of a next-state is a hand-coded heuristic; swap in V(obs) from a
    network and this is exactly the afterstate value-iteration policy.
    """
    def value(s: CombatSession) -> float:
        # crude: prefer killing enemies and keeping HP / block.
        from ..core import encoding as enc
        return (s.bc.player.cur_hp + s.bc.player.block
                - enc.enemy_total_hp(s.bc) + (1000 if s.won else 0))

    session.reset()
    total_reward = 0.0
    steps = 0
    while not (session.done or session.truncated):
        legal = session.legal_actions()
        if not legal:
            break
        # score each action by the state it leads to (peek = clone+apply, no mutation)
        best = max(legal, key=lambda a: value(session.afterstate(a)))
        t = session.step(best)
        total_reward += t.reward
        steps += 1
    return {"steps": steps, "reward": total_reward,
            "outcome": session.outcome, "hp": session.bc.player.cur_hp}


if __name__ == "__main__":
    cfg = CombatConfig(
        deck=["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
        relics=["Burning Blood"],
        max_hp=80,
        cur_hp=80,
        encounter="Jaw Worm",
        seed=42,
    )
    session = CombatSession(config=cfg)
    rng = random.Random(0)

    print("== random rollouts ==")
    for ep in range(3):
        r = random_rollout(session, rng)
        print(f"ep {ep}: steps={r['steps']} reward={r['reward']:.1f} "
              f"outcome={r['outcome']} hp={r['hp']}")

    print("== afterstate-greedy rollouts ==")
    for ep in range(3):
        r = afterstate_greedy(session)
        print(f"ep {ep}: steps={r['steps']} reward={r['reward']:.1f} "
              f"outcome={r['outcome']} hp={r['hp']}")

    session.close()
    print("ok")
