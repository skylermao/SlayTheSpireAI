"""Greedy evaluation of a trained combat net: the true (no-exploration) winrate.

Self-play winrate during training is *understated* because the agent explores (Dirichlet
root noise + temperature sampling). This plays dataset combats with GREEDY MCTS
(temperature=0, no root noise) -- the agent's actual playing strength -- and reports
winrate with a 95% confidence interval, average end HP, and average game length.

    python -m rl.combat.eval --ckpt checkpoints_v2/net_final.pt \
        --data data/ironclad_a0_fights.json.gz --games 300 --sims 128

The checkpoint must match the current network architecture (LayerNorm + bounded logits).
"""

import argparse
import random
import time

import numpy as np
import torch

from .scenario import DatasetSampler
from .session import CombatSession
from .mcts import MCTS, MCTSConfig
from .net import CombatNet, NeuralEvaluator
from .selfplay import collect_selfplay


def evaluate(ckpt: str, data_path: str, n_games: int = 300, sims: int = 128,
             concurrency: int = 16, device=None, seed: int = 0,
             max_fights=None, net_kwargs=None, exclude_encounters=None) -> dict:
    net = CombatNet(**(net_kwargs or {}))
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd.get("net", sd))          # accept {"net":..} or a bare state_dict
    net.eval()
    evaluator = NeuralEvaluator(net, device=device)
    # Greedy: pick the most-visited move, no exploration noise.
    mcts = MCTS(evaluator, MCTSConfig(num_simulations=sims, temperature=0.0,
                                      add_root_noise=False, seed=seed))

    fights = DatasetSampler.from_gzip(data_path, rng=random.Random(seed)).fights
    if exclude_encounters:
        fights = [f for f in fights if f["enemies"] not in exclude_encounters]
    if max_fights and len(fights) > max_fights:
        fights = random.Random(seed).sample(fights, max_fights)
    sampler = DatasetSampler(fights, rng=random.Random(seed))

    results = collect_selfplay(mcts, lambda: CombatSession(sampler=sampler),
                               n_games, concurrency)
    n = len(results)
    wins = sum(r.won for r in results)
    wr = wins / n
    se = (wr * (1 - wr) / n) ** 0.5
    return {"games": n, "wins": wins, "winrate": wr, "ci95": 1.96 * se,
            "avg_end_hp": float(np.mean([r.hp for r in results])),
            "avg_moves": float(np.mean([r.moves for r in results]))}


def main():
    ap = argparse.ArgumentParser(description="Greedy winrate eval for a combat net.")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt path")
    ap.add_argument("--data", default="data/ironclad_a0_fights.json.gz")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fights", type=int, default=0, help="0 = use all fights")
    ap.add_argument("--normal-only", action="store_true",
                    help="exclude elite/boss encounters (match normal-only training)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    exclude = None
    if args.normal_only:
        from .scenario import NON_NORMAL_ENCOUNTERS
        exclude = NON_NORMAL_ENCOUNTERS

    torch.set_num_threads(1)                        # so parallel shards don't oversubscribe
    t0 = time.time()
    r = evaluate(args.ckpt, args.data, args.games, args.sims, args.concurrency,
                 args.device, args.seed, args.max_fights or None,
                 exclude_encounters=exclude)
    dt = time.time() - t0
    print(f"RESULT ckpt={args.ckpt} seed={args.seed} games={r['games']} wins={r['wins']} "
          f"winrate={r['winrate']:.4f} ci95=+/-{r['ci95']:.4f} "
          f"avg_end_hp={r['avg_end_hp']:.1f} avg_moves={r['avg_moves']:.1f} "
          f"secs={dt:.0f}", flush=True)


if __name__ == "__main__":
    main()
