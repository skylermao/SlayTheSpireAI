# SlayTheSpireAI (Work in Progress)

An AlphaZero-style reinforcement learning agent for **Slay the Spire combat** (Ironclad),
built on the RNG-accurate [`sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed)
C++ simulator. The agent learns to play individual combats — given a deck, relics,
potions, entering HP, and an encounter — by self-play with Monte Carlo Tree Search guided
by a neural network.

This is **combat-only** RL: deck-building and map navigation are out of scope. Training
scenarios are drawn from real human winning-run data (per-combat snapshots).

## How it works

A standard AlphaZero loop specialized to single-player, stochastic combat:

- **MCTS** (`mcts.py`) searches from the current state using PUCT, with the network
  providing action priors and a state value. Stochastic transitions (enemy moves, card
  draws, reshuffles) are handled with **chance nodes** — sampled, deduped by an
  information-set state signature, and grown by progressive widening. Hidden state is
  *determinized* per search rollout with an independent RNG, so the search plans over the
  agent's information set rather than peeking at the real draw.
- **Network** (`net.py`, `CombatNet`): a permutation-invariant Deep-Sets encoder over the
  board state plus a **per-action scoring head** (each legal move is scored from
  `[state, action-kind, item, target]`) and a tanh value head. Action logits are bounded
  (`±logit_scale`) and the trunk uses LayerNorm for training stability.
- **Self-play → train** loop: games are played with exploration (Dirichlet root noise +
  temperature); each move records `(observation, legal actions, MCTS visit policy π)`, and
  at game end a value target `z`. The net is trained on `CE(π) + MSE(z)` with target
  smoothing and weight decay.

### Value target
`z` is HP-based and difficulty-aware:
- **Win** → `win_value_floor + (1 - win_value_floor)·hp_fraction` (in `[0.2, 1.0]`).
- **Loss** → scaled by how hard the real fight was (fraction of entering HP the human
  lost): from `-1.0` for an easy fight down to `-0.3` for one where the human nearly died.
  Losing a brutal fight is penalized less than losing an easy one.

### What the network sees (observation)
Fixed-shape tensors (~974 floats) plus per-action features:
- **Cards**: hand (per-card features), and draw/discard/exhaust piles as card-id count
  vectors. Card features = dense id, cost, upgraded, ethereal, exhausts, requires-target,
  type, special data.
- **Player**: HP/block/energy, per-turn play counts, persistent relic counters
  (Pen Nib, Nunchaku, Ink Bottle, Sundial, …), and 36 status stacks.
- **Monsters**: per-enemy HP/block, intent damage/hits/is-attacking, a **current intent
  move embedding**, and 31 status stacks; plus an alive/targetable mask.
- **Relics**: multi-hot presence over all relics. **Potions**: held-potion count vector.
- **Card-select task** id (so the net knows upgrade- vs exhaust- vs top-of-draw selects).
- **Scalars**: turn, card-select flags, potion/enemy counts.

## Repository layout

```
sts_rl/
├── rl/combat/              # the RL agent (Python)
│   ├── _sts.py             # single import point for the compiled sim
│   ├── scenario.py         # CombatConfig, DatasetSampler, name→enum resolvers,
│   │                       #   potion sampling, encounter difficulty tiers
│   ├── encoding.py         # BattleContext → observation tensors
│   ├── session.py          # CombatSession: the interface to one live combat
│   ├── mcts.py             # AlphaZero MCTS over CombatSession
│   ├── net.py              # CombatNet + NeuralEvaluator
│   ├── selfplay.py         # concurrent self-play (batched leaf eval)
│   ├── train.py            # single-process Trainer
│   ├── parallel.py         # multiprocessing actor–learner ParallelTrainer
│   └── eval.py             # greedy (no-exploration) winrate evaluation
├── sts_lightspeed/         # C++ simulator + pybind11 bindings (vendored)
└── data/                   # extracted per-combat training data (gitignored)
```

## Setup

Requires CMake ≥ 3.19, a C++17 compiler, and Python 3.11/3.12 (with a PyTorch wheel).

```bash
# 1. build the simulator + Python bindings
cd sts_lightspeed && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)        # produces slaythespire.<py>.so

# 2. Python deps
pip install torch numpy        # CPU wheel is fine; the net is small

# 3. sanity check
cd ../.. && PYTHONPATH=. python -c "import rl.combat as c; print('ok')"
```

Build `Release` — the simulator speed dominates self-play throughput.

## Training

Self-play is CPU-bound and parallelizes across processes. `ParallelTrainer` runs N actor
processes generating games and one learner process training + broadcasting weights.

```python
from rl.combat.parallel import ParallelTrainer, ParallelConfig
from rl.combat.train import TrainConfig
from rl.combat.mcts import MCTSConfig
from rl.combat.scenario import NON_NORMAL_ENCOUNTERS

ParallelTrainer(
    data_path="data/ironclad_a0_fights.json.gz",
    train_config=TrainConfig(checkpoint_dir="checkpoints"),
    mcts_config=MCTSConfig(num_simulations=128),
    parallel_config=ParallelConfig(num_actors=30, tensorboard=True),
    net_kwargs={"d_model": 128},
    exclude_encounters=NON_NORMAL_ENCOUNTERS,   # optional: skip elites/bosses
).run()
```

- **Data**: `data/ironclad_a0_fights.json.gz` — ~1.8M per-combat snapshots from Ironclad
  A0 winning runs (deck, relics, enemy, entering HP). Potions are sampled per combat
  (80% none / 15% one / 5% two) since the data has none.
- **TensorBoard**: scalars (loss, winrate, throughput) under `<checkpoint_dir>/tb`.
- **Resume**: `ParallelTrainer(..., resume_from="checkpoints/net_step40000.pt")`.

Single-process training (no multiprocessing) is also available via `train.Trainer`.

## Evaluation

`eval.py` reports the **greedy** winrate (no exploration noise) — the agent's true
strength, which is higher than the self-play number reported during training.

```bash
PYTHONPATH=. python -m rl.combat.eval \
    --ckpt checkpoints/net_final.pt --games 300 --sims 128
```

## Notes

- `sts_lightspeed` is RNG-accurate to the game; currently Ironclad + colorless cards.
- The agent's input is read-only: the same encoding works on the live state and on
  hypothetical clones used by the search.
- Checkpoints bundle both network and optimizer state. A change to the observation layout
  or network architecture makes prior checkpoints unloadable (a fresh run is required).
