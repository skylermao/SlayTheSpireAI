"""Multiprocessing actor-learner self-play for combat training.

Self-play (MCTS + the C++ sim) is CPU-bound and single-threaded per game, and Python's
GIL blocks thread parallelism, so throughput scales with cores only across *processes*.
This module runs N actor processes that each generate self-play games independently and
stream training examples to one learner process, which trains the network and
periodically broadcasts updated weights back.

    ACTORS (N procs)  ──examples──▶  QUEUE  ──▶  LEARNER (net + opt + replay buffer)
         ▲                                              │
         └─────────── versioned weights file ◀──────────┘

Each actor self-plays with its own CombatNet copy (CPU), reloading the learner's latest
weights whenever the shared version counter changes. The learner drains examples into a
replay buffer and runs gradient steps continuously; self-play is the bottleneck, so the
learner reuses buffer data across steps.

Transport: a LegalAction holds a C++ `sts.Action` that doesn't pickle, so each example is
converted to a transport form whose actions are lightweight `ActionRef` tuples carrying
only the fields `CombatNet.forward_batch` reads (kind, source_idx, target_idx,
potion_id). The C++ objects stay inside the actor and never cross the process boundary.
"""

import os
import sys
import time
import queue
import random
import multiprocessing as mp
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Optional

import torch

from ..algos.selfplay import SelfPlayExample, collect_selfplay
from ..core.session import CombatSession
from ..core.scenario import DatasetSampler, CombatConfig
from ..algos.mcts import MCTS, MCTSConfig
from ..algos.net import CombatNet, NeuralEvaluator
from .train import Trainer, TrainConfig


# A picklable stand-in for a LegalAction carrying exactly what forward_batch consumes.
ActionRef = namedtuple("ActionRef", ["kind", "source_idx", "target_idx", "potion_id",
                                     "calc_damage"])


def _to_transport(ex: SelfPlayExample) -> SelfPlayExample:
    """Strip the un-picklable C++ Action from an example so it can cross to the learner."""
    acts = [ActionRef(a.kind, a.source_idx, a.target_idx, a.potion_id, a.calc_damage)
            for a in ex.actions]
    return SelfPlayExample(obs=ex.obs, actions=acts, sel_feats=ex.sel_feats,
                           policy=ex.policy, z=ex.z)


@dataclass
class ParallelConfig:
    num_actors: int = max(1, (os.cpu_count() or 2) - 1)   # leave a core for the learner
    actor_concurrency: int = 8       # games in flight per actor (intra-actor eval batching)
    actor_chunk: int = 8             # games per work unit (then check weights / stop flag)
    max_fights: int = 300_000        # per-actor dataset cap (caps memory under spawn)
    total_steps: int = 100_000       # learner gradient steps to run
    min_buffer: int = 2_000          # start training once the buffer holds this many examples
    weight_sync_steps: int = 200     # push fresh weights to actors every N grad steps
    queue_chunks: int = 64           # bounded example queue (backpressure), in chunks
    log_every: int = 100             # log a line every N grad steps
    checkpoint_every: int = 2_000    # save a checkpoint every N grad steps
    device: str = "cpu"              # learner + actor device (CPU on a c7i box)
    tensorboard: bool = False        # also log scalars to TensorBoard
    tb_logdir: Optional[str] = None  # defaults to <checkpoint_dir>/tb


# =============================================================================
# Actor process
# =============================================================================

def _actor_loop(actor_id, fights, config_kw, net_kwargs, mcts_cfg, weights_path,
                version, example_q, stop_event, base_seed, device,
                actor_chunk, actor_concurrency, max_turns):
    """One self-play worker: generate games, stream transport examples to the learner."""
    torch.set_num_threads(1)                       # one BLAS thread; parallelism is across procs
    seed = base_seed + actor_id * 100_003
    rng = random.Random(seed)

    if fights is not None:
        sampler = DatasetSampler(fights, rng=rng)
        def make_session():
            return CombatSession(sampler=sampler, max_turns=max_turns)
    else:
        cfg = CombatConfig(**config_kw)
        def make_session():
            return CombatSession(config=cfg, max_turns=max_turns)

    net = CombatNet(**net_kwargs)
    evaluator = NeuralEvaluator(net, device=device)
    mcts = MCTS(evaluator=evaluator, config=replace(mcts_cfg, seed=seed))
    local_version = -1

    def maybe_reload_weights():
        nonlocal local_version
        v = version.value
        if v != local_version and os.path.exists(weights_path):
            try:
                net.load_state_dict(torch.load(weights_path, map_location=device))
                local_version = v
            except Exception:
                pass                                # mid-write race: try again next chunk

    maybe_reload_weights()
    while not stop_event.is_set():
        maybe_reload_weights()
        results = collect_selfplay(mcts, make_session, actor_chunk, actor_concurrency)
        batch = [_to_transport(ex) for r in results for ex in r.examples]
        stats = {"games": len(results), "wins": sum(r.won for r in results),
                 "hp": sum(r.hp for r in results), "moves": sum(r.moves for r in results)}
        if not batch:
            continue
        while not stop_event.is_set():             # block on backpressure, but stay killable
            try:
                example_q.put((batch, stats), timeout=0.5)
                break
            except queue.Full:
                continue


# =============================================================================
# Learner (main process)
# =============================================================================

def _plain_config(cfg: CombatConfig) -> dict:
    """A pickle-safe CombatConfig kwargs dict (enums -> names) for actor reconstruction."""
    def name(x):
        return x.name if hasattr(x, "name") else x
    return dict(deck=list(cfg.deck), encounter=name(cfg.encounter),
                max_hp=cfg.max_hp, cur_hp=cfg.cur_hp, relics=list(cfg.relics),
                potions=[name(p) for p in cfg.potions], ascension=cfg.ascension,
                seed=cfg.seed, human_hp_loss=cfg.human_hp_loss)


class ParallelTrainer:
    """Actor-learner AlphaZero training. Point it at the fights gzip (or a CombatConfig)."""

    def __init__(self, data_path: Optional[str] = None,
                 config: Optional[CombatConfig] = None,
                 train_config: Optional[TrainConfig] = None,
                 mcts_config: Optional[MCTSConfig] = None,
                 parallel_config: Optional[ParallelConfig] = None,
                 net_kwargs: Optional[dict] = None, seed: int = 0,
                 resume_from: Optional[str] = None,
                 exclude_encounters: Optional[set] = None,
                 include_encounters: Optional[set] = None,
                 max_turns: int = 60):
        if data_path is None and config is None:
            raise ValueError("Provide data_path (fights gzip) and/or a CombatConfig")
        self.max_turns = max_turns
        self.tcfg = train_config or TrainConfig()
        # Self-play search regime, taken from the config (MCTSConfig defaults already
        # explore: temperature=1.0, add_root_noise=True). Respecting the config -- rather
        # than forcing it -- lets temperature / noise be tuned. (Eval builds its own
        # greedy, no-noise MCTS separately, so it is unaffected.)
        self.mcfg = mcts_config or MCTSConfig()
        self.pcfg = parallel_config or ParallelConfig()
        self.net_kwargs = net_kwargs or {}
        self.seed = seed

        # Load + subsample the dataset once; actors inherit it (free under fork).
        self._fights = None
        self._config_kw = None
        learner_sampler = None
        if data_path is not None:
            fights = DatasetSampler.from_gzip(data_path, rng=random.Random(seed)).fights
            if include_encounters or exclude_encounters:
                n0 = len(fights)
                if include_encounters:
                    fights = [f for f in fights if f["enemies"] in include_encounters]
                if exclude_encounters:
                    fights = [f for f in fights if f["enemies"] not in exclude_encounters]
                print(f"[parallel] encounter filter: kept {len(fights):,}/{n0:,} fights",
                      flush=True)
            if self.pcfg.max_fights and len(fights) > self.pcfg.max_fights:
                fights = random.Random(seed).sample(fights, self.pcfg.max_fights)
            self._fights = fights
            learner_sampler = DatasetSampler(fights, rng=random.Random(seed + 1))
        else:
            self._config_kw = _plain_config(config)

        # Reuse Trainer for the net / optimizer / replay buffer / loss / checkpointing.
        self.trainer = Trainer(config=config, sampler=learner_sampler,
                               train_config=self.tcfg, mcts_config=self.mcfg,
                               net=CombatNet(**self.net_kwargs), device=self.pcfg.device)
        if resume_from is not None:
            self.trainer.load(resume_from)              # net + optimizer state
            for g in self.trainer.opt.param_groups:     # loading opt restores old lr; re-assert ours
                g["lr"] = self.tcfg.lr
            print(f"[parallel] resumed from {resume_from} (lr={self.tcfg.lr})", flush=True)
        os.makedirs(self.tcfg.checkpoint_dir, exist_ok=True)
        self.weights_path = os.path.join(self.tcfg.checkpoint_dir, "actor_weights.pt")

    # ----- weight broadcast -------------------------------------------------

    def _push_weights(self):
        """Atomically publish the learner's weights for actors to pick up."""
        sd = {k: v.detach().cpu() for k, v in self.trainer.net.state_dict().items()}
        tmp = f"{self.weights_path}.tmp.{os.getpid()}"
        torch.save(sd, tmp)
        os.replace(tmp, self.weights_path)          # atomic on POSIX -> no torn reads
        self._version.value += 1

    # ----- one gradient step ------------------------------------------------

    def _train_step(self):
        self.trainer.net.train()
        batch = random.sample(self.trainer.buffer, self.tcfg.batch_size)
        loss, p, v = self.trainer._loss(batch)
        if not torch.isfinite(loss):                        # backstop against a bad batch
            self.trainer.opt.zero_grad(set_to_none=True)
            return float(p), float(v), True
        self.trainer.opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.trainer.net.parameters(), self.tcfg.grad_clip)
        if not torch.isfinite(gnorm):                       # don't apply a non-finite update
            self.trainer.opt.zero_grad(set_to_none=True)
            return float(p), float(v), True
        self.trainer.opt.step()
        return float(p), float(v), False

    # ----- main loop --------------------------------------------------------

    def run(self):
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        ctx = mp.get_context(method)
        self._version = ctx.Value("i", 0)
        example_q = ctx.Queue(maxsize=self.pcfg.queue_chunks)
        stop_event = ctx.Event()

        self.trainer.net.eval()
        self._push_weights()                        # publish initial weights (version 1)

        writer = None
        if self.pcfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            logdir = self.pcfg.tb_logdir or os.path.join(self.tcfg.checkpoint_dir, "tb")
            writer = SummaryWriter(logdir)
            print(f"[parallel] tensorboard logdir: {logdir}", flush=True)

        actor_mcfg = replace(self.mcfg, seed=None)  # per-actor seed set inside the worker
        actors = []
        for i in range(self.pcfg.num_actors):
            p = ctx.Process(target=_actor_loop, args=(
                i, self._fights, self._config_kw, self.net_kwargs, actor_mcfg,
                self.weights_path, self._version, example_q, stop_event,
                self.seed + 1, self.pcfg.device,
                self.pcfg.actor_chunk, self.pcfg.actor_concurrency,
                self.max_turns), daemon=True)
            p.start()
            actors.append(p)
        print(f"[parallel] {self.pcfg.num_actors} actors x {self.pcfg.actor_concurrency} "
              f"games (start={method}, device={self.pcfg.device})", flush=True)

        steps = ex_seen = games_seen = wins = hp_sum = moves_sum = skips = 0
        prev_games = prev_wins = 0
        t0 = last = time.time()
        try:
            while steps < self.pcfg.total_steps:
                # Drain available example chunks into the replay buffer (don't starve train).
                drained = 0
                while drained < 20_000:
                    try:
                        batch, st = example_q.get(timeout=0.05)
                    except queue.Empty:
                        break
                    self.trainer.buffer.extend(batch)
                    ex_seen += len(batch); drained += len(batch)
                    games_seen += st["games"]; wins += st["wins"]
                    hp_sum += st["hp"]; moves_sum += st["moves"]

                if len(self.trainer.buffer) < max(self.pcfg.min_buffer, self.tcfg.batch_size):
                    time.sleep(0.05)
                    continue

                p, v, skipped = self._train_step()
                steps += 1
                skips += int(skipped)
                if steps % self.pcfg.weight_sync_steps == 0:
                    self._push_weights()
                if steps % self.pcfg.log_every == 0:
                    now = time.time()
                    sps = self.pcfg.log_every / max(1e-9, now - last)
                    gps = games_seen / max(1e-9, now - t0)
                    wr = wins / max(1, games_seen)
                    iwr = (wins - prev_wins) / max(1, games_seen - prev_games)  # recent window
                    print(f"step {steps:6d} | {sps:5.1f} steps/s | self-play {gps:5.2f} games/s "
                          f"winrate {wr:.2f} | buf {len(self.trainer.buffer)} ex {ex_seen} "
                          f"| ploss {p:.3f} vloss {v:.3f} skips {skips}", flush=True)
                    if writer is not None:
                        writer.add_scalar("train/skipped_steps", skips, steps)
                        writer.add_scalar("loss/policy", p, steps)
                        writer.add_scalar("loss/value", v, steps)
                        writer.add_scalar("winrate/cumulative", wr, steps)
                        writer.add_scalar("winrate/recent", iwr, steps)
                        writer.add_scalar("throughput/games_per_s", gps, steps)
                        writer.add_scalar("throughput/steps_per_s", sps, steps)
                        writer.add_scalar("buffer/size", len(self.trainer.buffer), steps)
                        writer.add_scalar("games/total", games_seen, steps)
                    prev_games, prev_wins = games_seen, wins
                    last = now
                if steps % self.pcfg.checkpoint_every == 0:
                    self.trainer.save(os.path.join(self.tcfg.checkpoint_dir, f"net_step{steps}.pt"))
                    self._push_weights()
        except KeyboardInterrupt:
            print("\n[parallel] interrupted; shutting down actors...", flush=True)
        finally:
            stop_event.set()
            # Unblock any actor parked in queue.put, then reap workers.
            try:
                while True:
                    example_q.get_nowait()
            except queue.Empty:
                pass
            for p in actors:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            self.trainer.save(os.path.join(self.tcfg.checkpoint_dir, "net_final.pt"))
            if writer is not None:
                writer.close()
            print(f"[parallel] done: {steps} steps, {games_seen} games, "
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    # Smoke test: a short parallel run on the wins dataset (CPU).
    torch.manual_seed(0)
    pcfg = ParallelConfig(num_actors=3, actor_concurrency=6, actor_chunk=4,
                          max_fights=20_000, total_steps=60, min_buffer=300,
                          weight_sync_steps=20, log_every=10, checkpoint_every=10_000,
                          device="cpu")
    tcfg = TrainConfig(batch_size=64, checkpoint_dir="/tmp/parallel_smoke")
    trainer = ParallelTrainer(data_path="data/ironclad_a0_fights.json.gz",
                              train_config=tcfg,
                              mcts_config=MCTSConfig(num_simulations=24),
                              parallel_config=pcfg)
    trainer.run()
    print("parallel smoke test ok")
