"""AlphaZero training loop for combat (step 3): self-play -> train -> repeat.

Single-GPU, single-process. Each iteration:
  1. SELF-PLAY: run a pool of games concurrently (batched eval), append the resulting
     (obs, legal actions, MCTS policy pi, value z) examples to a replay buffer.
  2. TRAIN: sample minibatches and minimize  CE(pi, net policy) + c * MSE(z, net value).
  3. Periodically checkpoint and evaluate (greedy, no root noise) vs. win-rate / HP.

The self-play evaluator and the trained network are the *same* nn.Module, so each
self-play round uses the latest weights. net.eval()/train() toggles between phases.
"""

import os
import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .session import CombatSession
from .scenario import CombatConfig, DatasetSampler
from .mcts import MCTS, MCTSConfig
from .net import CombatNet, NeuralEvaluator
from .selfplay import collect_selfplay, SelfPlayExample


@dataclass
class TrainConfig:
    iterations: int = 40
    games_per_iter: int = 32
    max_concurrent: int = 16
    train_steps_per_iter: int = 64
    batch_size: int = 256
    buffer_capacity: int = 50_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    value_loss_coef: float = 1.0
    grad_clip: float = 1.0
    eval_every: int = 5
    eval_games: int = 20
    checkpoint_dir: str = "checkpoints"
    seed: int = 0


class Trainer:
    def __init__(self, config: Optional[CombatConfig] = None,
                 sampler: Optional[DatasetSampler] = None,
                 train_config: Optional[TrainConfig] = None,
                 mcts_config: Optional[MCTSConfig] = None,
                 net: Optional[CombatNet] = None,
                 device: Optional[str] = None):
        if config is None and sampler is None:
            raise ValueError("Provide a CombatConfig and/or a DatasetSampler")
        self.config = config
        self.sampler = sampler
        self.tcfg = train_config or TrainConfig()

        self.net = net or CombatNet()
        self.evaluator = NeuralEvaluator(self.net, device=device)
        self.device = self.evaluator.device

        base_mcts = mcts_config or MCTSConfig()
        # Self-play: explore (root noise + temperature sampling).
        self.play_mcts = MCTS(self.evaluator,
                              replace(base_mcts, temperature=1.0, add_root_noise=True,
                                      seed=self.tcfg.seed))
        # Eval: greedy, no exploration noise.
        self.eval_mcts = MCTS(self.evaluator,
                              replace(base_mcts, temperature=0.0, add_root_noise=False,
                                      seed=self.tcfg.seed + 1))

        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.tcfg.lr,
                                    weight_decay=self.tcfg.weight_decay)
        self.buffer: "deque[SelfPlayExample]" = deque(maxlen=self.tcfg.buffer_capacity)
        self.rng = random.Random(self.tcfg.seed)
        os.makedirs(self.tcfg.checkpoint_dir, exist_ok=True)

    def make_session(self) -> CombatSession:
        return CombatSession(config=self.config, sampler=self.sampler)

    # ----- phases -----------------------------------------------------------

    def self_play(self) -> dict:
        self.net.eval()
        results = collect_selfplay(self.play_mcts, self.make_session,
                                   self.tcfg.games_per_iter, self.tcfg.max_concurrent)
        for r in results:
            self.buffer.extend(r.examples)
        wins = sum(r.won for r in results)
        return {"games": len(results), "wins": wins,
                "examples": sum(len(r.examples) for r in results),
                "avg_hp": np.mean([r.hp for r in results])}

    def _loss(self, examples: "list[SelfPlayExample]"):
        t = self.evaluator._stack([e.obs for e in examples])
        actions_batch = [e.actions for e in examples]
        sel_list = [torch.as_tensor(e.sel_feats, dtype=torch.float32, device=self.device)
                    for e in examples]
        logits_list, values = self.net.forward_batch(t, actions_batch, sel_list)

        z = torch.tensor([e.z for e in examples], dtype=torch.float32, device=self.device)
        value_loss = F.mse_loss(values, z)

        policy_loss = values.new_zeros(())
        for logits, e in zip(logits_list, examples):
            pi = torch.as_tensor(e.policy, dtype=torch.float32, device=self.device)
            policy_loss = policy_loss - (pi * F.log_softmax(logits, dim=-1)).sum()
        policy_loss = policy_loss / len(examples)

        loss = policy_loss + self.tcfg.value_loss_coef * value_loss
        return loss, policy_loss.detach(), value_loss.detach()

    def train(self) -> dict:
        if len(self.buffer) < self.tcfg.batch_size:
            return {"trained": False}
        self.net.train()
        pl = vl = 0.0
        for _ in range(self.tcfg.train_steps_per_iter):
            batch = self.rng.sample(self.buffer, self.tcfg.batch_size)
            loss, p, v = self._loss(batch)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
            self.opt.step()
            pl += float(p); vl += float(v)
        n = self.tcfg.train_steps_per_iter
        return {"trained": True, "policy_loss": pl / n, "value_loss": vl / n}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.net.eval()
        results = collect_selfplay(self.eval_mcts, self.make_session,
                                   self.tcfg.eval_games, self.tcfg.max_concurrent)
        return {"eval_winrate": float(np.mean([r.won for r in results])),
                "eval_avg_hp": float(np.mean([r.hp for r in results]))}

    # ----- loop -------------------------------------------------------------

    def run(self):
        for it in range(1, self.tcfg.iterations + 1):
            sp = self.self_play()
            tr = self.train()
            msg = (f"iter {it:3d} | games {sp['games']} winrate {sp['wins']/sp['games']:.2f} "
                   f"hp {sp['avg_hp']:.0f} buf {len(self.buffer)}")
            if tr["trained"]:
                msg += f" | ploss {tr['policy_loss']:.3f} vloss {tr['value_loss']:.3f}"
            if it % self.tcfg.eval_every == 0:
                ev = self.evaluate()
                msg += f" || EVAL winrate {ev['eval_winrate']:.2f} hp {ev['eval_avg_hp']:.0f}"
                self.save(os.path.join(self.tcfg.checkpoint_dir, f"net_iter{it}.pt"))
            print(msg, flush=True)

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CombatConfig(deck=["Strike_R"] * 5 + ["Defend_R"] * 4 + ["Bash"],
                       relics=["Burning Blood"], max_hp=80, cur_hp=80,
                       encounter="Jaw Worm", seed=None)
    # Tiny smoke config: validate the loop runs and losses move.
    tcfg = TrainConfig(iterations=4, games_per_iter=8, max_concurrent=8,
                       train_steps_per_iter=20, batch_size=64, eval_every=2, eval_games=8)
    trainer = Trainer(config=cfg, train_config=tcfg,
                      mcts_config=MCTSConfig(num_simulations=32))
    print(f"device={trainer.device}  params={sum(p.numel() for p in trainer.net.parameters())}")
    trainer.run()
    print("training loop smoke test ok")
