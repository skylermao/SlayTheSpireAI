"""Single source of truth for every tunable hyperparameter.

The knobs that govern the agent live in four different places (the network constructor,
MCTSConfig, TrainConfig, ParallelConfig) plus a couple of loose values. This module
gathers all of them into one `Hyperparams` dataclass -- grouped, commented, with current
defaults -- so you can see and tune them in one spot, then build the sub-configs the
pipeline expects via `.net_kwargs() / .mcts_config() / .train_config() / .parallel_config()`.

    from rl.combat.hyperparams import Hyperparams
    hp = Hyperparams(num_simulations=256, lr=3e-4)   # override what you want
    hp.describe()                                     # print every value, grouped
    trainer = hp.build_trainer()                      # ParallelTrainer wired from hp
    trainer.run()

Only the knobs the *parallel* (production) pipeline actually reads are included. Defaults
here mirror the live defaults in the underlying dataclasses.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Hyperparams:
    # ===================== Network architecture (net.CombatNet) =====================
    d_model: int = 128          # width of the shared state/action representation
    d_card: int = 32            # card embedding dim (hand, piles, select candidates)
    d_kind: int = 16            # action-kind embedding dim; also card-select task embed
    d_potion: int = 16          # potion embedding dim; also relic presence embed
    d_move: int = 16            # monster intent-move embedding dim
    logit_scale: float = 10.0   # hard cap on |policy logit| (tanh) -> CE can't blow up

    # ===================== MCTS search (mcts.MCTSConfig) =====================
    num_simulations: int = 128  # tree simulations per move (linear cost; play strength)
    c_puct: float = 1.5         # PUCT exploration constant (prior vs value tradeoff)
    pw_k: float = 1.0           # progressive widening: max chance children ~ k*N^pw_alpha
    pw_alpha: float = 0.5       # progressive widening exponent
    discount: float = 0.99      # <1 rewards closing fights sooner (anti-turtle)

    # ----- exploration (self-play only; eval forces greedy/no-noise) -----
    temperature: float = 1.0        # move sampling from visit counts (<=1e-3 -> greedy)
    dirichlet_alpha: float = 0.3    # Dirichlet root-noise concentration
    dirichlet_epsilon: float = 0.25 # root-noise mixing weight
    add_root_noise: bool = True     # root exploration noise on (off automatically for eval)

    # ===================== Value target / reward shaping (mcts.MCTSConfig) =====================
    win_value_floor: float = 0.2    # win value = floor + (1-floor)*hp_frac  -> [0.2, 1.0]
    loss_value_easy: float = -1.0   # loss value when the human lost ~no HP (easy fight)
    loss_value_hard: float = -0.3   # loss value when the human nearly died (hard fight)
    hp_loss_coef: float = 0.5       # scale of the per-step HP shaping (0 -> terminal-only)
    reward_w: float = 0.25          # HP-shaping tilt toward low-HP losses (0 -> pure linear)
    reward_k: float = 3.0           # sharpness of that near-death tilt
    max_turns: int = 60             # turns before a combat is truncated (z=0, undecided)

    # ===================== Optimization (train.TrainConfig) =====================
    lr: float = 1e-3                    # Adam learning rate
    weight_decay: float = 1e-4          # Adam L2 weight decay
    batch_size: int = 256               # gradient-step minibatch size
    grad_clip: float = 1.0              # global grad-norm clip
    value_loss_coef: float = 1.0        # weight of MSE(value) vs CE(policy) in the loss
    policy_target_smoothing: float = 0.03  # uniform floor on policy targets (anti one-hot)
    buffer_capacity: int = 100_000      # replay buffer size (FIFO)
    min_buffer: int = 5_000             # start training once buffer holds this many examples

    # ===================== Parallel / run control (parallel.ParallelConfig) =====================
    num_actors: int = 30            # self-play worker processes (~vCPUs - learner)
    actor_concurrency: int = 8      # games in flight per actor (intra-actor eval batch)
    actor_chunk: int = 8            # games per work unit before checking weights/stop
    total_steps: int = 200_000      # learner gradient steps to run
    weight_sync_steps: int = 200    # push fresh weights to actors every N steps
    queue_chunks: int = 64          # bounded example-queue depth (backpressure)
    log_every: int = 100            # log + TensorBoard scalar cadence (steps)
    checkpoint_every: int = 2_000   # checkpoint cadence (steps)
    device: str = "cpu"             # "cpu" / "cuda" / "mps"
    tensorboard: bool = True        # log scalars to TensorBoard

    # ===================== Data / scenario =====================
    data_path: str = "data/ironclad_a0_fights.json.gz"
    normal_only: bool = True        # exclude elite/boss encounters from training
    max_fights: int = 300_000       # cap on dataset size loaded (memory)
    checkpoint_dir: str = "checkpoints"
    tb_logdir: Optional[str] = None # defaults to <checkpoint_dir>/tb
    seed: int = 0

    # --------------------------------------------------------------------------
    # Builders: turn this single config into the sub-configs the pipeline expects.
    # --------------------------------------------------------------------------

    def net_kwargs(self) -> dict:
        return dict(d_model=self.d_model, d_card=self.d_card, d_kind=self.d_kind,
                    d_potion=self.d_potion, d_move=self.d_move, logit_scale=self.logit_scale)

    def mcts_config(self):
        from .mcts import MCTSConfig
        return MCTSConfig(
            num_simulations=self.num_simulations, c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha, dirichlet_epsilon=self.dirichlet_epsilon,
            pw_k=self.pw_k, pw_alpha=self.pw_alpha, temperature=self.temperature,
            discount=self.discount, hp_loss_coef=self.hp_loss_coef,
            reward_w=self.reward_w, reward_k=self.reward_k,
            win_value_floor=self.win_value_floor, loss_value_easy=self.loss_value_easy,
            loss_value_hard=self.loss_value_hard, add_root_noise=self.add_root_noise,
            seed=self.seed)

    def train_config(self):
        from .train import TrainConfig
        return TrainConfig(
            batch_size=self.batch_size, buffer_capacity=self.buffer_capacity,
            lr=self.lr, weight_decay=self.weight_decay,
            value_loss_coef=self.value_loss_coef, grad_clip=self.grad_clip,
            policy_target_smoothing=self.policy_target_smoothing,
            checkpoint_dir=self.checkpoint_dir, seed=self.seed)

    def parallel_config(self):
        from .parallel import ParallelConfig
        return ParallelConfig(
            num_actors=self.num_actors, actor_concurrency=self.actor_concurrency,
            actor_chunk=self.actor_chunk, max_fights=self.max_fights,
            total_steps=self.total_steps, min_buffer=self.min_buffer,
            weight_sync_steps=self.weight_sync_steps, queue_chunks=self.queue_chunks,
            log_every=self.log_every, checkpoint_every=self.checkpoint_every,
            device=self.device, tensorboard=self.tensorboard,
            tb_logdir=self.tb_logdir or f"{self.checkpoint_dir}/tb")

    def exclude_encounters(self):
        if not self.normal_only:
            return None
        from .scenario import NON_NORMAL_ENCOUNTERS
        return NON_NORMAL_ENCOUNTERS

    def build_trainer(self, resume_from: Optional[str] = None):
        from .parallel import ParallelTrainer
        return ParallelTrainer(
            data_path=self.data_path, train_config=self.train_config(),
            mcts_config=self.mcts_config(), parallel_config=self.parallel_config(),
            net_kwargs=self.net_kwargs(), exclude_encounters=self.exclude_encounters(),
            max_turns=self.max_turns, seed=self.seed, resume_from=resume_from)

    # --------------------------------------------------------------------------

    def describe(self):
        """Print every hyperparameter, grouped, with its current value."""
        groups = {
            "Network architecture": ["d_model", "d_card", "d_kind", "d_potion", "d_move",
                                     "logit_scale"],
            "MCTS search": ["num_simulations", "c_puct", "pw_k", "pw_alpha", "discount"],
            "Exploration (self-play)": ["temperature", "dirichlet_alpha",
                                        "dirichlet_epsilon", "add_root_noise"],
            "Value / reward shaping": ["win_value_floor", "loss_value_easy",
                                       "loss_value_hard", "hp_loss_coef", "reward_w",
                                       "reward_k", "max_turns"],
            "Optimization": ["lr", "weight_decay", "batch_size", "grad_clip",
                             "value_loss_coef", "policy_target_smoothing",
                             "buffer_capacity", "min_buffer"],
            "Parallel / run control": ["num_actors", "actor_concurrency", "actor_chunk",
                                       "total_steps", "weight_sync_steps", "queue_chunks",
                                       "log_every", "checkpoint_every", "device",
                                       "tensorboard"],
            "Data / scenario": ["data_path", "normal_only", "max_fights",
                                "checkpoint_dir", "tb_logdir", "seed"],
        }
        d = asdict(self)
        print("=" * 56)
        print("HYPERPARAMETERS")
        print("=" * 56)
        for group, keys in groups.items():
            print(f"\n[{group}]")
            for k in keys:
                print(f"  {k:24s} = {d[k]}")
        print()


if __name__ == "__main__":
    Hyperparams().describe()
