"""Fine-tune the v6 normal model on ELITE combats only, using trial-17's tuned config."""
import torch
torch.manual_seed(0)
torch.set_num_threads(2)
from rl.tune.hyperparams import Hyperparams
from rl.train.parallel import ParallelTrainer
from rl.core.scenario import ELITE_ENCOUNTERS

# Trial 17 (HPO best) hyperparameters.
hp = Hyperparams(
    lr=0.00021917938249826303, c_puct=1.093142882582371, num_simulations=128,
    hp_loss_coef=0.702692130620229, weight_decay=0.00034768883269150074, batch_size=256,
    grad_clip=1.1246283946169007, dirichlet_alpha=0.8178021662724047,
    dirichlet_epsilon=0.2931692362686325, temperature=0.8094581949321787,
    total_steps=100_000, num_actors=30, device="cpu", max_fights=300_000,
    checkpoint_dir="checkpoints_elite", tb_logdir="checkpoints_elite/tb",
    tensorboard=True, normal_only=False, seed=0,
)

if __name__ == "__main__":
    ParallelTrainer(
        data_path="data/ironclad_a0_fights.json.gz",
        train_config=hp.train_config(), mcts_config=hp.mcts_config(),
        parallel_config=hp.parallel_config(), net_kwargs=hp.net_kwargs(),
        include_encounters=ELITE_ENCOUNTERS,          # train on elites only
        max_turns=hp.max_turns, seed=hp.seed,
        resume_from="checkpoints_v6/net_final.pt",     # fine-tune from the full normal model
    ).run()
