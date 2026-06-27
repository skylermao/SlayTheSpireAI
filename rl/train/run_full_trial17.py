"""Full 200k-step run on normal fights with trial-17 (HPO best) hyperparameters."""
import torch
torch.manual_seed(0)
torch.set_num_threads(2)
from rl.tune.hyperparams import Hyperparams

hp = Hyperparams(
    # trial 17 (HPO best) config
    lr=0.00021917938249826303, c_puct=1.093142882582371, num_simulations=128,
    hp_loss_coef=0.702692130620229, weight_decay=0.00034768883269150074, batch_size=256,
    grad_clip=1.1246283946169007, dirichlet_alpha=0.8178021662724047,
    dirichlet_epsilon=0.2931692362686325, temperature=0.8094581949321787,
    # full run settings
    total_steps=200_000, num_actors=30, device="cpu", max_fights=300_000,
    normal_only=True, checkpoint_dir="checkpoints_v7", tb_logdir="checkpoints_v7/tb",
    tensorboard=True, seed=0,
)

if __name__ == "__main__":
    hp.build_trainer().run()
