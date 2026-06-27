"""Optuna hyperparameter optimization over short proxy training runs.

Each trial: sample hyperparameters, train a fresh agent for `proxy_steps` (a cheap proxy
for a full run), then score it by GREEDY eval on held-out normal fights with a *fixed*
eval search budget (so trials are compared at equal eval-time strength, isolating the
learned net). The objective is winrate + hp_weight*(avg end HP / 80) -- winrate primary,
HP retained as a secondary signal (rewards the defensive quality we care about).

    PYTHONPATH=. python -m rl.tune.hpo --trials 8 --proxy-steps 30000 \
        --eval-games 80 --eval-sims 128 --out checkpoints_hpo

Tuned (union of Core RL + Optimization + Exploration):
  lr, c_puct, num_simulations, hp_loss_coef, weight_decay, batch_size, grad_clip,
  dirichlet_alpha, dirichlet_epsilon, temperature.

Note: 10 dims over ~8 trials is underpowered for TPE (it behaves close to random until it
has more history) -- more trials, or fewer params, sharpen the Bayesian benefit.

Trials run sequentially: each uses all actors/cores. An Optuna SQLite study is persisted
so the search can be resumed (`--out` dir).
"""

import argparse
import os

import torch

from .hyperparams import Hyperparams
from ..train.eval import evaluate

HP_NORM = 80.0   # nominal max HP for normalizing the HP-retained term


def make_objective(args):
    def objective(trial):
        # ----- sample the tuned hyperparameters -----
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        c_puct = trial.suggest_float("c_puct", 0.75, 3.0)
        num_simulations = trial.suggest_categorical("num_simulations", [64, 128, 192, 256])
        hp_loss_coef = trial.suggest_float("hp_loss_coef", 0.0, 1.0)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)
        dirichlet_alpha = trial.suggest_float("dirichlet_alpha", 0.1, 1.0)
        dirichlet_epsilon = trial.suggest_float("dirichlet_epsilon", 0.1, 0.4)
        temperature = trial.suggest_float("temperature", 0.5, 1.5)

        trial_dir = os.path.join(args.out, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        hp = Hyperparams(
            lr=lr, c_puct=c_puct, num_simulations=num_simulations,
            hp_loss_coef=hp_loss_coef, weight_decay=weight_decay, batch_size=batch_size,
            grad_clip=grad_clip, dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon, temperature=temperature,
            # proxy run settings
            total_steps=args.proxy_steps, checkpoint_dir=trial_dir,
            checkpoint_every=args.proxy_steps + 1,   # only net_final.pt (save disk)
            tensorboard=False, normal_only=True, num_actors=args.num_actors,
            device=args.device, seed=args.seed,
        )
        print(f"\n=== trial {trial.number}: {trial.params} ===", flush=True)

        torch.manual_seed(args.seed)
        torch.set_num_threads(2)
        hp.build_trainer().run()

        ckpt = os.path.join(trial_dir, "net_final.pt")
        r = evaluate(ckpt, args.data, n_games=args.eval_games, sims=args.eval_sims,
                     concurrency=16, device=args.device, seed=args.eval_seed,
                     max_fights=args.max_fights, net_kwargs=hp.net_kwargs(),
                     exclude_encounters=hp.exclude_encounters())
        score = r["winrate"] + args.hp_weight * (r["avg_end_hp"] / HP_NORM)
        trial.set_user_attr("winrate", r["winrate"])
        trial.set_user_attr("avg_end_hp", r["avg_end_hp"])
        print(f"=== trial {trial.number}: winrate={r['winrate']:.3f} "
              f"avg_end_hp={r['avg_end_hp']:.1f} -> score={score:.4f} ===", flush=True)
        return score

    return objective


def main():
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--proxy-steps", type=int, default=30_000)
    ap.add_argument("--eval-games", type=int, default=80)
    ap.add_argument("--eval-sims", type=int, default=128)
    ap.add_argument("--hp-weight", type=float, default=0.5)
    ap.add_argument("--num-actors", type=int, default=30)
    ap.add_argument("--max-fights", type=int, default=150_000)
    ap.add_argument("--data", default="data/ironclad_a0_fights.json.gz")
    ap.add_argument("--out", default="checkpoints_hpo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=12345)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # TPE sampler; SQLite storage so the study resumes if interrupted. n_startup_trials=8
    # means once the study holds >=8 completed trials, TPE proposes from that history
    # rather than random-sampling -- so a *resumed* study exploits prior trials instead of
    # replaying the (re-seeded) random startup sequence.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=8),
        study_name="combat_hpo",
        storage=f"sqlite:///{os.path.join(args.out, 'study.db')}",
        load_if_exists=True,
    )
    study.optimize(make_objective(args), n_trials=args.trials)

    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print(f"  score   : {study.best_value:.4f}")
    print(f"  winrate : {study.best_trial.user_attrs.get('winrate')}")
    print(f"  end_hp  : {study.best_trial.user_attrs.get('avg_end_hp')}")
    print(f"  params  : {study.best_params}")
    df = study.trials_dataframe()
    csv = os.path.join(args.out, "trials.csv")
    df.to_csv(csv, index=False)
    print(f"  all trials -> {csv}")


if __name__ == "__main__":
    main()
