"""Combat RL for Slay the Spire (built on sts_lightspeed).

Layered packages:
- core   : sim interface, observation encoding, scenario/dataset (CombatSession, ...)
- algos  : the learning algorithm (MCTS, CombatNet, self-play, reward)
- train  : training orchestration + viewers (Trainer, ParallelTrainer, eval, ui)
- tune   : hyperparameters + HPO (Hyperparams, Optuna study)
- test   : small scenario/demo scripts

Subpackages are imported lazily (import what you need, e.g. `from rl.core import
CombatSession`) to keep `import rl` light and free of heavy deps like torch.
"""

__all__ = ["core", "algos", "train", "tune"]
