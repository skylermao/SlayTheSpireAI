"""Training orchestration + viewers.

Single-process `Trainer` and multiprocessing actor-learner `ParallelTrainer`, greedy
`evaluate`, plus the playthrough transcript and the localhost UI viewer.
"""

from .train import Trainer, TrainConfig
from .parallel import ParallelTrainer, ParallelConfig
from .eval import evaluate

__all__ = ["Trainer", "TrainConfig", "ParallelTrainer", "ParallelConfig", "evaluate"]
