"""Potential-based HP shaping reward.

reward(x, y) = Phi(y) - Phi(x), the value of a single health transition x -> y (HP on a
0-100 scale), where

    Phi(h) = (1 - w) * (h/100)  +  w * (1 - e^(-k*h/100)) / (1 - e^(-k))

The linear term cares how much HP changed; the concave term adds a mild "losing at low
health is worse" tilt. `w` controls how much absolute health *location* matters (0 =
amount only); `k` controls how sharply that tilt concentrates near death.

Because it is a difference of a potential, it satisfies (all verified in tests):
  reward(x, x) = 0;  reward(x, y) = -reward(y, x);  bounded to [-1, 1] with
  reward(0, 100) = 1 and reward(100, 0) = -1;  non-decreasing in y / non-increasing in x;
  and telescoping: reward(x, y) = reward(x, z) + reward(z, y).

Notes: HP is clamped to [0, 100]. Any death penalty must be a *separate* terminal reward
-- folding it in here would break telescoping. Scale the whole output by a constant when
combining with larger-magnitude rewards.

Usage downstream (mcts.py): the trainer uses STRICT potential-based shaping in the sense of
Ng, Harada & Russell (1999): the per-transition shaping is `coef * (gamma*Phi(s') - Phi(s))`
with `Phi(terminal) = 0`, built from `potential()` here. With the gamma factor and the zero
terminal potential it telescopes to a constant per episode, so it provably does not change
the optimal policy. `reward()` below (Phi(y) - Phi(x), no gamma) is the undiscounted
single-transition value, kept as a reference / for analysis and tests.
"""

import numpy as np


def potential(h, w=0.25, k=3.0):
    """Phi(h): the shaping potential at health h (0-100). Vectorized."""
    h = np.clip(h, 0.0, 100.0)
    norm = 1.0 - np.exp(-k)
    return (1.0 - w) * (h / 100.0) + w * (1.0 - np.exp(-k * h / 100.0)) / norm


def reward(start_hp, end_hp, w=0.25, k=3.0):
    """Reward for a health transition. HP on 0-100 scale. Vectorized."""
    x = np.clip(start_hp, 0.0, 100.0)
    y = np.clip(end_hp, 0.0, 100.0)
    norm = 1.0 - np.exp(-k)
    linear = (y - x) / 100.0
    concave = (np.exp(-k * x / 100.0) - np.exp(-k * y / 100.0)) / norm
    return (1.0 - w) * linear + w * concave
