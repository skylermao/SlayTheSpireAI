"""AlphaZero network for combat: set encoder + per-action policy + value (batched).

The net is the MCTS *evaluator*: given combat states it returns, per state, (priors
over that state's legal actions, scalar value in [-1, 1]). It is fully batched -- a
batch of B states, each with a *variable* number of legal actions and select
candidates, is scored in a single forward (Python only builds index arrays; every NN
op is vectorized). This is what keeps a GPU fed once self-play runs many games at once.

Design (locked earlier):
  * Permutation-invariant Deep-Sets encoder over the observation dict.
  * Per-action scoring head: each LegalAction is embedded from [state, kind, item,
    target] -- where `item` is the card/potion it acts with and `target` its monster --
    and scored; a softmax over each state's legal list gives that state's priors.
  * Value head: tanh -> [-1, 1], matching the MCTS backup scale.

Cards use the dense `card_index` into a learned embedding; piles are count-weighted
sums of it. Potions: held potions are a count vector (state) and a USE_POTION action
carries its `potion_id` (scored via the same potion embedding).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .session import (
    CombatSession, LegalAction,
    END_TURN, PLAY_CARD, USE_POTION, SELECT_CARD, SKIP_SELECT,
)
from . import encoding as enc

_KIND_IDX = {END_TURN: 0, PLAY_CARD: 1, USE_POTION: 2, SELECT_CARD: 3, SKIP_SELECT: 4}
_NUM_KINDS = len(_KIND_IDX)


def _mlp(sizes, act=nn.ReLU, norm=True):
    """MLP with LayerNorm on hidden layers (bounds activation scale -> training
    stability). The final layer is left unnormalized so heads can output raw logits."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if norm:
                layers.append(nn.LayerNorm(sizes[i + 1]))
            layers.append(act())
    return nn.Sequential(*layers)


class CombatNet(nn.Module):
    def __init__(self, d_model: int = 128, d_card: int = 32, d_kind: int = 16,
                 d_potion: int = 16, logit_scale: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.logit_scale = logit_scale   # hard cap on |policy logit| -> CE can't explode

        self.card_embed = nn.Embedding(enc.NUM_CARDS, d_card)
        self.card_proj = _mlp([d_card + (enc.CARD_FEATURES - 1), d_model, d_model])

        self.potion_embed = nn.Embedding(enc.NUM_POTIONS, d_potion)
        self.potion_state_proj = _mlp([d_potion, d_model])   # held-potion count vector
        self.potion_item_proj = _mlp([d_potion, d_model])    # single potion for an action

        self.relic_embed = nn.Embedding(enc.NUM_RELICS, d_potion)  # relic presence (multi-hot)
        self.relic_proj = _mlp([d_potion, d_model])
        self.task_embed = nn.Embedding(enc.NUM_SELECT_TASKS + 1, d_kind)  # +1 = "not selecting"
        self.task_proj = _mlp([d_kind, d_model])

        self.player_proj = _mlp([enc.PLAYER_DIM, d_model, d_model])
        self.monster_proj = _mlp([enc.MONSTER_DIM, d_model, d_model])
        self.scalar_proj = _mlp([enc.SCALAR_DIM, d_model])
        self.pile_proj = _mlp([d_card * 3, d_model])

        # [player, scalars, hand_pool, monster_pool, piles, potions, relics, task] -> h_state
        self.state_mlp = _mlp([d_model * 8, d_model, d_model])

        # [h_state, kind, item, target] -> logit
        self.kind_embed = nn.Embedding(_NUM_KINDS, d_kind)
        self.action_mlp = _mlp([d_model + d_kind + d_model + d_model, d_model, 1])

        self.value_mlp = _mlp([d_model, d_model, 1])

    # ----- encoders (batched: leading dim is the batch) ---------------------

    def encode_cards(self, feats: torch.Tensor) -> torch.Tensor:
        """(..., CARD_FEATURES) -> (..., d_model). Shared by hand and select cands."""
        idx = feats[..., 0].long().clamp(0, enc.NUM_CARDS - 1)
        emb = self.card_embed(idx)
        return self.card_proj(torch.cat([emb, feats[..., 1:]], dim=-1))

    def _encode_states(self, t: dict):
        """t: dict of (B, ...) tensors -> h_state (B,d), hand_enc (B,H,d), mon_enc (B,M,d)."""
        hand = self.encode_cards(t["hand_cards"])                  # (B,H,d)
        hmask = t["hand_mask"].unsqueeze(-1)
        hand_pool = (hand * hmask).sum(1) / hmask.sum(1).clamp(min=1)

        mon = self.monster_proj(t["monsters"])                     # (B,M,d)
        mmask = t["monster_mask"].unsqueeze(-1)
        mon_pool = (mon * mmask).sum(1) / mmask.sum(1).clamp(min=1)

        W = self.card_embed.weight
        piles = torch.cat([t["draw_pile"] @ W, t["discard_pile"] @ W,
                           t["exhaust_pile"] @ W], dim=-1)
        pile_vec = self.pile_proj(piles)
        potion_vec = self.potion_state_proj(t["potions"] @ self.potion_embed.weight)
        relic_vec = self.relic_proj(t["relics"] @ self.relic_embed.weight)
        task_id = t["select_task"].long().squeeze(-1).clamp(0, enc.NUM_SELECT_TASKS)
        task_vec = self.task_proj(self.task_embed(task_id))
        player_vec = self.player_proj(t["player"])
        scalar_vec = self.scalar_proj(t["scalars"])

        h = torch.cat([player_vec, scalar_vec, hand_pool, mon_pool, pile_vec, potion_vec,
                       relic_vec, task_vec], dim=-1)
        return self.state_mlp(h), hand, mon

    # ----- batched forward over ragged actions ------------------------------

    def forward_batch(self, t: dict, actions_batch: "list[list[LegalAction]]",
                      sel_list: "list[torch.Tensor]"):
        """Score every action of every state in one pass.

        Returns (logits_list, values): logits_list[b] is a (A_b,) tensor of action
        logits for state b (softmax -> priors); values is (B,) in [-1, 1].
        """
        B = t["player"].shape[0]
        device = self.card_embed.weight.device
        d = self.d_model
        h_state, hand_enc, mon_enc = self._encode_states(t)

        # Encode all select candidates across the batch at once.
        nonempty = [s for s in sel_list if s.shape[0] > 0]
        sel_enc = self.encode_cards(torch.cat(nonempty, 0)) if nonempty \
            else torch.zeros(0, d, device=device)
        sel_base, running = [], 0
        for s in sel_list:
            sel_base.append(running)
            running += s.shape[0]

        # Build flat index arrays for all actions (pure-Python ints; no NN ops here).
        st_idx, kind_idx, tgt_idx, A_sizes = [], [], [], []
        play_pos, play_st, play_src = [], [], []
        sel_pos, sel_glob = [], []
        pot_pos, pot_ids = [], []
        pos = 0
        for b, acts in enumerate(actions_batch):
            A_sizes.append(len(acts))
            sel_k = 0
            for a in acts:
                st_idx.append(b); kind_idx.append(_KIND_IDX[a.kind]); tgt_idx.append(a.target_idx)
                if a.kind == PLAY_CARD:
                    play_pos.append(pos); play_st.append(b); play_src.append(a.source_idx)
                elif a.kind == SELECT_CARD:
                    sel_pos.append(pos); sel_glob.append(sel_base[b] + sel_k); sel_k += 1
                elif a.kind == USE_POTION:
                    pot_pos.append(pos)
                    pot_ids.append(min(max(a.potion_id, 0), enc.NUM_POTIONS - 1))
                pos += 1
        total_A = pos

        values = torch.tanh(self.value_mlp(h_state)).squeeze(-1)
        if total_A == 0:
            return [h_state.new_zeros(0) for _ in range(B)], values

        def lt(x):
            return torch.tensor(x, dtype=torch.long, device=device)

        st_t = lt(st_idx)
        item = torch.zeros(total_A, d, device=device)
        if play_pos:
            item[lt(play_pos)] = hand_enc[lt(play_st), lt(play_src)]
        if sel_pos:
            item[lt(sel_pos)] = sel_enc[lt(sel_glob)]
        if pot_pos:
            item[lt(pot_pos)] = self.potion_item_proj(self.potion_embed(lt(pot_ids)))

        target = torch.zeros(total_A, d, device=device)
        tgt_t = lt(tgt_idx)
        has_t = (tgt_t >= 0).nonzero(as_tuple=True)[0]
        if has_t.numel() > 0:
            target[has_t] = mon_enc[st_t[has_t], tgt_t[has_t]]

        kind_emb = self.kind_embed(lt(kind_idx))
        feats = torch.cat([h_state[st_t], kind_emb, item, target], dim=-1)
        raw = self.action_mlp(feats).squeeze(-1)
        c = self.logit_scale
        logits = c * torch.tanh(raw / c)        # bound |logit| <= c so softmax/CE stay finite
        return list(torch.split(logits, A_sizes)), values


# =============================================================================
# Evaluator adapter for MCTS (single + batched, device-aware)
# =============================================================================

def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NeuralEvaluator:
    """Wraps a CombatNet as an MCTS evaluator.

    `__call__(session)` evaluates one state (what the current MCTS uses); `evaluate_many`
    scores a list of states in a single batched forward (for concurrent self-play).
    """

    def __init__(self, net: CombatNet, device: Optional[str] = None):
        self.device = device or _auto_device()
        self.net = net.to(self.device).eval()

    def _stack(self, obs_list: "list[dict]") -> dict:
        keys = obs_list[0].keys()
        return {k: torch.as_tensor(np.stack([o[k] for o in obs_list]),
                                   dtype=torch.float32, device=self.device) for k in keys}

    @torch.no_grad()
    def evaluate_many(self, sessions: "list[CombatSession]"):
        t = self._stack([s.observe() for s in sessions])
        actions_batch = [s.legal_actions() for s in sessions]
        sel_list = [torch.as_tensor(s.select_candidate_features(),
                                    dtype=torch.float32, device=self.device) for s in sessions]
        logits_list, values = self.net.forward_batch(t, actions_batch, sel_list)
        out = []
        for logits, v in zip(logits_list, values):
            priors = (F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
                      if logits.numel() else np.zeros(0, np.float32))
            out.append((priors, float(v)))
        return out

    @torch.no_grad()
    def __call__(self, session: CombatSession):
        return self.evaluate_many([session])[0]
