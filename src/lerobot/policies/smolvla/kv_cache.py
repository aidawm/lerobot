# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cross-timestep temporal KV cache for SmolVLA.

Saves K/V projections for static visual and text tokens across robot control
steps, so that only changed (dynamic) tokens are recomputed each step.

Key idea:
    - Text tokens:   never change within an episode → always reuse
    - Visual tokens: background patches don't move → reuse when cosine
                     similarity to previous step exceeds `sim_threshold`
    - State token:   changes every step → always recompute

Usage:
    cache = TemporalKVCache(sim_threshold=0.98)

    # inside your episode loop:
    policy.reset()              # clears cache at episode start
    for step in range(max_steps):
        action = policy.select_action(batch)   # cache used internally
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LayerKVEntry:
    """Cached K/V tensors for one transformer layer."""
    key_states:   torch.Tensor   # [B, N_total, H, D]
    value_states: torch.Tensor   # [B, N_total, H, D]
    static_mask:  torch.Tensor   # [B, N_total] bool  — True = static token
    static_idx:   torch.Tensor   # [N_static]   long  — positions of static tokens


class TemporalKVCache:
    """
    Stores key/value tensors for static tokens across robot control timesteps.

    Token layout assumed (must match the order used in SmolVLMWithExpertModel.forward):
        [visual_tokens (N_vis) | text_tokens (N_text) | state_token (1)]

    Args:
        sim_threshold: Cosine similarity above which a visual token is considered
            static and its K/V can be reused. Default 0.98 is conservative; tune
            down to ~0.95 for more aggressive caching.
        warmup_steps: Number of steps before the cache starts being used.
            Step 0 always does a full forward pass to populate the cache.
        protect_top_attn_frac: Fraction of visual tokens with highest cross-attention
            scores (from previous step) to force-recompute even if visually static.
            Set to 0.0 to disable. Helps on fine-grained manipulation tasks.
    """

    def __init__(
        self,
        sim_threshold: float = 0.98,
        warmup_steps: int = 1,
        protect_top_attn_frac: float = 0.0,
    ):
        self.sim_threshold = sim_threshold
        self.warmup_steps = warmup_steps
        self.protect_top_attn_frac = protect_top_attn_frac

        # Per-layer cache: layer_idx -> LayerKVEntry
        self._layer_cache: dict[int, LayerKVEntry] = {}

        # Visual embeddings from previous step (post-connector), used for sim check
        # Shape: [B, N_vis, D]
        self._prev_vis_embeds: Optional[torch.Tensor] = None

        # Cross-attention scores from previous step (for task-relevance protection)
        # Shape: [B, N_vis]
        self._prev_attn_scores: Optional[torch.Tensor] = None

        # Internal step counter
        self._step: int = 0

        # Diagnostics (populated each step for logging)
        self.last_static_fraction: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Must be called at every episode reset."""
        self._layer_cache.clear()
        self._prev_vis_embeds = None
        self._prev_attn_scores = None
        self._step = 0
        self.last_static_fraction = 0.0

    def is_ready(self) -> bool:
        """True once the cache has been populated for at least warmup_steps steps."""
        return self._step > self.warmup_steps and len(self._layer_cache) > 0

    def tick(self) -> None:
        """Advance internal step counter. Call once per control step."""
        self._step += 1

    # ------------------------------------------------------------------
    # Static mask computation
    # ------------------------------------------------------------------

    def compute_static_mask(
        self,
        curr_vis_embeds: torch.Tensor,  # [B, N_vis, D]
        n_text: int,
        n_state: int = 1,
    ) -> torch.Tensor:
        """
        Compute a boolean mask [B, N_total] where True = token is static.

        Saves curr_vis_embeds as prev for the next call — so call this exactly
        once per control step, before the forward pass.

        Args:
            curr_vis_embeds: Post-connector visual token embeddings for this step.
            n_text: Number of language tokens (constant within episode).
            n_state: Number of proprioceptive state tokens (default 1).

        Returns:
            static_mask: [B, N_total] bool tensor on the same device as curr_vis_embeds.
        """
        B, N_vis, _ = curr_vis_embeds.shape
        N_total = N_vis + n_text + n_state
        device = curr_vis_embeds.device

        mask = torch.zeros(B, N_total, dtype=torch.bool, device=device)

        # --- Visual tokens: static iff cosine sim >= threshold ---
        if self._prev_vis_embeds is not None:
            prev = self._prev_vis_embeds.to(device=device, dtype=curr_vis_embeds.dtype)
            # [B, N_vis]
            sim = F.cosine_similarity(curr_vis_embeds, prev, dim=-1)
            vis_static = sim >= self.sim_threshold

            # Task-relevance protection: un-static highly attended tokens
            if self.protect_top_attn_frac > 0.0 and self._prev_attn_scores is not None:
                attn = self._prev_attn_scores.to(device=device)
                k = max(1, int(self.protect_top_attn_frac * N_vis))
                # top-k attended token indices per batch element
                topk_idx = attn.topk(k, dim=-1).indices  # [B, k]
                for b in range(B):
                    vis_static[b, topk_idx[b]] = False

            mask[:, :N_vis] = vis_static

        # --- Text tokens: always static (instruction fixed within episode) ---
        mask[:, N_vis : N_vis + n_text] = True

        # --- State token(s): always dynamic (proprioception changes every step) ---
        # mask[:, N_vis + n_text :] stays False

        # Update prev embeddings for next step
        self._prev_vis_embeds = curr_vis_embeds.detach().clone()

        # Log diagnostic
        self.last_static_fraction = mask.float().mean().item()

        return mask  # [B, N_total]

    # ------------------------------------------------------------------
    # Store / retrieve
    # ------------------------------------------------------------------

    def store_layer(
        self,
        layer_idx: int,
        key_states:   torch.Tensor,  # [B, N_total, H, D]
        value_states: torch.Tensor,  # [B, N_total, H, D]
        static_mask:  torch.Tensor,  # [B, N_total] bool
    ) -> None:
        """
        Store full K/V tensors for a layer alongside the static mask.
        Only the static positions will be reused next step.
        """
        # Precompute static indices from the first batch element.
        # Assumes mask is identical across the batch (single-robot inference).
        static_idx = static_mask[0].nonzero(as_tuple=True)[0]  # [N_static]

        self._layer_cache[layer_idx] = LayerKVEntry(
            key_states=key_states.detach(),
            value_states=value_states.detach(),
            static_mask=static_mask,
            static_idx=static_idx,
        )

    def get_cached_kv(
        self,
        layer_idx: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns (cached_k, cached_v, static_mask, static_idx) for a layer,
        or None if this layer has no cache entry yet.

        cached_k / cached_v contain the FULL sequence K/V from the previous step.
        The caller splices in the cached static positions and recomputes dynamic ones.
        """
        if layer_idx not in self._layer_cache:
            return None
        entry = self._layer_cache[layer_idx]
        return (
            entry.key_states,
            entry.value_states,
            entry.static_mask,
            entry.static_idx,
        )

    def update_attn_scores(self, attn_scores: torch.Tensor) -> None:
        """
        Optionally record cross-attention scores [B, N_vis] from the action expert
        for task-relevance protection next step.
        Call this after each forward pass if protect_top_attn_frac > 0.
        """
        self._prev_attn_scores = attn_scores.detach().clone()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "step": self._step,
            "is_ready": self.is_ready(),
            "static_fraction": self.last_static_fraction,
            "cached_layers": len(self._layer_cache),
            "sim_threshold": self.sim_threshold,
        }
