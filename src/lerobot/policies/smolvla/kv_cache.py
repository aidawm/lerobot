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

Saves K/V projections for language tokens across robot control steps.
Language tokens are exactly reusable: the instruction never changes within
an episode, so their K/V projections are identical at every step.

Key idea:
    - Text tokens:  never change within an episode → always reuse
    - Visual tokens: may change → always recompute
    - State token:  changes every step → always recompute

Usage:
    cache = TemporalKVCache()

    # inside your episode loop:
    policy.reset()              # clears cache at episode start
    for step in range(max_steps):
        action = policy.select_action(batch)   # cache used internally
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LayerKVEntry:
    """Cached K/V tensors for one transformer layer."""
    key_states:   torch.Tensor   # [B, N_total, H, D]
    value_states: torch.Tensor   # [B, N_total, H, D]
    static_mask:  torch.Tensor   # [B, N_total] bool  — True = static token
    static_idx:   torch.Tensor   # [N_static]   long  — positions of static tokens


class TemporalKVCache:
    """
    Stores key/value tensors for language tokens across robot control timesteps.

    Token layout assumed (must match the order used in SmolVLMWithExpertModel.forward):
        [visual_tokens (N_vis) | text_tokens (N_text) | state_token (1)]

    Args:
        warmup_steps: Number of steps before the cache starts being used.
            Step 0 always does a full forward pass to populate the cache.
    """

    def __init__(self, warmup_steps: int = 1):
        self.warmup_steps = warmup_steps

        # Per-layer cache: layer_idx -> LayerKVEntry
        self._layer_cache: dict[int, LayerKVEntry] = {}

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
        n_vis: int,
        n_text: int,
        n_state: int = 1,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute a boolean mask [B, N_total] where True = token is static.

        Only language tokens are marked static; visual and state tokens are
        always recomputed.

        Args:
            n_vis: Number of visual tokens.
            n_text: Number of language tokens.
            n_state: Number of proprioceptive state tokens (default 1).
            batch_size: Batch dimension B.
            device: Target device for the mask tensor.

        Returns:
            static_mask: [B, N_total] bool tensor.
        """
        N_total = n_vis + n_text + n_state
        mask = torch.zeros(batch_size, N_total, dtype=torch.bool, device=device)

        # Only text tokens are static — instruction never changes within an episode
        mask[:, n_vis : n_vis + n_text] = True

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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "step": self._step,
            "is_ready": self.is_ready(),
            "static_fraction": self.last_static_fraction,
            "cached_layers": len(self._layer_cache),
        }
