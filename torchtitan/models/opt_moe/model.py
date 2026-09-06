# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses as _dc
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.models.common.attention import (
    AttentionMasksType,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.tools.logging import logger
from .gated_norm_swattention import GatedNormSWAttention
from .utils.inits import (
    build_init_fn,
    parse_depth_init,
    setup_depth_init,
    setup_residual_scale,
)
from .utils.norms import build_norm


def _parse_layer_pattern(
    pattern: "str | list[bool] | list[str] | None",
    n_layers: int,
    true_char: str,
    false_char: str,
    default_true: bool = False,
) -> "list[bool]":
    """Parse a per-layer pattern into a flat list of booleans.

    Args:
        pattern: A string (one char per layer, e.g. ``'SSSF'`` or ``'RRRN'``),
                 a ``list[bool]``, a list-wrapped string (e.g. ``['SSSF']``),
                 or ``None``.
        n_layers: Expected number of layers; length is validated.
        true_char: Character that maps to ``True`` (case-insensitive).
        false_char: Character that maps to ``False`` (case-insensitive).
        default_true: Value returned for every layer when ``pattern`` is ``None``.
    """
    if pattern is None:
        return [default_true] * n_layers
    if isinstance(pattern, list):
        # Some config frontends pass single string values as one-item lists.
        # Accept both ['SSSF'] and ['S','S','S','F'] in addition to list[bool].
        if len(pattern) == 1 and isinstance(pattern[0], str):
            pattern = pattern[0]
        elif pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            pattern = "".join(pattern)
        else:
            if len(pattern) != n_layers:
                raise ValueError(
                    f"Pattern list length {len(pattern)} != n_layers {n_layers}"
                )
            if not all(isinstance(x, bool) for x in pattern):
                raise ValueError(
                    "Pattern list must be list[bool], ['PATTERN'], or list of single-character strings."
                )
            return list(pattern)
    # String path
    allowed = {true_char.upper(), false_char.upper()}
    pattern_up = pattern.upper()
    if len(pattern_up) != n_layers:
        raise ValueError(
            f"Pattern string length {len(pattern_up)} != n_layers {n_layers}"
        )
    invalid = set(pattern_up) - allowed
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid!r} in pattern. Expected only {allowed!r}."
        )
    return [c == true_char.upper() for c in pattern_up]


class OPTMoETransformerBlock(TransformerBlock):
    """
    OPT MoE TransformerBlock Module
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        n_dense_layers: int = 0
        init_gate_as_residual: bool = False
        depth_init: bool | str = "total_depth"
        residual_scale: str = "identity"
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"

    def __init__(self, config: Config, *, layer_id: int, dim: int, n_layers: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = config.attention.build(dim=dim)
        self.attention_norm = build_norm(config.norm_type, dim=dim, eps=config.norm_eps)
        self.ffn_norm = build_norm(config.norm_type, dim=dim, eps=config.norm_eps)

        # Per-layer attention-mode flags (derived from the per-layer attention config).
        assert isinstance(config.attention, GatedNormSWAttention.Config)
        self.use_swa: bool = config.attention.sliding_window_size > 0
        self.attn_backend: str = config.attention.attn_backend

        # Pre-compute the mask dict key used in forward() so we avoid string
        # comparisons and conditional logic on every training step.
        #   None  → SDPA (no mask; PyTorch applies causal masking internally)
        #   "swa" → FlexAttention with sliding-window mask
        #   "full"→ FlexAttention with full-causal mask
        if self.attn_backend == "sdpa":
            self._mask_key: str | None = None
        elif self.use_swa:
            self._mask_key = "swa"
        else:
            self._mask_key = "full"

        # Per-layer debug metadata surfaced in __repr__/print(model).
        self._repr_use_rope: bool = config.attention.use_rope
        self._repr_swa_window_size: int = (
            config.attention.sliding_window_size if self.use_swa else -1
        )
        self._repr_rope_theta: float = -1.0

        self.moe_enabled = layer_id >= config.n_dense_layers
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build(dim=dim, layer_id=layer_id)
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build(dim=dim)

        self.init_gate_as_residual = config.init_gate_as_residual

        # x = identity_scale * x + block_scale * block(x)
        self.depth_init = parse_depth_init(config.depth_init)
        self.residual_div_attn, self.residual_div_ffn = setup_depth_init(
            self.depth_init, layer_id, n_layers
        )
        self.block_scale, self.identity_scale = setup_residual_scale(
            config.residual_scale, n_layers
        )

    def extra_repr(self) -> str:
        return (
            f"layer_id={self.layer_id}, "
            f"rope_theta={self._repr_rope_theta}, "
            f"use_rope={self._repr_use_rope}, "
            f"qk_rope_dim={self.attention.qk_rope_dim}, "
            f"swa_window_size={self._repr_swa_window_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            attention_masks: Attention mask(s) for this layer.
            positions: Optional position indices.
            loss_mask: Optional token loss mask for MoE load-balance loss.

        Returns:
            (output, lbl_loss): output tensor; lbl_loss is this layer's
            load-balance loss (MoE layers) or None (dense layers).
        """
        # _mask_key is pre-computed at init; no string comparisons or tensor ops here.
        if self._mask_key is None:
            layer_mask = None  # SDPA: causal masking is handled internally by PyTorch
        elif isinstance(attention_masks, dict):
            layer_mask = attention_masks[self._mask_key]
        else:
            layer_mask = attention_masks  # single BlockMask (backward compat)

        h = self.identity_scale * x + self.block_scale * self.attention(
            self.attention_norm(x), freqs_cis, layer_mask, positions
        )

        if self.moe_enabled:
            mlp_output, lbl_loss = self.moe(self.ffn_norm(h), loss_mask)
        else:
            mlp_output = self.feed_forward(self.ffn_norm(h))
            lbl_loss = None

        return self.identity_scale * h + self.block_scale * mlp_output, lbl_loss

    def init_weights(self, skip_init: bool = False):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(
            residual_div=self.residual_div_attn,
            skip_init=skip_init,
        )
        if self.moe_enabled:
            self.moe.init_weights(
                residual_div=self.residual_div_ffn,
                init_gate_as_residual=self.init_gate_as_residual,
                skip_init=skip_init,
            )
        else:
            self.feed_forward.init_weights(
                residual_div=self.residual_div_ffn,
                init_gate_as_residual=self.init_gate_as_residual,
                skip_init=skip_init,
            )


class OPTMoEModel(Decoder):
    """
    OPT MoE Transformer model with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        n_layers: int = 24
        vocab_size: int = 201088
        layer: TransformerBlock.Config

        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"

        first_in_init_fn_type: str = "scion_normal_input"
        first_in_init_std: float = 1.0

        final_out_init_fn_type: str = "scion_normal_output"
        final_out_init_std: float = 1.0

        use_embeddings_norm: bool = False
        # --- Flexible per-layer attention configuration ---

        rope_of_swa: RoPE.Config | None = None
        # RoPE frequency cache for SWA layers (typically a lower theta for local context).
        # The global ``rope`` config applies to full-attention layers.
        # None → all layers share the global ``rope`` cache.
        # Ignored for NoPE SWA layers.

        rope_pattern: str | list[bool] | None = None
        # Per-layer RoPE vs NoPE selection.
        # String: one char per layer — 'R' = RoPE, 'N' = NoPE.  E.g. ``"RRRN"`` for 4 layers.
        # list[bool]: True = RoPE, False = NoPE.
        # None keeps the legacy uniform-layer behaviour from ``layer.attention.use_rope``.

        swa_pattern: str | list[bool] | None = None
        # Per-layer sliding-window vs full-attention selection.
        # String: one char per layer — 'S' = SWA, 'F' = Full attention.  E.g. ``"SSSF"``.
        # list[bool]: True = SWA, False = Full attention.
        # None keeps the legacy uniform-layer behaviour from ``layer.attention``.
        # With the default ``layer.attention.sliding_window_size=-1``, this means full attention.
        # SWA layers are automatically assigned attn_backend="flex".
        # ``layer.attention.sliding_window_size`` must be > 0 when any layer is 'S'.

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )

            # Sync rope max_seq_len (both global and SWA-local caches)
            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)
            if self.rope_of_swa is not None:
                self.rope_of_swa = _dc.replace(self.rope_of_swa, max_seq_len=seq_len)

            if self.layer.moe is not None and self.layer.n_dense_layers < self.n_layers:
                self.layer.moe._debug_force_load_balance = debug.moe_force_load_balance

            # Validate per-layer patterns: length and character set are checked by
            # _parse_layer_pattern; also verify a SWA window size is set when needed.
            assert isinstance(self.layer.attention, GatedNormSWAttention.Config)
            use_swa = _parse_layer_pattern(self.swa_pattern, self.n_layers, "S", "F")
            has_swa_from_pattern = any(use_swa)
            # Backward compatibility: when swa_pattern is unset, a positive
            # sliding_window_size still means SWA is active for all layers.
            has_swa_from_base = (
                self.swa_pattern is None
                and self.layer.attention.sliding_window_size > 0
            )
            uses_swa = has_swa_from_pattern or has_swa_from_base

            if uses_swa and self.layer.attention.sliding_window_size <= 0:
                raise ValueError(
                    "SWA is enabled but layer.attention.sliding_window_size "
                    "is not set (must be > 0)."
                )
            if uses_swa and self.layer.attention.attn_backend == "varlen":
                raise ValueError("SWA is not supported with varlen attention.")
            # When swa_pattern is None (uniform base SWA via sliding_window_size>0),
            # no per-layer rebuild happens so the base attn_backend must be "flex".
            # When swa_pattern is explicitly set, the rebuild in __init__ auto-promotes
            # SWA layers to "flex" regardless of the base backend — so sdpa base is fine.
            if has_swa_from_base and self.layer.attention.attn_backend != "flex":
                raise ValueError(
                    "SWA requires attn_backend='flex'. "
                    f"Got attn_backend='{self.layer.attention.attn_backend}'."
                )
            if self.n_layers == self.layer.n_dense_layers:
                # Dense model
                assert self.layer.feed_forward is not None
            else:
                # MoE model
                assert self.layer.moe is not None

            if (
                parallelism.context_parallel_degree > 1
                and self.layer.attention.attn_backend == "varlen"
            ):
                raise NotImplementedError(
                    f"Context Parallel only supports SDPA and FlexAttention."
                    f"Got attn_backend='{self.layer.attention.attn_backend}'. "
                    f"Varlen attention is not supported with CP."
                )

            # Configure expert parallel communication backend from config
            if (
                parallelism.expert_parallel_comm_backend == "deepep"
                and parallelism.expert_parallel_degree > 1
            ):
                # we only use deepep for MoE when ep is enabled
                from torchtitan.models.opt_moe.norm_moe_deepep import DeepEPMoE

                self.layer.moe = DeepEPMoE.Config(**_dc.asdict(self.layer.moe))

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            assert isinstance(self.layer.attention, GatedNormSWAttention.Config)

            if self.layer.attention.head_dim is not None:
                head_dim = self.layer.attention.head_dim
            else:
                head_dim = self.dim // self.layer.attention.n_heads

            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)
        self.norm = build_norm(config.norm_type, dim=config.dim, eps=config.norm_eps)
        if config.use_embeddings_norm:
            self.embeddings_norm = build_norm(
                config.norm_type, dim=config.dim, eps=config.norm_eps
            )
        else:
            self.embeddings_norm = nn.Identity()
        n_layers = config.n_layers
        base_attn = config.layer.attention
        assert isinstance(base_attn, GatedNormSWAttention.Config)
        base_use_rope = base_attn.use_rope
        base_use_swa = base_attn.sliding_window_size > 0

        # Normalize patterns to full bool lists immediately — downstream code never
        # sees None, eliminating repeated null-checks.
        #   rope: None → legacy base setting from layer.attention.use_rope
        #   swa:  None → legacy base setting from layer.attention.sliding_window_size
        use_rope: list[bool] = _parse_layer_pattern(
            config.rope_pattern, n_layers, "R", "N", default_true=base_use_rope
        )
        use_swa: list[bool] = _parse_layer_pattern(
            config.swa_pattern, n_layers, "S", "F", default_true=base_use_swa
        )

        # Rebuild layers with per-layer attention configs when there is any variation.
        # Decoder.__init__ already built uniform layers; we replace the ModuleDict only
        # when needed to avoid double-building the common all-default case.
        # Explicit patterns always trigger a rebuild so all-F/all-R overrides are applied.
        has_explicit_patterns = (
            config.rope_pattern is not None or config.swa_pattern is not None
        )
        rebuild_layers = (
            has_explicit_patterns
            or any(v != base_use_rope for v in use_rope)
            or any(v != base_use_swa for v in use_swa)
        )
        if rebuild_layers:
            swa_window = (
                base_attn.sliding_window_size
            )  # validated > 0 in update_from_config

            self.layers = torch.nn.ModuleDict()
            for layer_id in range(n_layers):
                attn_cfg = _dc.replace(
                    base_attn,
                    use_rope=use_rope[layer_id],
                    sliding_window_size=swa_window if use_swa[layer_id] else -1,
                    # SWA requires FlexAttention; non-SWA keeps the configured backend.
                    attn_backend="flex"
                    if use_swa[layer_id]
                    else base_attn.attn_backend,
                )
                layer_cfg = _dc.replace(config.layer, attention=attn_cfg)
                self.layers[str(layer_id)] = layer_cfg.build(
                    layer_id=layer_id, dim=config.dim, n_layers=n_layers
                )

        # Effective per-layer flags used after construction (including the no-rebuild
        # backward-compatible path where uniform layers from Decoder.__init__ are kept).
        if rebuild_layers:
            layer_use_swa = use_swa
            layer_use_rope = use_rope
        else:
            assert isinstance(config.layer.attention, GatedNormSWAttention.Config)
            layer_use_swa = [config.layer.attention.sliding_window_size > 0] * n_layers
            layer_use_rope = [config.layer.attention.use_rope] * n_layers

        # RoPE frequency cache for SWA layers (separate theta from global).
        # Built only when rope_of_swa is configured AND some layers are SWA.
        if config.rope_of_swa is not None and any(layer_use_swa):
            self.rope_of_swa = config.rope_of_swa.build()
            self.register_buffer(
                "freqs_cis_local", self.rope_of_swa.cache, persistent=False
            )
        else:
            self.rope_of_swa = None
            self.freqs_cis_local = None

        # Pre-compute per-layer freqs_cis selection: True → use freqs_cis_local (SWA + RoPE).
        # Keyed by the string layer_id used in self.layers, matching PP-pruned subsets.
        # NoPE SWA layers (use_rope=False) don't consume the cache, so they stay False.
        self._layer_use_local_rope: dict[str, bool] = {
            str(i): (
                layer_use_swa[i]
                and layer_use_rope[i]
                and config.rope_of_swa is not None
            )
            for i in range(n_layers)
        }

        # Populate per-layer debug metadata shown in print(model).
        for layer_id_str, layer in self.layers.items():
            layer_idx = int(layer_id_str)
            assert isinstance(layer, OPTMoETransformerBlock)

            layer._repr_use_rope = layer_use_rope[layer_idx]
            layer._repr_swa_window_size = (
                layer.attention.sliding_window_size if layer_use_swa[layer_idx] else -1
            )
            if not layer_use_rope[layer_idx]:
                layer._repr_rope_theta = -1.0
            elif layer_use_swa[layer_idx] and config.rope_of_swa is not None:
                layer._repr_rope_theta = float(config.rope_of_swa.theta)
            else:
                layer._repr_rope_theta = float(config.rope.theta)

    def init_weights(
        self,
        **kwargs,
    ):
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        buffer_device = buffer_device or self.freqs_cis.device
        if self.rope is not None:
            self.rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = self.config.rope.build()
            rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = rope.cache

        if self.rope_of_swa is not None:
            self.rope_of_swa.init_weights(buffer_device=buffer_device)
            self.freqs_cis_local = self.rope_of_swa.cache
        elif any(self._layer_use_local_rope[k] for k in self.layers):
            # PP case: rope_of_swa was pruned from this stage, but some layers here
            # still need freqs_cis_local.  Rebuild transiently from the config.
            rope_of_swa = self.config.rope_of_swa.build()
            rope_of_swa.init_weights(buffer_device=buffer_device)
            self.freqs_cis_local = rope_of_swa.cache

        """
        We always init/reset the norm parameters, because its cheap.
        Then we pass the skip_init flag to the layer init_weights to skip the weight initialization.
        """
        if self.norm is not None:
            self.norm.reset_parameters()

        if not isinstance(self.embeddings_norm, nn.Identity):
            self.embeddings_norm.reset_parameters()
        skip_init = kwargs.get("skip_init", False)

        first_in_init_fn = build_init_fn(self.config.first_in_init_fn_type)
        if self.tok_embeddings is not None:
            first_in_init_fn(
                self.tok_embeddings.weight,
                mean=0.0,
                std=self.config.first_in_init_std,
            )

        for layer in self.layers.values():
            # pyrefly: ignore [not-callable]
            layer.init_weights(skip_init=skip_init)

        final_out_init_fn = build_init_fn(self.config.final_out_init_fn_type)
        if self.output is not None:
            final_out_init_fn(
                self.output.weight,
                mean=0.0,
                std=self.config.final_out_init_std,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        accumulated_load_balance_loss: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
            accumulated_load_balance_loss (torch.Tensor | None): Accumulated load balance loss.
            attention_masks (AttentionMasksType | None): Attention masks.
            positions (torch.Tensor | None): Positions.
            loss_mask (torch.Tensor | None): Loss mask.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        h = self.embeddings_norm(h)
        # Collect per-layer load-balance losses; accumulation happens after the loop
        # so we never thread a running tensor through every layer's signature.
        local_lbl_loss: torch.Tensor | None = None
        for layer_id_str, layer in self.layers.items():
            # _layer_use_local_rope is pre-computed at init (pure Python bool dict lookup,
            # no GPU interaction).  When rope_of_swa is None every entry is False so
            # freqs_cis_local is never referenced.
            freqs = (
                self.freqs_cis_local
                if self._layer_use_local_rope[layer_id_str]
                else self.freqs_cis
            )
            h, lbl = layer(h, freqs, attention_masks, positions, loss_mask)
            if lbl is not None:
                local_lbl_loss = lbl if local_lbl_loss is None else local_lbl_loss + lbl

        # Merge local losses with any incoming accumulated loss from a prior PP stage.
        if accumulated_load_balance_loss is not None:
            total_lbl_loss = (
                accumulated_load_balance_loss + local_lbl_loss
                if local_lbl_loss is not None
                else accumulated_load_balance_loss
            )
        elif local_lbl_loss is not None:
            total_lbl_loss = local_lbl_loss
        else:
            total_lbl_loss = torch.zeros((), device=h.device, dtype=torch.float32)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output, total_lbl_loss

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer,
        extra_inputs: "dict[str, torch.Tensor] | None" = None,
    ) -> "AttentionMasksType | None":
        """Return attention masks appropriate for the mix of layer backends.

        Returns:
            ``None``                                  — all layers use SDPA.
            ``{"full": BlockMask}``                   — flex layers, no SWA.
            ``{"full": BlockMask, "swa": BlockMask}`` — mixed flex full + SWA layers.
            Delegates to ``super()`` for varlen (existing behaviour).
        """
        has_flex = any(
            getattr(layer, "attn_backend", "sdpa") == "flex"
            for layer in self.layers.values()
        )
        has_swa = any(
            getattr(layer, "use_swa", False) for layer in self.layers.values()
        )
        has_varlen = any(
            getattr(layer, "attn_backend", "sdpa") == "varlen"
            for layer in self.layers.values()
        )

        if has_varlen and has_swa:
            raise ValueError("SWA is not supported with varlen attention.")

        if has_varlen:
            return super().get_attention_masks(input_batch, tokenizer, extra_inputs)

        if not has_flex:
            # All SDPA — PyTorch handles causal masking internally.
            return None

        # Build base mask modifiers.
        mask_mods = [get_causal_mask_mod()]
        attn_mask_type = self.config.layer.attention.attn_mask_type
        if attn_mask_type == "causal":
            B = 1
        elif attn_mask_type == "block_causal":
            B = input_batch.shape[0]
            assert tokenizer.eos_id is not None
            mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
        else:
            raise ValueError(f"Unknown attn_mask_type: {attn_mask_type!r}")

        seqlen = input_batch.shape[1]
        full_mask = create_attention_mask(
            and_masks(*mask_mods), B, None, seqlen, seqlen
        )

        if not has_swa:
            return {"full": full_mask}

        assert isinstance(self.config.layer.attention, GatedNormSWAttention.Config)
        swa_window = self.config.layer.attention.sliding_window_size
        swa_mask = create_attention_mask(
            and_masks(*mask_mods, get_sliding_window_mask_mod(swa_window)),
            B,
            None,
            seqlen,
            seqlen,
        )
        return {"full": full_mask, "swa": swa_mask}
