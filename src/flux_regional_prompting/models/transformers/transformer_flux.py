from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
)

logger = logging.get_logger(__name__)


class RegionalFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
        hidden_states_base: Optional[torch.Tensor] = None,
        base_ratio: Optional[float] = None,
        image_rotary_emb_base: Optional[torch.Tensor] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        # when using ControlNet, only one prompt is provided
        if hidden_states_base is not None:
            residual_base = hidden_states_base
            norm_hidden_states_base, gate_base = self.norm(hidden_states_base, emb=temb)
            mlp_hidden_states_base = self.act_mlp(
                self.proj_mlp(norm_hidden_states_base)
            )
        else:
            norm_hidden_states_base = None

        output = self.attn(
            hidden_states=norm_hidden_states,
            hidden_states_base=norm_hidden_states_base,
            base_ratio=base_ratio,
            image_rotary_emb=image_rotary_emb,
            image_rotary_emb_base=image_rotary_emb_base,
            additional_kwargs=additional_kwargs,
        )

        if hidden_states_base is not None:
            attn_output, attn_output_base = output
        else:
            attn_output = output

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if hidden_states_base is not None:
            hidden_states_base = torch.cat(
                [attn_output_base, mlp_hidden_states_base], dim=2
            )
            gate_base = gate_base.unsqueeze(1)
            hidden_states_base = gate_base * self.proj_out(hidden_states_base)
            hidden_states_base = residual_base + hidden_states_base
            if hidden_states_base.dtype == torch.float16:
                hidden_states_base = hidden_states_base.clip(-65504, 65504)

        if hidden_states_base is not None:
            return hidden_states, hidden_states_base
        else:
            return hidden_states


class RegionalFluxTransformerBlock(FluxTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: str = "rms_norm",
        eps: float = 0.000001,
    ) -> None:
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_ratio: Optional[float] = None,
        encoder_hidden_states_base: Optional[torch.Tensor] = None,
        image_rotary_emb_base: Optional[torch.Tensor] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # when using ControlNet, only one prompt is provided
        if encoder_hidden_states_base is not None:
            (
                norm_encoder_hidden_states_base,
                c_gate_msa_base,
                c_shift_mlp_base,
                c_scale_mlp_base,
                c_gate_mlp_base,
            ) = self.norm1_context(encoder_hidden_states_base, emb=temb)
        else:
            norm_encoder_hidden_states_base = None

        # Attention.
        output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            encoder_hidden_states_base=norm_encoder_hidden_states_base,
            base_ratio=base_ratio,
            image_rotary_emb=image_rotary_emb,
            image_rotary_emb_base=image_rotary_emb_base,
            additional_kwargs=additional_kwargs,
        )

        if encoder_hidden_states_base is not None:
            attn_output, context_attn_output, context_attn_output_base = output
        else:
            attn_output, context_attn_output = output

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        # Process attention outputs for the `encoder_hidden_states_base`.
        if encoder_hidden_states_base is not None:
            context_attn_output_base = (
                c_gate_msa_base.unsqueeze(1) * context_attn_output_base
            )
            encoder_hidden_states_base = (
                encoder_hidden_states_base + context_attn_output_base
            )

            norm_encoder_hidden_states_base = self.norm2_context(
                encoder_hidden_states_base
            )
            norm_encoder_hidden_states_base = (
                norm_encoder_hidden_states_base * (1 + c_scale_mlp_base[:, None])
                + c_shift_mlp_base[:, None]
            )

            context_ff_output_base = self.ff_context(norm_encoder_hidden_states_base)
            encoder_hidden_states_base = (
                encoder_hidden_states_base
                + c_gate_mlp_base.unsqueeze(1) * context_ff_output_base
            )
            if encoder_hidden_states_base.dtype == torch.float16:
                encoder_hidden_states_base = encoder_hidden_states_base.clip(
                    -65504, 65504
                )

        if encoder_hidden_states_base is not None:
            return encoder_hidden_states, hidden_states, encoder_hidden_states_base
        else:
            return encoder_hidden_states, hidden_states


class RegionalFluxTransformer2DModel(FluxTransformer2DModel):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),
    ) -> None:
        super().__init__(
            patch_size,
            in_channels,
            num_layers,
            num_single_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            pooled_projection_dim,
            guidance_embeds,
            axes_dims_rope,
        )

        self.transformer_blocks = nn.ModuleList(
            RegionalFluxTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
            )
            for _ in range(self.config.num_layers)
        )

        self.single_transformer_blocks = nn.ModuleList(
            RegionalFluxSingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
            )
            for _ in range(self.config.num_single_layers)
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_base: Optional[torch.Tensor] = None,
        base_ratio: Optional[float] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )

        # prepare additional kwargs for regional control
        additional_kwargs = {}
        additional_kwargs["regional_attention_mask"] = joint_attention_kwargs[
            "regional_attention_mask"
        ]
        additional_kwargs["hidden_seq_len"] = hidden_states.shape[1]
        # when using controlnet, only one prompt is provided, so we need to consider the case
        if encoder_hidden_states_base is not None:
            txt_ids_base = txt_ids
            encoder_hidden_states_base = self.context_embedder(
                encoder_hidden_states_base
            )
            additional_kwargs["encoder_seq_len_base"] = (
                encoder_hidden_states_base.shape[1]
            )
        else:
            encoder_hidden_states_base = None
        # prepare txt_ids for concatenated regional prompts
        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
            device=txt_ids.device, dtype=txt_ids.dtype
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        additional_kwargs["encoder_seq_len"] = encoder_hidden_states.shape[1]

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # prepare rotary embeddings for base prompt
        if encoder_hidden_states_base is not None:
            ids_base = torch.cat((txt_ids_base, img_ids), dim=0)
            image_rotary_emb_base = self.pos_embed(ids_base)
        else:
            image_rotary_emb_base = None

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            else:
                encoder_hidden_states, hidden_states, encoder_hidden_states_base = (
                    block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_base=encoder_hidden_states_base,
                        base_ratio=base_ratio,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        image_rotary_emb_base=image_rotary_emb_base,
                        additional_kwargs=additional_kwargs
                        if index_block
                        % joint_attention_kwargs["double_inject_blocks_interval"]
                        == 0
                        else {
                            k: v
                            for k, v in additional_kwargs.items()
                            if k != "regional_attention_mask"
                        },  # delete attention mask to avoid region control
                    )
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block // interval_control]
                )

        if encoder_hidden_states_base is not None:
            hidden_states_base = torch.cat(
                [encoder_hidden_states_base, hidden_states], dim=1
            )
        else:
            hidden_states_base = None

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states, hidden_states_base = block(
                    hidden_states=hidden_states,
                    hidden_states_base=hidden_states_base,
                    base_ratio=base_ratio,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    image_rotary_emb_base=image_rotary_emb_base,
                    additional_kwargs=additional_kwargs
                    if index_block
                    % joint_attention_kwargs["single_inject_blocks_interval"]
                    == 0
                    else {
                        k: v
                        for k, v in additional_kwargs.items()
                        if k != "regional_attention_mask"
                    },  # delete attention mask to avoid region control
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
