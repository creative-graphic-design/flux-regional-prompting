from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0


class RegionalFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    def __init__(self) -> None:
        super().__init__()
        self.regional_mask: Optional[torch.Tensor] = None

    def FluxAttnProcessor2_0_call(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        #
        # The difference from the original __call__ is that
        # a mask is applied when calculating attention, as follows.
        #
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False, attn_mask=attention_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def __call__(  # type: ignore[override]
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        hidden_states_base: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_base: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_rotary_emb_base: Optional[torch.Tensor] = None,
        additional_kwargs: Dict[str, Any] = None,
        base_ratio: Optional[float] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        if base_ratio is not None:
            attn_output_base = self.FluxAttnProcessor2_0_call(
                attn=attn,
                hidden_states=hidden_states_base
                if hidden_states_base is not None
                else hidden_states,
                encoder_hidden_states=encoder_hidden_states_base,
                attention_mask=None,
                image_rotary_emb=image_rotary_emb_base,
            )

            if encoder_hidden_states_base is not None:
                hidden_states_base, encoder_hidden_states_base = attn_output_base
            else:
                hidden_states_base = attn_output_base

        # move regional mask to device
        if base_ratio is not None and "regional_attention_mask" in additional_kwargs:
            if self.regional_mask is not None:
                regional_mask = self.regional_mask.to(hidden_states.device)
            else:
                self.regional_mask = additional_kwargs["regional_attention_mask"]
                regional_mask = self.regional_mask.to(hidden_states.device)
        else:
            regional_mask = None

        attn_output = self.FluxAttnProcessor2_0_call(
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=regional_mask,
            image_rotary_emb=image_rotary_emb,
        )

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = attn_output
        else:
            hidden_states = attn_output

        if encoder_hidden_states is not None:
            if base_ratio is not None:
                # merge hidden_states and hidden_states_base
                hidden_states = (
                    hidden_states * (1 - base_ratio) + hidden_states_base * base_ratio
                )
                return hidden_states, encoder_hidden_states, encoder_hidden_states_base
            else:  # both regional and base input are base prompts, skip the merge
                return hidden_states, encoder_hidden_states, encoder_hidden_states

        else:
            if base_ratio is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : additional_kwargs["encoder_seq_len"]],
                    hidden_states[:, additional_kwargs["encoder_seq_len"] :],
                )

                encoder_hidden_states_base, hidden_states_base = (
                    hidden_states_base[:, : additional_kwargs["encoder_seq_len_base"]],
                    hidden_states_base[:, additional_kwargs["encoder_seq_len_base"] :],
                )

                # merge hidden_states and hidden_states_base
                hidden_states = (
                    hidden_states * (1 - base_ratio) + hidden_states_base * base_ratio
                )

                # concat back
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                hidden_states_base = torch.cat(
                    [encoder_hidden_states_base, hidden_states_base], dim=1
                )

                return hidden_states, hidden_states_base

            else:  # both regional and base input are base prompts, skip the merge
                return hidden_states, hidden_states
