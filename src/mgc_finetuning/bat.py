# MIT License
#
# Copyright (c) 2026 Houtan Ghaffari
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Better Audio Transformer (BAT) ViT-B/16 encoder, loaded from the official Hugging Face weights.

Adapted from https://github.com/houtan-ghaffari/BAT_ICML2026 (arXiv:2602.16305). The official
weights are used through this local implementation instead of ``trust_remote_code``. It differs
from the official encoder in three ways, none of which changes the computed function:

- The merged ``qkv`` layer is split into ``q_proj``, ``k_proj`` and ``v_proj``, so LoRA can adapt
  the same attention projections as in AST (peft cannot target part of a single linear layer).
- ``attn.proj`` is named ``o_proj``, like in AST, because peft matches modules by name suffix and
  a ``proj`` target would also catch the ``patch_embed.proj`` convolution.
- Time chunking, drop path and layer scale are omitted: the released configuration does not use
  them for the fixed 1024-frame inputs of this project.
"""

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


class BATConfig(PretrainedConfig):
    model_type = "bat"

    def __init__(
        self,
        input_shape: tuple[int, int] = (1024, 128),
        patch_size: tuple[int, int] = (16, 16),
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_shape = list(input_shape)
        self.patch_size = list(patch_size)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range


class BATPatchEmbed(nn.Module):
    def __init__(self, config: BATConfig):
        super().__init__()
        patch_size = tuple(config.patch_size)
        self.proj = nn.Conv2d(1, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return self.proj(spectrogram).flatten(2).transpose(1, 2)


class BATAttention(nn.Module):
    """Multi-head self-attention whose output is modulated by a learned gate on the input tokens."""

    def __init__(self, config: BATConfig):
        super().__init__()
        dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)

    def _split_heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tokens.shape
        return tokens.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self._split_heads(self.q_proj(tokens))
        key = self._split_heads(self.k_proj(tokens))
        value = self._split_heads(self.v_proj(tokens))
        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attended = (attention @ value).transpose(1, 2).reshape(tokens.shape)
        return self.o_proj(attended * self.gate(tokens).sigmoid())


class BATMlp(nn.Module):
    def __init__(self, config: BATConfig):
        super().__init__()
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, config.hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(tokens)))


class BATBlock(nn.Module):
    """Transformer block with layer normalization applied after each residual connection."""

    def __init__(self, config: BATConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = BATAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BATMlp(config)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm1(tokens + self.attn(tokens))
        return self.norm2(tokens + self.mlp(tokens))


class BATPreTrainedModel(PreTrainedModel):
    config_class = BATConfig
    base_model_prefix = "bat"
    main_input_name = "input_values"
    _no_split_modules = ["BATBlock"]


def _convert_official_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename the official encoder weights and split each ``qkv`` matrix into its three parts."""
    converted = {}
    for key, value in state_dict.items():
        if ".attn.qkv." in key:
            for name, part in zip(("q_proj", "k_proj", "v_proj"), value.chunk(3, dim=0)):
                converted[key.replace("qkv", name)] = part
        else:
            converted[key.replace(".attn.proj.", ".attn.o_proj.")] = value
    return converted


class BATModel(BATPreTrainedModel):
    """BAT encoder mapping [batch, time, mel] spectrograms to CLS and patch tokens."""

    def __init__(self, config: BATConfig):
        super().__init__(config)
        num_patches = (config.input_shape[0] // config.patch_size[0]) * (config.input_shape[1] // config.patch_size[1])
        self.patch_embed = BATPatchEmbed(config)
        # Both tokens are filled in by the pretrained weights; the sin-cos positions stay fixed.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size), requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.blocks = nn.ModuleList(BATBlock(config) for _ in range(config.num_hidden_layers))
        self.post_init()

    @classmethod
    def from_hub(cls, repo_id: str) -> "BATModel":
        """Load the encoder from a Hugging Face repository with the official weight layout."""
        config = BATConfig.from_pretrained(repo_id)
        # Saved checkpoints must not point at the repository's remote code.
        config.__dict__.pop("auto_map", None)
        model = cls(config)
        state_dict = load_file(hf_hub_download(repo_id, "model.safetensors"))
        model.load_state_dict(_convert_official_state_dict(state_dict))
        return model

    def forward(self, input_values: torch.Tensor) -> BaseModelOutputWithPooling:
        tokens = self.patch_embed(input_values.unsqueeze(1)) + self.pos_embed
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = self.pre_norm(torch.cat((cls_token, tokens), dim=1))
        for block in self.blocks:
            tokens = block(tokens)
        return BaseModelOutputWithPooling(last_hidden_state=tokens, pooler_output=tokens[:, 0])
