"""Audio transformer classifiers fine-tuned through their softmax head or through LoRA adapters."""

import json
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from transformers import ASTModel, ASTPreTrainedModel

from .bat import BATModel, BATPreTrainedModel

FINETUNING_METHODS = ("head", "lora")
# Attention layers named alike in AST and BAT, so both models adapt the same projections. Other
# targets: k_proj, o_proj, fc1 and fc2 in both models, and gate in BAT only.
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")


class _FineTunerMixin:
    """Softmax classification head and fine-tuning logic shared by every backbone.

    The fine-tuning method is stored in the config so that ``from_pretrained`` rebuilds the same
    module structure that was saved:

    - ``head``: the backbone is frozen and only the classifier is trained.
    - ``lora``: the original backbone weights are frozen, low-rank adapters are injected into the
      attention projections and trained together with the classifier.
    """

    @classmethod
    def _load_pretrained_backbone(cls, model_name: str) -> nn.Module:
        raise NotImplementedError

    def _build_classifier(self, num_labels: int | None, dropout: float) -> None:
        # from_pretrained moves a num_labels kwarg into the config, so fall back to it.
        self.num_labels = num_labels if num_labels is not None else self.config.num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self._classification_loss = nn.CrossEntropyLoss()
        self.post_init()
        self.freeze_backbone()
        if self.finetuning_method == "lora":
            self._inject_lora()

    @property
    def finetuning_method(self) -> str:
        return getattr(self.config, "finetuning_method", "head")

    def freeze_backbone(self) -> None:
        """Freeze backbone weights and keep the backbone deterministic during training."""
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def enable_lora(
        self,
        rank: int = 1,
        alpha: int = 2,
        dropout: float = 0.2,
        target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    ) -> None:
        """Store the LoRA hyperparameters in the config and inject the adapters into the backbone."""
        self.config.finetuning_method = "lora"
        self.config.lora_rank = rank
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout
        self.config.lora_target_modules = list(target_modules)
        self._inject_lora()

    def _inject_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
        )
        # Only the injected lora_A/lora_B matrices are left trainable inside the backbone.
        inject_adapter_in_model(lora_config, self.backbone)

    def train(self, mode: bool = True):
        """Train the head while keeping a fully frozen backbone in evaluation mode."""
        super().train(mode)
        if self.finetuning_method == "head":
            self.backbone.eval()
        return self

    @classmethod
    def from_backbone(
        cls,
        model_name: str,
        num_labels: int,
        dropout: float = 0.2,
        finetuning_method: str = "head",
        lora_rank: int = 1,
        lora_alpha: int = 2,
        lora_dropout: float = 0.2,
        lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    ):
        """Create the classifier while loading weights from a Hugging Face checkpoint."""
        if finetuning_method not in FINETUNING_METHODS:
            raise ValueError(f"finetuning_method must be one of {FINETUNING_METHODS}, got {finetuning_method!r}")
        backbone = cls._load_pretrained_backbone(model_name)
        backbone.config.finetuning_method = "head"
        # Replace the AudioSet label count so the saved config matches the new classifier.
        backbone.config.num_labels = num_labels
        model = cls(backbone.config, num_labels=num_labels, dropout=dropout)
        # Pretrained weights must be loaded before LoRA wraps the target linear layers.
        model.backbone.load_state_dict(backbone.state_dict())
        if finetuning_method == "lora":
            model.enable_lora(lora_rank, lora_alpha, lora_dropout, lora_target_modules)
        return model

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None):
        outputs = self.backbone(input_values=input_values)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(self.head_norm(pooled)))
        loss = self._classification_loss(logits, labels) if labels is not None else None
        probabilities = self.softmax(logits)
        return {"loss": loss, "logits": logits, "probabilities": probabilities}


class ASTFineTuner(_FineTunerMixin, ASTPreTrainedModel):
    """Audio Spectrogram Transformer backbone with a trainable Softmax classification head."""

    def __init__(self, config, num_labels: int | None = None, dropout: float = 0.1):
        super().__init__(config)
        self.ast = ASTModel(config)
        self.head_norm = nn.Identity()
        self._build_classifier(num_labels, dropout)

    @property
    def backbone(self) -> nn.Module:
        return self.ast

    @classmethod
    def _load_pretrained_backbone(cls, model_name: str) -> ASTModel:
        return ASTModel.from_pretrained(model_name)


class BATFineTuner(_FineTunerMixin, BATPreTrainedModel):
    """Better Audio Transformer backbone with a trainable Softmax classification head."""

    def __init__(self, config, num_labels: int | None = None, dropout: float = 0.1):
        super().__init__(config)
        self.bat = BATModel(config)
        # BAT's CLS features are about 200 times smaller than AST's and barely train a linear layer
        # directly, so they are normalized first, as in the official probing and fine-tuning configs.
        self.head_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._build_classifier(num_labels, dropout)

    @property
    def backbone(self) -> nn.Module:
        return self.bat

    @classmethod
    def _load_pretrained_backbone(cls, model_name: str) -> BATModel:
        return BATModel.from_hub(model_name)


FINETUNER_CLASSES = {"ast": ASTFineTuner, "bat": BATFineTuner}
DEFAULT_MODEL_NAMES = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "bat": "lrauch/BAT-vit-b16-pretrainedAS2M",
}
# BAT learns much more slowly than AST at 1e-4; 1e-3 is also its official linear probing rate.
DEFAULT_LEARNING_RATES = {"ast": 1e-4, "bat": 1e-3}
_MODEL_TYPES = {"audio-spectrogram-transformer": "ast", "bat": "bat"}


def checkpoint_model(checkpoint: Path) -> str:
    """Return the key of ``FINETUNER_CLASSES`` for the model saved in a checkpoint folder."""
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    return _MODEL_TYPES[config["model_type"]]
