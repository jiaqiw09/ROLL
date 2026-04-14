try:
    from megatron.core.transformer.moe import moe_utils
    if not hasattr(moe_utils, "te_general_gemm"):
        moe_utils.te_general_gemm = None
except ImportError:
    pass

from .models import McaGPTModel, McaModelConfig
from .trainer import McaTrainer
from .training_args import Seq2SeqTrainingArguments, TrainingArguments


__all__ = ["McaModelConfig", "McaGPTModel", "TrainingArguments", "Seq2SeqTrainingArguments", "McaTrainer"]
