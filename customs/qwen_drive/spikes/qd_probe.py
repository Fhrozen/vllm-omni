"""Spike helper: Qwen3.5 VLM wrapper that reads the Qwen-Drive checkpoint layout (`vlm.` prefix)."""

from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration


class QwenDriveVLMProbe(Qwen3_5ForConditionalGeneration):
    def load_weights(self, weights):
        # Expert/perception tensors live in other files; anything without the `vlm.` prefix is ignored.
        return super().load_weights((n[4:], w) for n, w in weights if n.startswith("vlm."))
