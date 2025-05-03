"""
Wrapper around vLLM generation.

User -> Preset Prompt, Engine Kwargs, File -> Generate Text
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams  # type: ignore[import]
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to generate text.")

try:
    import easydel as ed  # type: ignore[import]
except ImportError:
    logger.warning("EasyDeL is not installed, so we will not be able to generate text.")


class BaseLLMProvider(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        pass

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        """The input is a list of prompts and the output is a list of generated text."""
        raise NotImplementedError


class vLLMProvider(BaseLLMProvider):
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "tensor_parallel_size": 1,
        "enforce_eager": True,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_name)

        self.model_name = model_name
        self.engine_kwargs = {**vLLMProvider.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        self.generation_kwargs = {**vLLMProvider.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        self.llm = LLM(model=self.model_name, **self.engine_kwargs)
        self.sampling_params = SamplingParams(**self.generation_kwargs)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_text: list[str] = []
        for output in outputs:
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return generated_text


class vSurgeProvider(BaseLLMProvider):
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "sharding_axis_dims": (-1, 1, 4, 1),  # v4 static
        "max_concurrent_decodes": 64,
        "page_size": 128,
        "hbm_utilization": 0.7,
        "max_length": 32768,
        "prefill_lengths": [1024, 2048, 4096, 8192, 16384],
        "verbose": False,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "max_tokens": 1024,
        "top_p": 0.95,
    }

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_name)
        from easydel import SamplingParams
        from jax import lax
        from jax import numpy as jnp
        from transformers import AutoTokenizer

        engine_kwargs = engine_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        self.model_name = model_name

        engine_kwargs = {**vSurgeProvider.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        generation_kwargs = {**vSurgeProvider.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        self.engine_kwargs = engine_kwargs
        self.generation_kwargs = generation_kwargs

        max_length = engine_kwargs.get("max_length")
        max_concurrent_decodes = engine_kwargs.get("max_concurrent_decodes")
        page_size = engine_kwargs.get("page_size")
        hbm_utilization = engine_kwargs.get("hbm_utilization")
        prefill_lengths = engine_kwargs.get("prefill_lengths")

        model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
            model_name,
            sharding_axis_dims=engine_kwargs.get("sharding_axis_dims"),
            config_kwargs=ed.EasyDeLBaseConfigDict(
                attn_mechanism=ed.AttentionMechanisms.PAGED_ATTENTION,
                attn_dtype=jnp.bfloat16,
                attn_softmax_dtype=jnp.bfloat16,
                freq_max_position_embeddings=max_length,
                mask_max_position_embeddings=max_length,
                kv_cache_quantization_blocksize=ed.EasyDeLQuantizationMethods.NONE,
            ),
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            auto_shard_model=True,
            quantize_tensors=ed.EasyDeLQuantizationMethods.NONE,
            precision=lax.Precision.HIGHEST,
        )
        processor = AutoTokenizer.from_pretrained(model_name)
        if processor.pad_token_id is None:
            processor.pad_token_id = processor.eos_token_id
        processor.padding_side = "left"
        self.processor = processor
        self.surge = ed.vSurge.create_odriver(
            model=model,
            processor=processor,
            storage=None,
            manager=None,
            page_size=page_size,
            hbm_utilization=hbm_utilization,
            max_length=max_length,
            prefill_lengths=prefill_lengths,
            max_prefill_length=max(prefill_lengths),
            max_concurrent_decodes=max_concurrent_decodes,
            verbose=engine_kwargs.get("verbose", False),
        )
        self.surge.start()
        self.sampling_params = SamplingParams(**self.generation_kwargs)

    def stop(self):
        self.surge.stop()

    def generate(self, prompts: list[str]) -> list[str]:
        import asyncio

        async def _execute():
            final_results = await self.surge.generate(
                prompts=prompts,
                sampling_params=self.sampling_params,
                stream=False,
            )
            generated_data = []
            for result_list in final_results:
                generated_data.append(result_list.text)
            return generated_data

        return asyncio.run(_execute())
