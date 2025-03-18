"""
Wrapper around vLLM generation.

User -> Preset Prompt, Engine Kwargs, File -> Generate Text
"""

from typing import Any, ClassVar, Literal

import easydel as ed
import jax
import transformers
from jax import lax
from jax import numpy as jnp


@ed.traversals.auto_pytree
class vInferenceLoadConfig:
    dtype: jnp.dtype = jnp.bfloat16
    param_dtype: jnp.dtype = jnp.bfloat16
    precision: lax.PrecisionLike = None
    attention_mechanisms: ed.AttentionMechanisms = ed.AttentionMechanisms.AUTO
    sharding_dims: tuple[int] = (..., 1, -1, 1)
    sharding_dcn_dims: tuple[int] = (-1, 1, 1, 1)
    kv_cache_quantization_method: ed.EasyDeLQuantizationMethods = ed.EasyDeLQuantizationMethods.NONE
    task: Literal["llm", "vlm"] = "llm"

    def __post_init__(self):
        reordered_sharding_dims = ()
        for shardin_dim in self.sharding_dims:
            if shardin_dim == Ellipsis:
                shardin_dim = jax.process_count()
            reordered_sharding_dims += (shardin_dim,)
        self.sharding_dims = reordered_sharding_dims


@ed.traversals.auto_pytree
class BaseLLMGenerationConfig:
    prefill_length: int = 1024
    max_new_tokens: int = 1024
    min_length: int | None = None
    streaming_chunks: int = 64
    temperature: float = 0.1
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    no_repeat_ngram_size: int | None = None
    num_return_sequences: int | dict[int, int] | None = 1
    suppress_tokens: list | None = None
    forced_bos_token_id: int | None = None
    forced_eos_token_id: int | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


class BaseLLMProvider:
    def __init__(self, model_name: str):
        pass

    def generate(self, prompts: list[str]) -> list[str]:
        """The input is a list of prompts and the output is a list of generated text."""
        raise NotImplementedError


class vInferenceProvider(BaseLLMProvider):
    def __init__(
        self,
        model_name: str,
        generation_config: BaseLLMGenerationConfig,
        load_config: vInferenceLoadConfig,
    ):
        super().__init__(model_name)
        match load_config.task:
            case "llm":
                processor_class = transformers.AutoTokenizer
                model_loader = ed.AutoEasyDeLModelForCausalLM
            case "vlm":
                processor_class = transformers.AutoProcessor
                model_loader = ed.AutoEasyDeLModelForImageTextToText
            case _:
                raise NotImplementedError()
        max_length = generation_config.max_new_tokens + generation_config.prefill_length
        processor = processor_class.from_pretrained(self.model_name)
        model = model_loader.from_pretrained(
            self.model_name,
            dtype=load_config.dtype,
            param_dtype=load_config.param_dtype,
            precision=load_config.precision,
            config_kwargs=ed.EasyDeLBaseConfigDict(
                attn_mechanism=load_config.attention_mechanisms,
                kv_cache_quantization_method=load_config.kv_cache_quantization_method,
                attn_dtype=load_config.param_dtype,
                attn_softmax_dtype=jnp.float32,
                mask_max_position_embeddings=max_length,
                freq_max_position_embeddings=max_length,
            ),
            sharding_axis_dims=load_config.sharding_dims,
            sharding_dcn_axis_dims=load_config.sharding_dcn_dims,
        )
        self.model = model
        self.vinference = ed.vInference(
            model=model,
            processor_class=processor,
            generation_config=ed.vInferenceConfig(
                max_new_token=generation_config.max_new_token,
                min_length=generation_config.min_length,
                streaming_chunks=generation_config.streaming_chunks,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                do_sample=generation_config.do_sample,
                no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                num_return_sequences=generation_config.num_return_sequences,
                suppress_tokens=generation_config.suppress_tokens,
                forced_bos_token_id=generation_config.forced_bos_token_id,
                forced_eos_token_id=generation_config.forced_eos_token_id,
                pad_token_id=generation_config.pad_token_id,
                bos_token_id=generation_config.bos_token_id,
                eos_token_id=generation_config.eos_token_id,
            ),
            seed=42,
        )
        self.vinference.precompile(
            ed.vInferencePreCompileConfig(
                batch_size=generation_config.batch_size,
                prefill_length=generation_config.prefill_length,
            )
        )

    def generate(self, prompts: list[str]) -> list[str]:
        """The input is a list of prompts and the output is a list of generated text."""
        ids = self.vinference.processor_class.batch_encode_plus(
            prompts,
            return_tensors="jax",
            return_attention_mask=True,
        )
        for response in self.vinference.generate(**ids):  # noqa: B007
            ...
        return self.vinference.processor_class.batch_decode(response.sequences[..., response.padded_length :])


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
        from vllm import LLM, SamplingParams  # type:ignore

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
