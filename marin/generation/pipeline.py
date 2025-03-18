from typing import Any

from marin.generation.llm_generation import (
    BaseLLMGenerationConfig,
    BaseLLMProvider,
    vInferenceLoadConfig,
    vInferenceProvider,
    vLLMProvider,
)
from marin.generation.templates import STEP_BY_STEP_TEMPLATE


class TextGeneration:
    def __init__(
        self,
        llm: BaseLLMProvider,
        template: str | None = None,
        num_generations: int = 1,
        prompt_column: str = "text",
    ):
        self.llm = llm

        # Template is a string that contains a placeholder for "example"
        # which will be replaced with the actual example
        self.template = template or STEP_BY_STEP_TEMPLATE
        self.num_generations = num_generations
        self.prompt_column = prompt_column

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generate a batch of text using an LLM where the example text is in dolma format in the "text" column."""

        prompts = [self.template.format(example=example) for example in batch[self.prompt_column]]
        generated_text = self.llm.generate(prompts)

        return {
            "prompt": prompts,
            "generated_text": generated_text,
        }


class vInferenceTextGeneration(TextGeneration):
    def __init__(
        self,
        *,
        model_name: str,
        generation_config: BaseLLMGenerationConfig,
        load_config: vInferenceLoadConfig,
        template: str | None = None,
        prompt_column: str = "text",
        **ignore,
    ):
        llm = vInferenceProvider(
            model_name=model_name,
            generation_config=generation_config,
            load_config=load_config,
        )
        super().__init__(llm, template, generation_config.num_return_sequences, prompt_column)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return super().__call__(batch)


class vLLMTextGeneration(TextGeneration):
    def __init__(
        self,
        *,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: str | None = None,
        num_generations: int = 1,
        num_instances: tuple[int, int] = (1, 4),  # not used
        prompt_column: str = "text",
        **ignore,
    ):
        # Initialize the LLM Provider here for the pipeline since we need the model
        # to be placed in the same placement group as the pipeline
        llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)

        super().__init__(llm, template, num_generations, prompt_column)

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        return super().__call__(batch)
