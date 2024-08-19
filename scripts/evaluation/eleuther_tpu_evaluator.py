from dataclasses import dataclass
from typing import Dict, List
import time
from scripts.evaluation.evaluator import Evaluator, Dependency

import ray

from scripts.evaluation.vllm_tpu_evaluator import VllmTpuEvaluator

class EleutherTpuEvaluator(VllmTpuEvaluator):
    """For `Evaluator`s that runs inference with VLLM on TPUs."""

    # Default pip packages to install for VLLM on TPUs
    # Some versions were fixed in order to resolve dependency conflicts.
    TORCH_DATE = "20240601"
    DEFAULT_PIP_PACKAGES: List[Dependency] = [
        Dependency(
            name=f"https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly+{TORCH_DATE}"
            "-cp310-cp310-linux_x86_64.whl",
        ),
        Dependency(
            name=f"https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+{TORCH_DATE}"
            "-cp310-cp310-linux_x86_64.whl",
        ),
        Dependency(name="aiohttp"),
        Dependency(name="attrs", version="22.2.0"),
        Dependency(name="click", version="8.1.3"),
        Dependency(name="jsonschema", version="4.23.0"),
        Dependency(name="packaging"),
        Dependency(name="starlette", version="0.37.2"),
        Dependency(name="tokenizers", version="0.19.1"),
        Dependency(name="transformers", version="4.43.2")
    ]

    # def install_eleuther_evaluator() -> None:
        # EleutherTpuEvaluator.run_bash_command("git clone https://github.com/TheQuantumFractal/OLMo.git", check=False)
        # EleutherTpuEvaluator.run_bash_command(
        #     "cd OLMo && pip install -e .[all]"
        # )

    _python_version: str = "3.10"
    _pip_packages: List[Dependency] = DEFAULT_PIP_PACKAGES
    _py_modules: List[Dependency] = []
    
    @staticmethod
    def install_vllm_from_source() -> None:
        """
        Runs the necessary commands to install VLLM from source, following the instructions here:
        https://docs.vllm.ai/en/v0.5.0.post1/getting_started/tpu-installation.html
        TPUs require installing VLLM from source.
        """
        # Additional dependencies to install in order for VLLM to work on TPUs
        EleutherTpuEvaluator.run_bash_command("sudo apt-get update && sudo apt-get install libopenblas-dev --yes")
        EleutherTpuEvaluator.run_bash_command(
            "pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html"
        )
        EleutherTpuEvaluator.run_bash_command(
            "pip install torch_xla[pallas] "
            "-f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html "
            "-f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
        )
        # Clone the VLLM repository to install it from source. Can fail if the repository already exists.
        
        EleutherTpuEvaluator.run_bash_command("mkdir vllmsrc && cd vllmsrc && git clone https://github.com/vllm-project/vllm.git", check=False)
        # Runs https://github.com/vllm-project/vllm/blob/main/setup.py with the `tpu` target device
        EleutherTpuEvaluator.run_bash_command(
            f"cd vllmsrc && cd vllm && git checkout tags/{EleutherTpuEvaluator.VLLM_VERSION} "
            '&& VLLM_TARGET_DEVICE="tpu" pip install -e .'
        )

    @staticmethod
    @ray.remote(memory=8 * 1024 * 1024 * 1024)  # 8 GB
    def _evaluate(model_name_or_path: str, evals: List[str], output_path: str) -> Dict[str, float]:
        # Install VLLM from source
        EleutherTpuEvaluator.install_vllm_from_source()

        # Install Eleuther from source
        # EleutherTpuEvaluator.install_eleuther_evaluator()
        eval_str = ",".join(evals)
        
        command = f"lm_eval --model=vllm --model_args pretrained={model_name_or_path},trust_remote_code=True --tasks={eval_str} --batch_size=auto"
        print(command)

        EleutherTpuEvaluator.run_bash_command(command)

    def evaluate(self, model_name_or_path: str, evals: List[str], output_path: str) -> None:
        """
        Run the evaluator.
        """
        print(f"Running {evals} on {model_name_or_path} and saving results to {output_path}...")
        ray.init(runtime_env=self.get_runtime_env())
        result = ray.get(self._evaluate.remote(model_name_or_path, evals, output_path))
        print(f"Inference times (in seconds): {result}")
