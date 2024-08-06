import fsspec
import ray

@ray.remote(memory=512 * 1024 * 1024)  # 512 MB
def evaluate_model(gcs_path, task, output_path=None):
    print(f"Starting Evaluation for the model: {gcs_path}")
    import subprocess
    args = ["lm_eval", "--model", "vllm", "--model_args", f"pretrained={gcs_path},trust_remote_code=True", 
            "--tasks", {task}, "--batch_size", "auto"]
    if output_path:
        args += ["--output_path", output_path]
    subprocess.run(args)
    

if __name__ == '__main__':
    ray.init(runtime_env={
    "py_modules": 
        [
            "vllm-0.5.3.post1+tpu-py3-none-any.whl",
        ],
    "pip": {"packages": [
            "lm_eval[vllm]",
            "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly+20240601-cp310-cp310-linux_x86_64.whl", 
            "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly+20240601-cp310-cp310-linux_x86_64.whl", 
            "https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20240527%2Bdefault-py3-none-any.whl", 
            "https://storage.googleapis.com/jax-releases/nightly/jax/jax-0.4.29.dev20240527-py3-none-any.whl", 
            "https://storage.googleapis.com/jax-releases/nightly/nocuda/jaxlib-0.4.29.dev20240527-cp310-cp310-manylinux2014_x86_64.whl", 
            "https://download.pytorch.org/whl/cpu/torchvision-0.19.0%2Bcpu-cp310-cp310-linux_x86_64.whl"
            ], "pip_check": False, "pip_version": "==23.0.1;python_version=='3.10'"}})
    a = evaluate_model.remote("gs://levanter-checkpoints/marin/olmoish7b_v4_1024_0627/dlwh_7b0627/hf/step-600000", "mmlu")
    ray.get(a)
