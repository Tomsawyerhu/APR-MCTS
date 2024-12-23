from vllm import LLM, SamplingParams

STOP_SEQUENCES = ["```"]


def make_model(model_path=""):
    kwargs = {
        "tensor_parallel_size": 1,  # int(os.getenv("VLLM_N_GPUS", "1"))
        "dtype": "float16",
        "trust_remote_code": True,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.98
    }
    model = LLM(model_path, max_model_len=8000, **kwargs)
    return model


def generate_patches(llm, prompt, num_samples=1):
    vllm_outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0.9,
            max_tokens=8000,
            # stop=STOP_SEQUENCES,
            frequency_penalty=0,
            presence_penalty=0,
            n=num_samples,
        ),
        use_tqdm=False,
    )

    output_texts = [x.text for x in vllm_outputs[0].outputs]
    return output_texts


def generate(llm, prompt):
    vllm_outputs = llm.generate(
        prompt,
        SamplingParams(
            temperature=0,
            max_tokens=1024,
            # stop=STOP_SEQUENCES,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
        ),
        use_tqdm=False,
    )

    text = vllm_outputs[0].outputs[0].text
    return text
