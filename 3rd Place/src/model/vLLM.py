from vllm import LLM, SamplingParams


def instantiate(MODEL_DIR, CACHE_DIR=None):
    return LLM(model=MODEL_DIR, seed=48, dtype="bfloat16")


def inference(llm, prompts):
    sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.1,
        min_p=0.05,
        top_k=10,
        repetition_penalty=1,
        max_tokens=1000,
        frequency_penalty=0,
    )
    outputs = llm.generate(prompts, sampling_params)

    return [o.outputs[0].text for o in outputs]
