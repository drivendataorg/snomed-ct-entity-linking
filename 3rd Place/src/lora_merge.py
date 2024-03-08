from torch import cuda
import torch
import transformers
from peft import PeftModel


def merge_lora(base_model_path, peft_model_path, unload_model):
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)

    device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device,
        return_dict=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        peft_model_path,
    )

    model = model.merge_and_unload()

    model.save_pretrained(unload_model, safe_serialization=True)
    tokenizer.save_pretrained(unload_model)
    print("End Merge and Unload")
