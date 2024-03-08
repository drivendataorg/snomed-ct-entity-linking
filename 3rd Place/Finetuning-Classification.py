"""python 2.LLAmaQuant-Finetuning-classification.py 2>&1 | tee -a finetuning_classification_out.txt"""

import os
import pprint
import sys
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset
from langchain import PromptTemplate
from peft import LoraConfig, get_peft_model
from torch import cuda
from tqdm.auto import tqdm
from transformers import TrainingArguments
from trl import SFTTrainer

import src.document as Document
from src.config import cache_dir

pd.options.display.max_rows = 20
pd.options.display.max_columns = 500
ppsetup = pprint.PrettyPrinter(indent=4)
pp = ppsetup.pprint
tqdm.pandas()
wandb.init(mode="disabled")
warnings.filterwarnings("ignore")

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

PROGRAM_PATH = os.path.abspath(os.getcwd())
SRC_PATH = os.path.join(PROGRAM_PATH, "assets")
cache_folder = os.path.join(PROGRAM_PATH, "backup")

annotations_train_classification_PATH = os.path.join(
    cache_folder, "annotations_extended_for_classification.gzip"
)  # with data [150:]

# base mistral model used for the fine tune
model_id = "models/mistralai_Mistral-7B-Instruct-v0.2"

# output lora model path
peft_model_path = "models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2/"

# The first element of sys.argv is the input parquet file generated from faiss_classification_data_preparation.py
if len(sys.argv) > 1:
    annotations_train_classification_PATH = sys.argv[1]
# The second element of sys.argv is the base model_id path or name
if len(sys.argv) > 2:
    model_id = sys.argv[2]
# The third element of sys.argv is the output lora model path
if len(sys.argv) > 3:
    peft_model_path = sys.argv[3]

# i.e. python Finetuning-Classification.py backup/annotations_extended_for_classification.gzip models/mistralai_Mistral-7B-Instruct-v0.2 models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2/

df_annotations = pd.read_parquet(annotations_train_classification_PATH)

mode = "complex"
Document.mode = mode
Document.is_submission = False


def load_prompt_template(prompt_path):
    # read file
    prompt_file = open(prompt_path, "r")

    sys_msg = prompt_file.read()
    return sys_msg


def gen_faiss_list(faiss: list):
    result = []
    for i, data in enumerate(faiss):
        id, score, docs = data
        result.append(f"{i}: {id} - {docs}")
    return "\n".join(result)


def get_index_concept_in_faiss(concept_id, faiss: list):
    faiss = [str(f) for f in faiss]
    if str(concept_id) in faiss:
        return faiss.index(str(concept_id))
    return -1


prompt_path = os.path.join(SRC_PATH, "prompts", "prompt_classification_v1_train.txt")
template = load_prompt_template(prompt_path)


def instruction_format(term: str, section: str, context: str, faiss_list: list, output: int):
    prompt_template = PromptTemplate(
        input_variables=["term", "section", "context", "faiss"], template=template
    )
    prompt = prompt_template.format(
        term=term,
        section=section,
        context=context.strip(),
        faiss=gen_faiss_list(faiss_list),
        output=output,
    )
    return prompt


df_annotations["prompt"] = [[] for _ in range(len(df_annotations))]

for i, row in df_annotations.iterrows():
    df_annotations.at[i, "prompt"] = instruction_format(
        term=row["concept_name_clean"],
        section=row["section"],
        context=row["term_context"],
        faiss_list=list(
            zip(
                row["inf_concept_id_20"][:10],
                row["inf_concept_id_score_20"][:10],
                row["inf_concept_id_docs_20"][:10],
            )
        ),
        output=get_index_concept_in_faiss(row["concept_id"], row["inf_concept_id_20"][:10]),
    )

df_annotations[
    [
        "note_id",
        "start",
        "end",
        "concept_id",
        "concept_name_clean",
        "prompt",
        "section",
        "term_context",
    ]
].to_csv(os.path.join(SRC_PATH, "train_classification_prompt.csv"), index=False)

torch.manual_seed(48)
device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

print(device)
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normalized float 4
    bnb_4bit_compute_dtype=getattr(torch, "float16"),  # Computation type
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
)

# Llama 2 Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Llama 2 Model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map=device,
    cache_dir=cache_dir / "hf",
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

# KEVIN PARAMS
LORA_R = 64
LORA_ALPHA = 2 * LORA_R
peft_params = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_params)


# KEVIN PARAMS
training_args = TrainingArguments(
    output_dir=cache_dir / "hf-tmp",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    # gradient_checkpointing=True,
    optim="paged_adamw_8bit",  # "paged_adamw_8bit",
    save_strategy="no",
    logging_steps=25,
    learning_rate=1e-4,
    # weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    # group_by_length=True,
    lr_scheduler_type="constant",
    seed=48,
)

dataset = Dataset.from_pandas(df_annotations)
dataset
print(dataset)

# Check max tokens
df_annotations["token_count"] = df_annotations["prompt"].apply(
    lambda x: Document.get_num_tokens(x, tokenizer=tokenizer)
)
# Since the approach remains essentially the same, the maximum token count will be unchanged
max_tokens = df_annotations["token_count"].max()
print(
    "Template prompt tokens: ",
    Document.get_num_tokens(instruction_format("", "", "", [], 0), tokenizer=tokenizer),
)
print("Max tokens: ", max_tokens)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    max_seq_length=max_tokens + 10,
)

trainer.train()
print("End training")

trainer.model.save_pretrained(peft_model_path)
trainer.tokenizer.save_pretrained(peft_model_path)
