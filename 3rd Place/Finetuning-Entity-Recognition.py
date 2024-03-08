"""python Finetuning-Entity-Recognition.py 2>&1 | tee -a finetuning_out.txt"""

import os
import pprint
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset
from IPython.display import display
from langchain.prompts import PromptTemplate
from peft import LoraConfig, get_peft_model
from torch import cuda
from tqdm.auto import tqdm
from transformers import TrainingArguments
from trl import SFTTrainer

import src.document as Document
from src.config import cache_dir

warnings.filterwarnings("ignore")

ppsetup = pprint.PrettyPrinter(indent=4)
pp = ppsetup.pprint

pd.options.display.max_rows = 20
pd.options.display.max_columns = 500
wandb.init(mode="disabled")
tqdm.pandas()

PROGRAM_PATH = os.path.abspath(os.getcwd())
SRC_PATH = os.path.join(PROGRAM_PATH, "assets")

# training notes path
NOTES_PATH = os.path.join(PROGRAM_PATH, "data", "mimic-iv_notes_training_set.csv")
# train annotations path
ANNOTATIONS_PATH = os.path.join(SRC_PATH, "train_annotations.csv")

# The first element of sys.argv is the notes path
if len(sys.argv) > 1:
    NOTES_PATH = sys.argv[1]
# The second element of sys.argv is the train annotations path
if len(sys.argv) > 2:
    ANNOTATIONS_PATH = sys.argv[2]

model_id = "models/mistralai_Mistral-7B-Instruct-v0.2"
# The third element of sys.argv is the base model_id path or name
if len(sys.argv) > 3:
    model_id = sys.argv[3]

peft_model_path = "models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4/"
# The fourth element of sys.argv is the output lora model path
if len(sys.argv) > 4:
    peft_model_path = sys.argv[4]

max_tokens_split_target = 100
# The fifth element of sys.argv max tokens split target
if len(sys.argv) > 5:
    max_tokens_split_target = int(sys.argv[5])

max_tokens_merge_target = 100
# The sixth element of sys.argv is the output lora model path
if len(sys.argv) > 6:
    max_tokens_merge_target = int(sys.argv[6])

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

df_notes = Document.load_notes(NOTES_PATH)

df_annotations = Document.load_annotations(ANNOTATIONS_PATH)

df_notes = Document.split_documents(
    df_notes, tokenizer, max_tokens=max_tokens_split_target
)  # split in chuncks and split even more if nr tokens too high i.e. 500
df_notes = Document.merge_chuncks(
    df_notes, tokenizer, max_tokens=max_tokens_merge_target
)  # merge chuncks untill you reach max tokens i.e. 400

df_notes = Document.add_annotations_to_notes(df_notes, df_annotations)

torch.manual_seed(48)
np.random.seed(48)

df_notes = df_notes.sample(frac=1).reset_index(drop=True)

Document.minmax_text_tokens(df_notes, tokenizer)
Document.minmax_chunck_tokens(df_notes, tokenizer)


def are_all_elements_present(target_list, lists, i, dict_sections):
    for t in target_list:
        if not any(t in lst for lst in lists):
            print(i, t)
            for section, value_array in sorted(
                list(dict_sections.items()), key=lambda x: int(x[1][0])
            ):
                if t[4] in section:
                    print(t[4] + " is section name")
                if (
                    t[4] in value_array[2]
                    and int(t[1]) > int(value_array[0])
                    and int(t[1]) < int(value_array[1])
                ):
                    print(t[4] + " is missing in section " + section)
            return False
    return True


def split_annotated_words(df_notes):
    df_notes["annotated_words_chuncks"] = [[] for _ in range(len(df_notes))]
    df_notes["chuncks_annotated_train"] = [[] for _ in range(len(df_notes))]
    missing_annotations = 0
    # Add annotations based on sections chunks
    for i, row in df_notes.iterrows():
        for i_chunk, chunk in enumerate(row["text_chunks_mini_sections"]):
            chunck_index = [
                int(row["text_chunks_mini_sections_indices"][i_chunk][0]),
                int(row["text_chunks_mini_sections_indices"][i_chunk][1]),
            ]
            curr_annotated_words_chuncks = list(
                filter(
                    lambda x: chunck_index[0] <= int(x[1]) and int(x[2]) <= chunck_index[1] + 1,
                    row["annotated_words"],
                )
            )
            df_notes.iloc[i]["annotated_words_chuncks"].append(curr_annotated_words_chuncks)
            df_notes.iloc[i]["chuncks_annotated_train"].append(
                Document.annotatate_notes(
                    chunk,
                    row["text_chunks_mini_sections_indices"][i_chunk][0],
                    curr_annotated_words_chuncks,
                )
            )
        missing_annotations += len(row["annotated_words"]) - sum(
            len(x) for x in row["annotated_words_chuncks"]
        )
    print("Nr. of missing annotations in the text: " + str(missing_annotations))


split_annotated_words(df_notes)


def load_prompt_template(prompt_path):
    # read file
    prompt_file = open(prompt_path, "r")

    sys_msg = prompt_file.read()
    return sys_msg


prompt_path = os.path.join(SRC_PATH, "prompts", "prompt.txt")
sys_msg = load_prompt_template(prompt_path)


# print(sys_msg)
def instruction_format(query: str, output: str):
    # note, don't "</s>" to the end
    template = """<s>[INST] {sys_msg}

# Hospital discharge note:
{query} [/INST]
{output} </s>"""

    prompt_template = PromptTemplate(
        input_variables=["sys_msg", "query", "output"], template=template
    )
    prompt = prompt_template.format(sys_msg=sys_msg, query=query, output=output)
    return prompt


df_notes["prompt_chuncks"] = [[] for _ in range(len(df_notes))]

df_train = pd.DataFrame(columns=["note_id", "chunck_nr", "text"])
for i, row in df_notes.iterrows():
    for i_ch, chunk in enumerate(row["text_chunks_mini_sections"]):
        df_train.loc[len(df_train)] = [
            row["note_id"],
            str(i_ch),
            instruction_format(chunk, row["chuncks_annotated_train"][i_ch]),
        ]

display(df_train)
df_train.to_csv(os.path.join(SRC_PATH, "train_prompt.csv"), index=False)

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

print(device)
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normalized float 4
    bnb_4bit_compute_dtype=getattr(torch, "float16"),  # Computation type
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
)

# tokenizer.pad_token = tokenizer.eos_token
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

# LORA PARAMS
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


# Training args
training_args = TrainingArguments(
    output_dir=cache_dir / "tmp",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_8bit",
    save_strategy="no",
    logging_steps=25,
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_steps=-1,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

dataset = Dataset.from_pandas(df_train)
print(dataset)

# Check max tokens
df_train["token_count"] = df_train["text"].apply(
    lambda x: Document.get_num_tokens(x, tokenizer=tokenizer)
)
# Since the approach remains essentially the same, the maximum token count will be unchanged
max_tokens = df_train["token_count"].max()
print(
    "Template prompt tokens: ",
    Document.get_num_tokens(instruction_format("", ""), tokenizer=tokenizer),
)
print("Max tokens: ", max_tokens)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
    max_seq_length=max_tokens + 10,
)

trainer.train()
print("End training")

trainer.model.save_pretrained(peft_model_path)
trainer.tokenizer.save_pretrained(peft_model_path)
