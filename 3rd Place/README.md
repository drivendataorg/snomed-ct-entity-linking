# SNOMED CT Entity Linking Challenge 3rd Place Solution: Team MITEL-UNIUD

Team MITEL-UNIUD: [vdellamea](https://www.drivendata.org/users/vdellamea/), [mihaihoria.popescu](https://www.drivendata.org/users/mihaihoria.popescu/), [kevinr](https://www.drivendata.org/users/kevinr/)

## Summary

Our approach to the challenge involves two primary tasks: first, an entity recognition task aided by a Large Language Model (LLM), which involves annotating terms within the input text by means of a specific prompt tailored on annotating clinical entities. Secondly, we implement classification in two stages: an initial document retrieval task carried out using the vector database [Faiss](https://github.com/facebookresearch/faiss), followed by a classification task performed leveraging an LLM, again with a specific prompt.

### Entity Recognition

Initially, we split the text of each note into chunks to allow our chosen model, [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), to extract terms in a narrow and more context-homogeneous text. The model was fine-tuned on part of the training set. We also experimented with longer texts but we did observe a decline in entity recognition effectiveness. Our optimal strategy involves using two models: the former processing chunks with a length of 100 tokens, and the latter processing chunks of 500 tokens. After splitting the text into chunks, we apply these fine-tuned models on each segment to annotate and identify terms (in terms of span of text). Post-annotation, we combine the results of the two models, resolving potential overlaps by retaining the longer annotation from the two models. This dual-model approach yielded better effectiveness compared to a single-model application.

### SNOMED-CT Coding

In our classification framework, a pivotal role is played by the integration of `Faiss` for document retrieval, which includes the index for all relevant SNOMED-CT terms and their synonyms as pertinent to the challenge. We augment the Faiss index with pairs of annotations, creating a comprehensive dictionary that tracks term usage and their corresponding codes, acknowledging that a single term may have multiple occurrences and be annotated with different codes based on its context. This enriched vector database is used to retrieve the top 10 text chunks, which then inform the subsequent classification phase.

For the fine-tuning process, we employed a `Mistral` model that incorporates various contextual elements: the term to be classified itself, its surrounding text used to provide context, the section title, and the top 10 text chunks as retrieved from the Faiss vector database. We found this multifaceted approach to have higher classification effectiveness for the final code attributed to each term.

Moreover, to further enhance the results, our methodology included a "remove list" and an "add list" set of terms that have been used to refine the entity recognition output. The "remove list" set excludes terms that, while suggested by the model, are either irrelevant or not part of the training dataset, according to a dictionary-based analysis of the training corpora. This exclusion list addresses issues like stop-words and other non-annotatable but frequently appearing sets of terms. Conversely, the "add list" includes terms that, though consistently overlooked by the model, should always be annotated due to their significance.

The embedding model employed to build the Faiss vector database is `sentence-transformers/all-MiniLM-L12-v2`, chosen for its effectiveness in generating meaningful vector representations that facilitate accurate document retrieval and subsequent classification.

### Open source status

In the development of our solution, in addition to the usual software (Python, etc) we exploited the following external components, which are all open source:

- [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2): Apache 2.0 license
- [Faiss](https://github.com/facebookresearch/faiss): MIT license
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2): Apache 2.0 license
- [vLLM](https://github.com/vllm-project/vllm): Apache 2.0 license
- [PEFT](https://github.com/huggingface/peft): Apache 2.0 license

## Hardware

Our solution was implemented on a server running Ubuntu, equipped with the following hardware specifications:

- CPU: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz, featuring 8 cores and 16 threads.
- GPU: NVIDIA RTX A6000.
- Memory: 64GB of RAM.
- Operating System: Ubuntu 22.04.


## Data

The following input data are needed to train the model:

```
data
├── mimic-iv_notes_training_set.csv  # training notes
└── SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition  # Challenge SNOMED CT dataset
    ├── Readme_en_20230531-challenge-edition.txt
    ├── release_package_information.json
    └── Snapshot/...

assets
└── train_annotations.csv  # training annotations
```

- MIMIC IV training notes (`data/mimic-iv_notes_training_set.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
- MIMIC IV training annotations (`assets/train_annotations.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/). We corrected a few obvious errors in the training annotations including: (1) removing annotations that consisted solely of spaces and non-alphanumeric characters, (2) removing trailing and leading spaces from span text, and (3) correcting misaligned spans. This resulted in approximately 150 corrections across the entire training set.
- Challenge SNOMED CT dataset (`data/SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition`): Download `SnomedCT_InternationalRF2.zip` from the [challenge data download page](https://www.drivendata.org/competitions/258/competition-snomed-ct/data/). Unzip it to the `data` directory.

## Create runtime environment

All of the steps needed to set up an environment in which your code will run, starting on a fresh system with no dependencies (e.g. a brand new Linux, Mac OS X, or Windows installation).

**Prerequisites:**

  - Nvidia CUDA drivers
  - Nvidia CUDA toolkit
  - OpenSSL
  - gcc
  - conda (version 23.11.0)
  - Python

**Python environment:**

```bash
conda env create -f requirements_snomed.yml
```

> [!NOTE]
> In our environment we have used `faiss-cpu`, but as it was shown in the submission, it can work also with `faiss-gpu` which can be installed manually maintaining the same version as `faiss-cpu`.

## Prepare data inputs

1. Create flat terminology (script based on [competition documentation](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/823/#option-4-creating-a-flat-terminology-csv)):

    ```bash
    python process_data.py make-flattened-terminology
    ```

1. Create SNOMED-CT terms file (`assets/newdict_snomed.txt`) that includes all the terms related to the concepts present in the flat terminology:

    ```bash
    python process_data.py generate-sct-dictionary --output-path assets/newdict_snomed.txt
    ```




## Run training

The instructions for preparing the data for the training and train the models are listed below.

The shell script `train.sh` contains the full pipeline that does everything: 

```sh
sh train.sh
```

> [!NOTE]
> The scripts will cache results to `appdirs.user_cache_dir / "mitel_uniud"` (based on [appdirs](https://pypi.org/project/appdirs/), usually `~/.cache/mitel_unid` on a Unix machine).

In detail, the shell script does the following steps, which can be also executed individually:

1. Generation of Faiss database and dictionaries with the `faiss_db_preparation.py` script:

    ```bash
    python faiss_db_preparation.py \
           --notes-path data/mimic-iv_notes_training_set.csv \
           --annotations-path assets/train_annotations.csv \
           --terminologies-path assets/dataflattened_terminology.csv \
           --terminologies-path-syn assets/newdict_snomed.txt \
           --terminologies-path-syn-extended assets/newdict_snomed_extended-150.txt \
           --model-id mistralai/Mistral-7B-Instruct-v0.2 \
           --model-path-faiss sentence-transformers/all-MiniLM-L12-v2 \
           --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150
    ```

    Parameters have the following meaning:
      - `--notes-path data/mimic-iv_notes_training_set.csv`: training notes
      - `--annotations-path assets/train_annotations.csv`: training annotations
      - `--terminologies-path assets/dataflattened_terminology.csv`: SNOMED-CT allowed entities
      - `--terminologies-path-syn assets/newdict_snomed.txt`: SNOMED-CT terms
      - `--terminologies-path-syn-extended assets/newdict_snomed_extended.txt`: OUTPUT file, always paired and used together with faiss index 
      - `--model-id models/mistralai_Mistral-7B-Instruct-v0.2`: base Mistral model name or path, of which we are using the tokenizer
      - `--model-path-faiss sentence-transformers/all-MiniLM-L12-v2`: model used to generate faiss db embeddings
      - `--faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned`: OUTPUT faiss index, always paired and used together with the dictionary set before.

2. Generation of a second Faiss database on part of the notes and dictionaries using the same script `faiss_db_preparation.py` but adding a new argument which consists in the number of notes to use. In our template we have used the first 150 notes. Then this Faiss index and dictionary is used to prepare the data for classification finetuning.

    ```bash
    python faiss_db_preparation.py \
           --notes-path data/mimic-iv_notes_training_set.csv \
           --annotations-path assets/train_annotations.csv \
           --terminologies-path assets/dataflattened_terminology.csv \
           --terminologies-path-syn assets/newdict_snomed.txt \
           --terminologies-path-syn-extended assets/newdict_snomed_extended-150.txt \
           --model-id mistralai/Mistral-7B-Instruct-v0.2 \
           --model-path-faiss sentence-transformers/all-MiniLM-L12-v2 \
           --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150 \
           --nr-of-notes 150
    ```

3. Prepare data to fine tune the classification.

    ```bash
    python faiss_classification_data_preparation.py \
           data/mimic-iv_notes_training_set.csv \
           assets/train_annotations.csv \
           assets/newdict_snomed_extended-150.txt \
           assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150 \
           backup/annotations_extended_for_classification.gzip \
           mistralai/Mistral-7B-Instruct-v0.2 \
           sentence-transformers/all-MiniLM-L12-v2
    ```

    Parameters have the following meaning:
      - `data/mimic-iv_notes_training_set.csv`: training notes
      - `assets/train_annotations.csv`: training annotations
      - `assets/newdict_snomed_extended-150.txt`: dictionary file generated at step 2
      - `assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150`: Faiss index generated at step 2
      - `backup/annotations_extended_for_classification.gzip`: OUTPUT file, which will then be used to fine tune
      - `models/mistralai_Mistral-7B-Instruct-v0.2`: base Mistral model name or path, of which we are using the tokenizer
      - `sentence-transformers/all-MiniLM-L12-v2` model used to generate Faiss DB embeddings.

4. Fine-tune an Entity Recognition model with target split 100, merge 100. This fine-tunes the base Mistral-7B-Instruct-v0.2 model using low-rank approximation (LoRA) as implemented in the [PEFT](https://github.com/huggingface/peft) library.

    ```bash
    python Finetuning-Entity-Recognition.py \
           data/mimic-iv_notes_training_set.csv \
           assets/train_annotations.csv \
           mistralai/Mistral-7B-Instruct-v0.2 \
           models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4/ \
           100 \
           100
    ```

    Parameters have the following meaning:
      - `data/mimic-iv_notes_training_set.csv`: training notes
      - `assets/train_annotations.csv`: training annotations
      - `mistralai/Mistral-7B-Instruct-v0.2`: base Mistral model name or path, used for finetuning
      - `models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4/`: OUTPUT trained model peft path trainind with lora
      - `100`: target nr. of tokens to split text in chunks
      - `100`: target nr. of tokens to merge chunks

5. Fine tune another Entity Recognition model with target split 500, merge 400.

    ```bash
    python Finetuning-Entity-Recognition.py \
           data/mimic-iv_notes_training_set.csv \
           assets/train_annotations.csv \
           mistralai/Mistral-7B-Instruct-v0.2 \
           models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6 \
           500 \
           400
    ```

    Parameters have the following meaning:
      - `data/mimic-iv_notes_training_set.csv`: training notes
      - `assets/train_annotations.csv`: training annotations
      - `mistralai/Mistral-7B-Instruct-v0.2`: base Mistral model name or path, used for finetuning
      - `models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6/` OUTPUT trained model peft path trained with lora
      - `500`: target nr. of tokens to split text in chunks
      - `400`: target nr. of tokens to merge chunks

6. Fine tune the SNOMED-CT classifier. This fine-tunes the base Mistral-7B-Instruct-v0.2 model using LoRA.

    ```bash
    python Finetuning-Classification.py \
           backup/annotations_extended_for_classification.gzip \
           mistralai/Mistral-7B-Instruct-v0.2 \
           models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2
    ```

    Parameters have the following meaning:
      - `backup/annotations_extended_for_classification.gzip`: input file generated at step 3
      - `mistralai/Mistral-7B-Instruct-v0.2`: base Mistral model name or path, used for finetuning
      - `models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2/`: OUTPUT trained model peft path trained with lora.

7. Inference on full training notes and using annotations.

    ```bash
    python main.py \
           --notes-path data/mimic-iv_notes_training_set.csv \
           --model-path-peft models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4 \
           --model-path-2-peft models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6 \
           --model-classification-path-peft models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2 \
           --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned \
           --terminologies assets/newdict_snomed_extended.txt \
           --annotations-path assets/train_annotations.csv
    ```

8. Generate remove and add lists that are used during inference. This uses the predictions generated on the train annotations to build lists of terms to remove and add.

    ```bash
    python remove-add-lists.py \
           backup/df_notes_v4_v6.gzip \
           assets/train_annotations.csv \
           submission.csv \
           sentence-transformers/all-MiniLM-L12-v2
    ```

    Parameters have the following meaning:
      - `backup/df_notes_v4_v6.gzip`: training notes parquet generated at step 7
      - `assets/train_annotations.csv`: training annotations
      - `submission.csv`: submission generated at step 7
      - `sentence-transformers/all-MiniLM-L12-v2`: sentence-transformers model name or path, used for faiss db embeddings


## Run inference

To run inference:

```bash
python main.py
```

This will automatically use the `data/test_notes.csv` as input file, and generate `submission.csv` as the output. You can also infer on other notes by passing the path of the file:

```bash
python main.py <notes_path>
```

It will generate a `submission.csv` file.

In rare cases, a known issue when using Faiss in multithreading may lead to a segmentation fault. If this happens, you may need to deactivate multithreading in `src/snomedctentitylinking.py::L128` as follows:

```python
assign_condition(df_notes, faiss_index, terminologies, model_path_faiss, model_path_faiss_cache, multiprocessing=False)
```
