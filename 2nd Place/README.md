# SNOMED CT Entity Linking Challenge 2nd Place Solution: Team SNOBERT

Team SNOBERT: [bproduct](https://www.drivendata.org/users/bproduct/), [MikhailK](https://www.drivendata.org/users/MikhailK/), [dogwork](https://www.drivendata.org/users/dogwork/)

## Approach

The solution consists of two stages:

- First stage: train NER segmentation task with four classes ('find', 'proc', 'body', 'none');
- Second stage: for each span from the first stage, predict its SNOMED ID.

### First stage

Training an ensemble of BERT models, NER task in B-I-O format (7 classes, B-find, I-find, B-proc, I-proc, B-body, I-body, 0)

- 4 folds note-based split, each fold consists of 51 notes
- [microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)
- [OPTIONAL] Masked lanaguge model (MLM) pretraining of BiomedBERT-large, for a small improvement of a 0.005 over BiomedBERT-base model (24hrs 4xA100)


### Second stage

Using a pretrained embedder, predict ID based on cosine similarity.

1. For the whole SNOMED database extract concepts that are in [Body structure, Findings, Procedure] nodes (about 200k of unique IDs).

2. Get a database of embeddings:

    ```
    # pseudocode
    ID2EMB = {}
    for id in all_ids:
      embeds = []
      for synonym in get_all_synonyms(id):
        embed = embedder(synonym)
        embeds.append(embed)
      true_embed = embeds.mean(0)
      ID2EMB[id] = true_embed
    ```

3. Match extracted mentions (from stage 1) with database:

    ```
    # pseudocode
    for mention in predicted_mentions:
       qvec = embed(mention)
       similarities = qvec @ ID2EMB.values()
       top1_idx = argsort(similarities)
       qID = ID2EMB.keys()[top1_idx]
    ```

    - cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token
    - static dictionary postprocess.

## Data

### Folder structure

```
├── configs
├── data
│   ├── competition_data
│   │   ├── mimic-iv_notes_training_set.csv
│   │   ├── SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition
│   │   └── train_annotations.csv
│   ├── first_stage
│   │   ├── S0_0_score_0.1129
│   │   └── Sall_9_score_0.2807
│   ├── preprocess_data
│   └── second_stage
│       └── sapbert
│           ├── embeds
│           └── model
├── docker
├── output
│   ├── 03-20
│   │   ├── 22_00_40
│   │   │   ├── models
│   │   │   │   └── Sall_56_score_0.9491
│   │   │   ├── src
│   │   │   ├── tb
│   │   │   └── tokenizer
│   │   └── 22_22_46
│   │       ├── models
│   │       │   └── S0_11_score_0.4197
│   │       ├── src
│   │       ├── tb
│   │       └── tokenizer
├── src
└── submission
```


### Raw data

Prior to training, add the following raw data sources do `data/competition_data`:

```
data
└── competition_data
    ├── mimic-iv_notes_training_set.csv
    ├── SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition
    └── train_annotations.csv
```

- MIMIC IV training notes (`data/competition_data/mimic-iv_notes_training_set.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
- MIMIC IV training annotations (`data/competition_data/train_annotations.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
- Challenge SNOMED CT dataset (`data/competition_data/SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition`): Download `SnomedCT_InternationalRF2.zip` from the [challenge data download page](https://www.drivendata.org/competitions/258/competition-snomed-ct/data/). Unzip it to the `data/competition_data` directory.

**Fixing train annotations**: To address certain annotation inaccuracies in `train_annotations.csv`, such as those caused by shifts due to `<br>` tags, we utilized the NER pipeline in [labelstudio](https://labelstud.io/). Approximately 8-10 notes (out of 204) underwent corrections, involving adjustments to around 100-200 annotation IDs (out of over 50,000). These corrections specifically targeted errors resulting from shifted annotations. As a result of these adjustments, the overall score changed by approximately 0.002.

**No-annotation-parts**: Several parts of almost every note were excluded from annotations. We excluded them from the training process. Excluded parts-segments-headers: ['medications on admission:', '___ on admission:', 'discharge medications:']

## Setup

To run a container using the image:

```bash
make docker
```

Or [Anaconda](https://www.anaconda.com/download/):

```bash
conda create -n snomed-snobert python=3.10
conda activate snomed-snobert
pip install .  # installs requirements from pyproject.toml
```

## Technical details and hardware

We used NVIDIA A100-SXM4-40GB (4 cards), but the solution should be replicatable on single A10 24gb

Optional (small score improvement): MLM pretrained model + 24hrs on 4 x A100

Training time:
  - preprocess: 4 minutes
  - train (4GPU): ~30 minutes
  - train (1GPU): ~60 minutes

Inference time: ~1 minute

## Preprocessing

Preprocess the raw data into intermediate features:

```bash
python src/preprocess.py
```

To create data splits for validation run: `python src/preprocess.py --val`

This performs a variety of preprocessing steps including:

- Exclude ['medications on admission:', '___ on admission:', 'discharge medications:'] from train data
- Download pretrained weights from HuggingFace (HF)
- Generate synonyms dictionaries for each concept category from Snomed CT → `data/preprocess_data/{proc, find, body}_sctid_syn.json`
- Calculate embeddings of synonyms dictionary → `data/second_stage/{embedder}/embeds/{name}_{concept_type}.pth`
- Get a static dictionary of span text and concept IDs found in the training data → `data/preprocess_data/most_common_concept.pkl`

## Run training

We use hydra config to configure training of the first stage, an NER task. Set the value of `OUTPUTS` in  `configs/snom.yaml` to the desired output directory (where model weights will be saved).

To run training in [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) mode:

```bash
# DDP mode (multiple GPUs)
torchrun --nproc-per-node=4 src/main.py PARALLEL.DDP=true

# or in single GPU mode
torchrun --nproc-per-node=1 src/main.py PARALLEL.DDP=false
```

As an example:

```bash
torchrun --nproc-per-node=1 src/main.py split=0 class_weights=[0.142,0.142,0.142,0.142,0.142,0.142,0.142] epochs=100 chunked_repeat=2
```

trains in 80 minutes on a single A10 24GB GPU and results in IoU of 0.4277 and first stage score of 0.7410.

The final leaderboard submission was an ensemble of six models:

```bash
torchrun --nproc-per-node=1 src/main.py split=0 class_weights=[0.142,0.142,0.142,0.142,0.142,0.142,0.142] epochs=100
torchrun --nproc-per-node=1 src/main.py split=3 class_weights=[0.142,1,1,1,1,1,1] epochs=100
torchrun --nproc-per-node=1 src/main.py split=all class_weights=[1,1,1,1,1,1,1] epochs=110
torchrun --nproc-per-node=1 src/main.py split=all class_weights=[0.142,0.142,0.142,0.142,0.142,0.142,0.142] epochs=110
torchrun --nproc-per-node=1 src/main.py split=all class_weights=[0.142,0.571,0.571,0.571,0.571,0.571,0.571] epochs=120
torchrun --nproc-per-node=1 src/main.py split=all class_weights=[0.142,0.571,0.571,0.571,0.571,0.571,0.571] epochs=120
```

Training can be started from:
 - the Hugging Face model [`microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract)
 - the Hugging Face model [`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
 - MLM pretrained `microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract`

This should be set in `configs/snom.yaml`, field `model`.

| Model                                       | GPU | First stage score | IoU    | Epoch |
|---------------------------------------------|-----|-------------------|--------|-------|
| MLM pretrained (420000 epochs)              | 1   | 0.7429            | 0.4231 | 75    |
| BiomedNLP-BiomedBERT-large-uncased-abstract | 1   | 0.7487            | 0.4199 | 76    |
| MLM pretrained (420000 epochs)              | 4   | 0.7514            | 0.4302 | 74    |
| BiomedNLP-BiomedBERT-large-uncased-abstract | 4   | 0.7499            | 0.4257 | 72    |

### Output model folder structure

```
output
├── 03-20           # date
│   ├── 22_00_40    # time
│   │   ├── models  # last 3 checkpoints in HF format
│   │   │   ├── Sall_54_score_0.9418  # S - split; all - alldata mode; 54- epoch; 0.9418 - F1 macro score
│   │   │   ├── Sall_55_score_0.9462
│   │   │   └── Sall_56_score_0.9491
│   │   ├── src        # source code for debugging
│   │   ├── tb         # tensorboard logs
│   │   └── tokenizer
```

## Run inference

To run inference:
- Select checkpoints from `output/<date>/<time>/models`, (e.g., `output/03-20/22_00_40/Sall_54_score_0.9418`), place them in `data/first_stage` folder.
- Add test notes to `data/competition_data/test_notes.csv`

Then run the following:

```bash
python submission/main.py
```

To run in validation-scoring mode, run `python submission/main.py --val`.

`main.py` contains variables that point to assets required for inference:

- **model checkpoints**:
    - `FIRST_STAGE_CHECKPOINTS` (default `data/first_stage`): list of paths to model weights for ensembling
    - `SECOND_STAGE_CHECKPOINTS` (default `data/second_stage/sapbert`): path to SapBERT weights
- **static dict path**: `STATIC_DICT_PATH` (default `data/preprocess_data/most_common_concept.pkl`)
- **train data**:
    - `TRAIN_NOTES_PATH` (`data/competition_data/cutmed_fixed_train_annotations.csv`)
    - `TRAIN_ANNOTAIONS_PATH` (`data/competition_data/cutmed_fixed_train_annotations.csv`)
- **inference data**: `TEST_NOTES_PATH` (`data/competition_data/test_notes.csv`)
- **path to save the results of inference**: `SUBMISSION_PATH` (`submission.csv`)
