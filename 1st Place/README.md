# SNOMED CT Entity Linking Challenge 1st Place Solution: Team KIRIs

Team KIRIs: [YonatanBilu](https://www.drivendata.org/users/YonatanBilu/), [cyanover](https://www.drivendata.org/users/cyanover/), [IrenaG](https://www.drivendata.org/users/IrenaG/), [guyamit](https://www.drivendata.org/users/guyamit/)

## Summary

The solution is mostly based on a dictionary that maps pairs of (section header, mention) to a SNOMED CT concept ID. In the offline training phase the dictionary is constructed from two sources – the training data and the SNOMED CT concept names (augmented with [OMOP](https://www.ohdsi.org/data-standardization/)-based synonyms). The dictionary is then expanded by some simple linguistic rules, and by permuting multiword expression. More precisely, two dictionaries are constructed – one which is not case sensitive, and one which is. In the inference phase each document is processed independently of the others. It is broken into sections, and in each section the relevant mentions are matched with the text. Successful matches are annotated with the corresponding concept ID. This is done with both dictionaries. Overlaps are then removed by preferring longer mentions, and section-specific keys to general ones. Finally, in a post processing phase mentions may be expanded and annotations made more specific based on SNOMED CT relations.

## Hardware

The solution was run on an x64 Windows 11 machine.

- Number of CPUs: 1
- Processor: Intel(R) Core(TM) i7-10700 CPU @ 2.90GHz
- Memory: 128 GB

Both training and inference were run on CPU.

- Training time (on 204 documents): ~6.5 minutes
- Inference time: 5 documents/minute

## Software

The solution uses an [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) Python environment, although other Python builds should work.

```bash
conda create --name snomedct-kiri python=3.11.5
conda activate snomedct-kiri
pip install -r requirements.txt
```

To use the commands in the `Makefile`, you will need [GNU make](https://www.gnu.org/software/make/), although you can run the commands directly without it.


## Set up data

Acquire the various sources of raw data and save them to the `data/raw` directory.

```
data/raw
├── athena  # OHDSI vocabulary
│   ├── CONCEPT.csv
│   └── CONCEPT_RELATIONSHIP.csv
├── discharge.csv.gz
├── medical_abbreviations.csv
├── mimic-iv_notes_training_set.csv  # training notes
├── SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition  # Challenge SNOMED CT dataset
│   ├── Readme_en_20230531-challenge-edition.txt
│   ├── release_package_information.json
│   └── Snapshot
└── train_annotations.csv  # training annotations
```

- OHDSI Vocabulary: Downloaded the latest [OHDSI Vocabulary Release from Athena](https://athena.ohdsi.org/vocabulary/list). Register for a free account, go to the Download tab, click the checkbox at the top to select all vocabularies that do not require a license, then click "Download vocabularies." Enter any name for your bundle, select version 5.x, and click Download. You will receive an email with a link download a zip archive of the files. Extract the archive to the `data/raw/athena` directory. Note that only the `CONCEPT.csv` and `CONCEPT_RELATIONSHIP.csv` files are needed.
- MIMIC IV discharge notes (`discharge.csv.gz`): From the [MIMIC-IV-Note dataset](https://physionet.org/content/mimic-iv-note/2.2/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/) for access.
- MIMIC IV training notes (`mimic-iv_notes_training_set.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
- MIMIC IV training annotations (`train_annotations.csv`): From the [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/). See [Data access instructions](https://www.drivendata.org/competitions/258/competition-snomed-ct/page/821/).
- Challenge SNOMED CT dataset (`SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition`): Download `SnomedCT_InternationalRF2.zip` from the [challenge data download page](https://www.drivendata.org/competitions/258/competition-snomed-ct/data/). Unzip it to the `data/raw` directory.
- Medical abbreviations (`medical_abbreviations.csv`): A list of medical abbreviations from https://github.com/imantsm/medical_abbreviations. Download and concatenate these into a single CSV by running `bash download_medical_abbreviations.sh`.

## Preprocess the data

After adding the raw data, preprocess the data to produce the required interim files in `data/interim`:

```
./data/interim
├── abbr_dict.pkl
├── abbreviations_snomed_v5.csv
├── flattened_terminology.csv
├── flattened_terminology_syn_snomed+omop_v5.csv
├── medical_abbreviations.csv
├── snomed_relations.csv
├── snomed_unigrams_annotation_dict_20k_v4_fsn.pkl
├── snomed_unigrams_annotation_dict_3k_v4_new.pkl
├── term_extension.csv
└── train_annotations_cln.csv
```

To run all of the preprocessing, run:

```bash
make interim
```

which runs the following Python scripts that produce interim data files:

```bash
python src/process_data.py make-flattened-terminology  # → data/interim/flattened_terminology.csv
python src/process_data.py make-synonyms               # → data/interim/flattened_terminology_syn_snomed+omop_v5.csv
python src/process_data.py make-clean-annotations      # → data/interim/train_annotations_cln.csv
python src/process_data.py make-abbreviations          # → data/interim/abbreviations_snomed_v5.csv
python src/process_data.py make-abbr-dict              # → data/interim/abbr_dict.pkl
python src/process_data.py make-term-extension         # → data/interim/term_extension.csv
python src/process_data.py make-unigrams
# ├── data/interim/snomed_unigrams_annotation_dict_20k_v4_fsn.pkl
# └── data/interim/snomed_unigrams_annotation_dict_3k_v4_new.pkl
```

## Run training

To run training:

```bash
make submission/main.py
```

This will create the following files, which will enable you to run inference:

```
submission
├── assets
│   ├── abbr_dict.pkl
│   ├── kiri_dicts.pkl
│   └── term_extension.csv
├── data
├── main.py
├── mimic_common.py
├── mimic_postprocess_attributes.py
└── mimic_predict.py
```

If you only want to create the dictionaries, or if you want to create them from a different set of annotated documents, you can write your own code that calls `mimic_dev_main.make_kiri_dicts()`.

## Run inference

To run inference, first copy the notes to annotate to `submission/data/test_notes.csv`, then run:

```bash
cd submission
python main.py
```

This will result in an annotation file `submission/submission.csv`.
