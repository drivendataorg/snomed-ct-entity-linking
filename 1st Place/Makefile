.PHONY: interim

data/raw/medical_abbreviations.csv:
	bash download_medical_abbreviations.sh

data/interim/flattened_terminology.csv:
	python src/process_data.py make-flattened-terminology

data/interim/flattened_terminology_syn_snomed+omop_v5.csv: data/interim/flattened_terminology.csv
	python src/process_data.py make-synonyms

data/interim/train_annotations_cln.csv:
	python src/process_data.py make-clean-annotations

data/interim/abbreviations_snomed_v5.csv: data/raw/medical_abbreviations.csv data/interim/flattened_terminology_syn_snomed+omop_v5.csv
	python src/process_data.py make-abbreviations

data/interim/abbr_dict.pkl: data/interim/abbreviations_snomed_v5.csv data/interim/train_annotations_cln.csv
	python src/process_data.py make-abbr-dict

data/interim/term_extension.csv: data/interim/flattened_terminology_syn_snomed+omop_v5.csv
	python src/process_data.py make-term-extension

data/interim/snomed_unigrams_annotation_dict_20k_v4_fsn.pkl data/interim/snomed_unigrams_annotation_dict_3k_v4_new.pkl: data/interim/flattened_terminology_syn_snomed+omop_v5.csv
	python src/process_data.py make-unigrams

## Process all of the raw data into interim data
interim: data/interim/snomed_unigrams_annotation_dict_20k_v4_fsn.pkl data/interim/snomed_unigrams_annotation_dict_3k_v4_new.pkl data/interim/abbr_dict.pkl data/interim/term_extension.csv

## Collect all the assets needed for inference
submission/main.py: interim
	python src/make_inference_env.py submission
