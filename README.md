[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://s3.amazonaws.com/drivendata-prod-public/comp_images/snomed-ct-banner.png'>](https://www.drivendata.org/competitions/258/competition-snomed-ct)

# SNOMED CT Entity Linking Challenge

Much of the world's healthcare data is stored in free-text documents, usually clinical notes taken by doctors. One way to analyze clinical notes is to identify and label the portions of each note that correspond to specific medical concepts. This process is called **entity linking** because it involves identifying candidate spans in the unstructured text (the _entities_) and _linking_ them to a particular concept in a knowledge base of medical terminology. Medical notes are often rife with abbreviations (some of them context-dependent) and assumed knowledge. Furthermore, the target knowledge bases can easily include hundreds of thousands of concepts, many of which occur infrequently leading to a “long tail” effect in the distribution of concepts.

The objective of this competition was to **link spans of text in clinical notes with specific topics in the [SNOMED CT](https://www.snomed.org/) clinical terminology**. Participants trained models based on real-world doctor's notes which have been de-identified and annotated with SNOMED CT concepts by medically trained professionals. This is the largest publicly available dataset of labelled clinical notes, and participants in this competition were among the first to demonstrate it's potential!

## What's in this Repository

This repository contains code from winning competitors in the [SNOMED CT Entity Linking Challenge](https://www.drivendata.org/competitions/258/competition-snomed-ct) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | ---            | ---          | ---           | ---
1   | KIRIs          | 0.4452       | 0.4202        | Construct a dictionary that maps (section header, text) to concept IDs by compiling [OMOP](https://www.ohdsi.org/data-standardization/)-based synonyms, openly available medical abbreviations, and simple linguistic rules.
2   | SNOBERT        | 0.4447       | 0.4194        | Fine-tune an ensemble of BERT-based named entity recognition models to extract finding, procedures, body parts, and other entity types. Then classify spans using a pretrained embedding model to find SNOMED CT concepts with the closest embedding distance to the extracted span.
3   | MITEL-UNIUD    | 0.4065       | 0.3777        | First, use a low-rank approximation (LoRA) fine-tuned Large Language Model (LLM) to extract clinical entities from notes. Next classify the extracted spans in two stages: retrieve relevant context using the vector database [Faiss](https://github.com/facebookresearch/faiss), then classify the spans using an LLM.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post: [Meet the winners of the SNOMED CT Entity Linking Challenge](https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners)**

**Benchmark Blog Post: [SNOMED CT Entity Linking Challenge - Benchmark](https://drivendata.co/blog/snomed-ct-entity-linking-benchmark)**
