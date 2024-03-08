import os
from typing import Optional

from loguru import logger
import pandas as pd
from transformers import AutoTokenizer
from typer import Typer

import src.document as Document
import src.vectorDB as LoadVectorize

app = Typer()

PROGRAM_PATH = os.path.abspath(os.getcwd())
ASSETS_PATH = os.path.join(PROGRAM_PATH, "assets")
DATA_ROOT = os.path.join(PROGRAM_PATH, "data")
model_path_faiss_cache = None


def aggregate_subsections(data):
    # Ensure the 'Default' key exists and is initialized as an empty dict
    data["default"] = data.get("default", {})

    # Iterate through all sections except 'Default'
    for section, subsections in data.items():
        if section == "default":
            continue  # Skip the 'Default' section itself

        # Iterate through the subsections and their counts
        for subsection, count in subsections.items():
            # Add to 'Default', aggregating counts
            if subsection in data["default"]:
                data["default"][subsection] += count
            else:
                data["default"][subsection] = count

    return data


def merge_and_extract_annotations(df_annotations, df_notes):
    # Merge the dataframes on 'note_id'
    merged_df = pd.merge(
        df_annotations, df_notes[["note_id", "text", "dict_sections"]], on="note_id"
    )

    # Extract the text segment based on 'start' and 'end' into a new column 'term'
    merged_df["term"] = merged_df.apply(
        lambda row: row["text"][row["start"] : row["end"]].replace("\n", ""), axis=1
    )

    return merged_df


def merge_and_extract(df_concepts, df_concepts_syn):
    # Merge the dataframes on 'note_id'
    concatenated_df = pd.concat([df_concepts_syn, df_concepts], ignore_index=True)
    return concatenated_df.drop_duplicates(subset=["concept_name_clean"])


def set_training_text(term, section):
    if section == "Allergies:":
        return ("Allergy to " + term + " finding").lower().capitalize()
    elif section == "Chief Complaint:":
        return ("Disorder of " + term).lower().capitalize()
    elif section == "Major Surgical or Invasive Procedure:":
        return (term + " procedure").lower().capitalize()
    elif len(term) <= 4 and term.isupper():
        return term
    else:
        return term.lower().capitalize()


@app.command()
def main(
    NOTES_PATH: str = os.path.join(DATA_ROOT, "mimic-iv_notes_training_set.csv"),
    ANNOTATIONS_PATH: str = os.path.join(ASSETS_PATH, "train_annotations.csv"),
    terminologies_path: str = os.path.join(ASSETS_PATH, "dataflattened_terminology.csv"),
    terminologies_path_syn: str = os.path.join(ASSETS_PATH, "newdict_snomed.txt"),
    terminologies_path_syn_extended: str = os.path.join(ASSETS_PATH, "newdict_snomed_extended.txt"),
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    model_path_faiss: str = "sentence-transformers/all-MiniLM-L12-v2",
    faiss_index: str = os.path.join(
        ASSETS_PATH, "faiss_index_constitution_all-MiniLM-L12-v2_finetuned"
    ),
    nr_of_notes: Optional[int] = None,
):
    # PREPARE ANNOTATIONS EXTENDED
    logger.info("Loading documents")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path_faiss_cache)

    if nr_of_notes is not None and nr_of_notes > 0:
        df_notes = Document.load_notes(NOTES_PATH).head(nr_of_notes)
        df_notes = Document.split_documents(df_notes, tokenizer, max_tokens=100)

        print("Preparing data for the first", nr_of_notes, "notes")
    else:
        df_notes = Document.load_notes(NOTES_PATH)
        df_notes = Document.split_documents_2(df_notes, tokenizer, max_tokens=100)

    df_notes = Document.merge_chuncks(df_notes, tokenizer, max_tokens=100)

    df_annotations = Document.load_annotations(ANNOTATIONS_PATH)
    df_annotations["term"] = pd.Series([[]] * len(df_annotations))
    df_annotations = merge_and_extract_annotations(df_annotations, df_notes)
    df_annotations["section"] = pd.Series([""] * len(df_annotations))

    logger.info("Preparing dictionary")
    for row in df_annotations.itertuples():
        dict_sections = {k: v for k, v in row.dict_sections.items() if v is not None}
        for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
            if int(row.start) >= int(value_array[0]) and int(row.end) <= int(value_array[1]):
                df_annotations.loc[row.Index, "section"] = section

    df_annotations["concept_name_clean"] = df_annotations.apply(
        lambda row: set_training_text(row["term"], row["section"]), axis=1
    )
    df_annotations["section"].replace("", pd.NA, inplace=True)
    df_annotations["concept_id"] = df_annotations.concept_id.astype("Int64")

    logger.info("Merging SNOMED-CT terminology, synonyms, and annotations")
    df_concepts = LoadVectorize.load_dataset(terminologies_path)
    df_concepts_syn = LoadVectorize.load_dataset_synonyms(terminologies_path_syn)

    df_grouped = (
        df_annotations.groupby(["concept_name_clean", "section", "concept_id"])
        .size()
        .reset_index(name="count")
    )

    # Transform the grouped DataFrame to have a dictionary of subsections with their counts for each section per title
    df_annotations2 = (
        df_grouped.groupby("concept_name_clean")
        .apply(
            lambda x: x.groupby("section")
            .apply(lambda y: dict(zip(y["concept_id"], y["count"])))
            .to_dict()
        )
        .reset_index(name="dict_sections")
    )

    df_annotations2["dict_sections"] = df_annotations2["dict_sections"].apply(aggregate_subsections)
    df_annotations2["concept_id"] = df_annotations2["dict_sections"].apply(
        lambda d: max(d["default"], key=lambda k: d["default"][k])
    )

    new_df_concepts_syn = merge_and_extract(
        df_concepts[["concept_id", "concept_name_clean"]], df_concepts_syn
    )
    new_df_concepts_syn["dict_sections"] = pd.Series([{}] * len(new_df_concepts_syn))
    new_df_concepts_syn_annotations = merge_and_extract(new_df_concepts_syn, df_annotations2)
    new_df_concepts_syn_annotations.rename(columns={"concept_name_clean": "term"}, inplace=True)
    new_df_concepts_syn_annotations.rename(columns={"concept_id": "code"}, inplace=True)

    new_df_concepts_syn_annotations.to_csv(terminologies_path_syn_extended, sep="\t", index=False)

    logger.info("Generating FAISS index")
    LoadVectorize.generate_db(
        faiss_index, model_path_faiss, model_path_faiss_cache, terminologies_path_syn_extended
    )


if __name__ == "__main__":
    app()
