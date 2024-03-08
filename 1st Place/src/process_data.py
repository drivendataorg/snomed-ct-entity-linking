import csv
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from mimic_common import common_headers
from mimic_train import (
    get_allowed_sections,
    get_snomed_synonyms,
    limit_any_to_allowed_sections,
)

app = typer.Typer()
data_directory = Path(__file__).parent.parent / "data"
raw_directory = data_directory / "raw"
(interim_directory := data_directory / "interim").mkdir(exist_ok=True, parents=True)

invalid_vocabs = (
    "SNOMED",
    "CAP",
    "CPT4",
    "MedDRA",
    "Gemscript",
    "ICD10CN",
    "ICD10GM",
    "ICD9ProcCN",
    "OPS",
)


def load_snomed_ct(data_path: Path):
    """
    Create a SNOMED CT concept DataFrame.

    Derived from: https://github.com/CogStack/MedCAT/blob/master/medcat/utils/preprocess_snomed.py

    Returns:
        pandas.DataFrame: SNOMED CT concept DataFrame.
    """

    def _read_file_and_subset_to_active(filename):
        with open(filename, encoding="utf-8") as f:
            entities = [[n.strip() for n in line.split("\t")] for line in f]
            df = pd.DataFrame(entities[1:], columns=entities[0])
        return df[df.active == "1"]

    active_terms = _read_file_and_subset_to_active(
        data_path / "sct2_Concept_Snapshot_INT_20230531.txt"
    )
    active_descs = _read_file_and_subset_to_active(
        data_path / "sct2_Description_Snapshot-en_INT_20230531.txt"
    )

    df = pd.merge(active_terms, active_descs, left_on=["id"], right_on=["conceptId"], how="inner")[
        ["id_x", "term", "typeId"]
    ].rename(columns={"id_x": "concept_id", "term": "concept_name", "typeId": "name_type"})

    # active description or active synonym
    df["name_type"] = df["name_type"].replace(
        ["900000000000003001", "900000000000013009"], ["P", "A"]
    )
    active_snomed_df = df[df.name_type.isin(["P", "A"])]

    active_snomed_df["hierarchy"] = active_snomed_df["concept_name"].str.extract(
        r"\((\w+\s?.?\s?\w+.?\w+.?\w+.?)\)$"
    )
    active_snomed_df = active_snomed_df[active_snomed_df.hierarchy.notnull()].reset_index(drop=True)
    return active_snomed_df


@app.command()
def make_flattened_terminology(
    snomed_ct_directory: Path = data_directory
    / "raw"
    / "SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition",
    output_path: Path = interim_directory / "flattened_terminology.csv",
):
    # unzip the terminology provided on the data download page and specify the path to the folder here
    snomed_rf2_path = Path(snomed_ct_directory)

    # load the SNOMED release
    df = load_snomed_ct(snomed_rf2_path / "Snapshot" / "Terminology")
    logger.debug(f"Loaded SNOMED CT release containing {len(df):,} rows (expected 364,323).")

    concept_type_subset = [
        "procedure",  # top level category
        "body structure",  # top level category
        "finding",  # top level category
        "disorder",  # child of finding
        "morphologic abnormality",  # child of body structure
        "regime/therapy",  # child of procedure
        "cell structure",  # child of body structure
    ]

    filtered_df = df[
        (
            df.hierarchy.isin(concept_type_subset)
        )  # Filter the SNOMED data to the selected Concept Types
        & (df.name_type == "P")  # Preferred Terms only (i.e. one row per concept, drop synonyms)
    ].copy()
    logger.debug(f"Filtered to {len(filtered_df):,} relevant rows (expected 218,467).")

    logger.debug(f"Value counts:\n{filtered_df.hierarchy.value_counts()}")

    logger.debug(f"Saving flattened terminology with {len(filtered_df):,} rows to {output_path}")
    filtered_df.drop("name_type", axis="columns", inplace=True)
    filtered_df.to_csv(output_path)
    return filtered_df


@app.command()
def make_clean_annotations():
    logger.info("Loading SNOMED CT terms, notes, and annotations...")
    snomed = pd.read_csv(
        interim_directory / "flattened_terminology.csv", usecols=["concept_id", "concept_name"]
    )
    snomed = snomed.drop_duplicates("concept_id").set_index("concept_id")["concept_name"]
    texts = pd.read_csv(raw_directory / "mimic-iv_notes_training_set.csv").set_index("note_id")[
        "text"
    ]

    annotations = pd.read_csv(raw_directory / "train_annotations.csv")
    logger.info(f"""Loaded {len(annotations):,} from {raw_directory / "train_annotations.csv"}""")

    annotations["source"] = [
        texts[annotations.loc[i, "note_id"]][
            annotations.loc[i, "start"] : annotations.loc[i, "end"]
        ]
        for i in annotations.index
    ]

    annotations["concept text"] = [
        snomed[annotations.loc[i, "concept_id"]] for i in annotations.index
    ]

    annotations["source"] = [" ".join(s.split()) for s in annotations["source"]]
    output_path = interim_directory / "train_annotations_cln.csv"
    logger.info(f"Saving {len(annotations):,} cleaned annotations to {output_path}")
    annotations.to_csv(output_path, index=False)


def get_snomed_ct_synonyms(snomed_ct_directory: Path, flattened_path: Path):
    descriptions = pd.read_csv(
        Path(snomed_ct_directory)
        / "Snapshot"
        / "Terminology"
        / "sct2_Description_Snapshot-en_INT_20230531.txt",
        sep="\t",
    )

    flattened = pd.read_csv(flattened_path)

    df = flattened.merge(descriptions, left_on="concept_id", right_on="conceptId")

    df = df[["concept_id", "term", "hierarchy", "typeId"]].copy()
    df.rename(columns={"term": "concept_name"}, inplace=True)

    df["type"] = df.typeId.replace({900000000000013009: "SYN", 900000000000003001: "FSN"})
    del df["typeId"]

    return df


@app.command()
def make_synonyms(
    athena_directory: Path = raw_directory / "athena",
    snomed_ct_directory: Path = data_directory
    / "raw"
    / "SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition",
    flattened_path: Path = interim_directory / "flattened_terminology.csv",
    output_path: Path = interim_directory / "flattened_terminology_syn_snomed+omop_v5.csv",
):
    athena_directory = Path(athena_directory)
    snomed_ct_directory = Path(snomed_ct_directory)
    flattened_path = Path(flattened_path)
    output_path = Path(output_path)

    concepts = pd.read_csv(
        athena_directory / "CONCEPT.csv",
        usecols=["concept_id", "concept_name", "vocabulary_id", "concept_code", "standard_concept"],
        dtype={"concept_id": int, "concept_code": str, "standard_concept": str},
        sep="\t",
    )
    concepts.dtypes

    concept_relationships = pd.read_csv(
        athena_directory / "CONCEPT_RELATIONSHIP.csv",
        usecols=["concept_id_1", "concept_id_2", "relationship_id"],
        dtype={"concept_id_1": int, "concept_id_2": int},
        sep="\t",
    )

    code_list = pd.read_csv(flattened_path, usecols=["concept_id"]).concept_id.astype(str).tolist()
    logger.debug(f"Subsetting Athena synonyms to {len(code_list):,} SNOMED CT codes")

    athena_synonyms = (
        concepts.loc[
            (concepts.vocabulary_id == "SNOMED") & (concepts.concept_code.isin(code_list)),
            ["concept_id", "concept_code"],
        ]
        .merge(
            concept_relationships.loc[concept_relationships.relationship_id == "Maps to"],
            left_on="concept_id",
            right_on="concept_id_2",
        )
        .drop(columns=["concept_id", "concept_id_2", "relationship_id"])
        .merge(
            concepts.loc[
                ~concepts.vocabulary_id.isin(invalid_vocabs),
                ["concept_id", "concept_name", "vocabulary_id"],
            ],
            left_on="concept_id_1",
            right_on="concept_id",
            suffixes=("", "_other"),
            validate="many_to_one",
        )
        .drop(columns=["concept_id", "concept_id_1", "vocabulary_id"])
        .drop_duplicates(keep="first")
        .rename(columns={"concept_code": "concept_id"})
    )

    athena_synonyms["concept_id"] = athena_synonyms.concept_id.astype(int)
    athena_synonyms.sort_values("concept_id", inplace=True)
    logger.debug(f"Loaded {len(athena_synonyms):,} synonyms from Athena.")

    snomed_synonyms = get_snomed_ct_synonyms(snomed_ct_directory, flattened_path)
    logger.debug(f"Loaded {len(snomed_synonyms):,} synonyms from SNOMED CT.")

    # lookup hierarchy from snomed
    athena_synonyms = athena_synonyms.merge(
        snomed_synonyms.set_index("concept_id")["hierarchy"], left_on="concept_id", right_index=True
    )

    synonyms = (
        pd.concat(
            [
                athena_synonyms[["concept_id", "concept_name", "hierarchy"]].assign(type="SYN"),
                snomed_synonyms,
            ]
        )
        .drop_duplicates(keep="first")
        .sort_values(["concept_id", "type", "hierarchy", "concept_name"])
        .reset_index(drop=True)
    )

    logger.debug(f"Saving {len(synonyms):,} synonyms to {output_path}")
    synonyms.to_csv(output_path, index=False)
    return synonyms


@app.command()
def make_abbreviations():
    logger.info("Loading abbreviations...")
    abbr = pd.read_csv(
        raw_directory / "medical_abbreviations.csv",
        usecols=["Abbreviation/Shorthand", "Meaning"],
    )
    logger.info("Loading synonyms...")
    snomed = pd.read_csv(interim_directory / "flattened_terminology_syn_snomed+omop_v5.csv")

    abbr_meanings = abbr.Meaning.tolist()
    snomed_concept_names = snomed.concept_name.tolist()

    snomed_to_abbr = []
    logger.info(f"Processing {len(abbr_meanings):,} meanings...")
    for abbr_meaning in abbr_meanings:
        for concept_name in snomed_concept_names:
            if str(abbr_meaning).lower() == str(concept_name).lower():
                snomed_to_abbr.append([concept_name, abbr_meaning])
                break
            elif str(abbr_meaning).lower() == str(concept_name).lower().split(" (")[0]:
                if (
                    len(str(concept_name).lower().split(" (")[-1]) <= 10
                    and len(str(concept_name).lower().split(" (")) <= 2
                ):
                    snomed_to_abbr.append([concept_name, abbr_meaning])
                    break
                else:
                    continue
            else:
                continue

    snomed_to_abbr_df = pd.DataFrame(snomed_to_abbr, columns=["concept_name", "Meaning"])
    snomed_to_abbr_df = snomed_to_abbr_df.merge(
        snomed[["concept_id", "concept_name"]], on="concept_name", how="left"
    )
    snomed_to_abbr_df = snomed_to_abbr_df.merge(abbr, on="Meaning", how="left")
    snomed_to_abbr_df.drop_duplicates(
        subset=["concept_name", "Meaning", "concept_id"], inplace=True
    )

    output_path = interim_directory / "abbreviations_snomed_v5.csv"
    logger.info(f"Saving output to {output_path}")
    snomed_to_abbr_df.to_csv(output_path, index=False)


@app.command()
def make_abbr_dict():
    logger.info("Loading abbreviations, notes, and annotations...")
    abbr = pd.read_csv(interim_directory / "abbreviations_snomed_v5.csv")
    abbr = abbr[abbr["Abbreviation/Shorthand"].str.len() > 3].drop_duplicates(
        "Abbreviation/Shorthand", keep="first"
    )
    texts = pd.read_csv(raw_directory / "mimic-iv_notes_training_set.csv").set_index("note_id")[
        "text"
    ]
    annotations = pd.read_csv(interim_directory / "train_annotations_cln.csv")
    abbr_dict = {("any", k): v for k, v in abbr[["Abbreviation/Shorthand", "concept_id"]].values}

    sno_syns, sno_fsn, cid_to_type = get_snomed_synonyms()

    allowed_sec = get_allowed_sections(texts, annotations, common_headers, cid_to_type)
    limit_any_to_allowed_sections(abbr_dict, allowed_sec, cid_to_type)

    output_path = interim_directory / "abbr_dict.pkl"
    logger.info(f"Saving dictionary of {len(abbr_dict):,} abbreviations to {output_path}")
    with output_path.open("wb") as f:
        pickle.dump(abbr_dict, f)


@app.command()
def make_term_extension():
    logger.info("Loading SNOMED CT relationships, descriptions, and flattened terminology...")
    relationships = pd.read_csv(
        raw_directory
        / "SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition"
        / "Snapshot"
        / "Terminology"
        / "sct2_Relationship_Snapshot_INT_20230531.txt",
        dtype={"sourceId": int, "typeId": int, "destinationId": int},
        sep="\t",
    )
    descriptions = pd.read_csv(
        raw_directory
        / "SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition"
        / "Snapshot"
        / "Terminology"
        / "sct2_Description_Snapshot-en_INT_20230531.txt",
        sep="\t",
        dtype={"conceptId": int, "typeId": int, "active": int},
        quoting=csv.QUOTE_NONE,
    )

    terminology = pd.read_csv(
        interim_directory / "flattened_terminology_syn_snomed+omop_v5.csv",
        dtype={"concept_id": int},
    )

    logger.info("Processing...")
    relationships = relationships[relationships["active"] == 1]
    relationships = relationships[
        relationships["sourceId"].isin(terminology["concept_id"])
        | relationships["destinationId"].isin(terminology["concept_id"])
    ]

    descriptions = descriptions.loc[
        (descriptions["active"] == 1) & (descriptions["typeId"] == 900000000000003001),
        ["conceptId", "term"],
    ].drop_duplicates(keep="first")

    relationships = relationships[["sourceId", "typeId", "destinationId"]].drop_duplicates(
        keep="first"
    )

    relationships = (
        relationships.merge(
            descriptions.rename(columns={"term": "sourceName"}),
            left_on="sourceId",
            right_on="conceptId",
        )
        .merge(
            descriptions.rename(columns={"term": "destinationName"}),
            left_on="destinationId",
            right_on="conceptId",
        )
        .merge(
            descriptions.rename(columns={"term": "typeName"}),
            left_on="typeId",
            right_on="conceptId",
        )
    )

    relationships = relationships.loc[
        relationships["typeId"] == 116680003,  # Is A
        ["sourceId", "sourceName", "typeId", "typeName", "destinationId", "destinationName"],
    ]

    stop_words = set(
        """a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with""".split()
    )

    res = []
    for i, row in enumerate(relationships.itertuples()):
        s_term = re.split(r"\(", row.sourceName.lower())[0]
        d_term = re.split(r"\(", row.destinationName.lower())[0]

        s_words = re.split(r"\s", s_term)
        d_words = re.split(r"\s", d_term)

        s_min_d = set(s_words) - set(d_words) - stop_words
        d_min_s = set(d_words) - set(s_words) - stop_words

        if len(d_min_s) == 0 and len(s_min_d) == 1:
            res.append(
                {
                    "generalId": row.destinationId,
                    "generalName": row.destinationName,
                    "specificId": row.sourceId,
                    "specificName": row.sourceName,
                    "typeName": row.typeName,
                    "additionalWord": list(s_min_d)[0],
                }
            )

    logger.info(
        f"""Saving {len(res):,} term extensions to {interim_directory / "term_extension.csv"}"""
    )
    pd.DataFrame(res).to_csv(interim_directory / "term_extension.csv", index=False)


@app.command()
def make_unigrams():
    discharge = pd.read_csv(raw_directory / "discharge.csv.gz", usecols=["text"])

    text = "\n".join(discharge["text"]).lower()
    all_text = " ".join(text.split())
    text_counter = Counter(all_text.split())

    th = 20_000
    snomed_syns, sno_fsn, z_ = get_snomed_synonyms(min_len=1, max_len=1, fsn_only=True)
    unigram_dict_20k = {k: v for k, v in snomed_syns.items() if text_counter[k[1]] < th}
    with (interim_directory / "snomed_unigrams_annotation_dict_20k_v4_fsn.pkl").open("wb") as fp:
        pickle.dump(unigram_dict_20k, fp)

    th = 3000
    snomed_syns, sno_fsn, z_ = get_snomed_synonyms(min_len=1, max_len=1, fsn_only=False)
    unigram_dict_3k = {k: v for k, v in snomed_syns.items() if text_counter[k[1]] < th}

    with (interim_directory / "snomed_unigrams_annotation_dict_3k_v4_new.pkl").open("wb") as fp:
        pickle.dump(unigram_dict_3k, fp)


if __name__ == "__main__":
    app()
