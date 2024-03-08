# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:29:11 2024

@author: Yonatan
"""

import pickle
from pathlib import Path

import pandas as pd
from mimic_common import annotate_with_dict, common_headers, remove_overlaps
from mimic_postprocess_attributes import postprocess_annotations
from tqdm import tqdm

data_directory = Path(__file__).parent.parent / "data"


def predict(texts, headers, d, submission, run_name):
    pred = []
    if not submission:
        print("generating predictions")
    for i in tqdm(texts.index, disable=submission):
        pred.append(annotate_with_dict(texts[i], d, headers, i))
    pred = pd.concat(pred)
    if not submission:
        pred.to_csv(f"../debug/{run_name}_pred.csv")
    return pred


def get_case_sensitive_dict():
    csd = {
        (("other", "Pertinent Results:"), "K"): 312468003,
        ("any", "T"): 105723007,
        (("other", "Pertinent Results:"), "Mg"): 271285000,
        ("Physical Exam:", "RA"): 722742002,
        (("other", "Pertinent Results:"), "Plt"): 61928009,
        (("other", "Pertinent Results:"), "MR"): 48724000,
    }
    return csd


def join_predictions(predictions):
    predictions = pd.concat(predictions)
    no_overlaps = []
    for nid in predictions["note_id"].unique():
        df = predictions.query(f'note_id == "{nid}"')
        no_overlaps.append(remove_overlaps(df))
    return pd.concat(no_overlaps)


def get_abbr_dict(submission: bool):
    if submission:
        path = Path("assets") / "abbr_dict.pkl"
    else:
        path = data_directory / "interim" / "abbr_dict.pkl"
    with path.open("rb") as f:
        abbr_dict = pickle.load(f)
    return abbr_dict


def get_attr_file(submission):
    if submission:
        fn = "assets/term_extension.csv"
    else:
        fn = "../term_extension.csv"
    df = pd.read_csv(fn)
    return df


def make_predictions(texts, d, uc_dict, submission=False, run_name="default"):
    assert isinstance(uc_dict, dict)
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in common_headers]

    pred_lc = predict(texts_lc, headers, d, submission, run_name + "_lc")
    uc_dict.update(get_case_sensitive_dict())
    abbr_dict = get_abbr_dict(submission)
    abbr_dict.update(uc_dict)
    uc_dict = abbr_dict
    pred_uc = predict(texts, common_headers, uc_dict, submission, run_name + "_uc")
    pred = join_predictions((pred_lc, pred_uc))

    att_df = get_attr_file(submission)
    pred = postprocess_annotations(texts, pred, att_df, submission)

    return pred
