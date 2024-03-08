# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:50:10 2024

@author: Yonatan
"""

import pickle
from pathlib import Path

import pandas as pd
from mimic_predict import make_predictions

NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")


def run_main():
    with open("assets/kiri_dicts.pkl", "rb") as f:
        kiri_dicts = pickle.load(f)

    texts = pd.read_csv(NOTES_PATH).set_index("note_id")["text"]
    pred = make_predictions(texts, kiri_dicts[0], kiri_dicts[1], submission=True)

    pred = pred[["note_id", "start", "end", "concept_id"]]
    pred.to_csv(SUBMISSION_PATH, index=False)


if __name__ == "__main__":
    run_main()
