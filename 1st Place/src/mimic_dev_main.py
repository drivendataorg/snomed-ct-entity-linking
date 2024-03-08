# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:29:48 2024

@author: Yonatan
"""

import os
import pickle
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from mimic_common import common_headers
from mimic_predict import make_predictions
from mimic_submission_main import run_main
from mimic_train import train
from note_scoring import iou_per_note
from scoring import iou_per_class
from tqdm import tqdm

submission_dir = Path(__file__) / "submissions"
data_directory = Path(__file__).parent.parent / "data"
src_directory = Path(__file__).parent


def mimic_train_test(
    texts=None,
    annotations=None,
    run_name="default",
    train_size=150,
    test_size=None,
    headers=common_headers,
):
    if texts is None:
        texts = pd.read_csv(data_directory / "raw" / "mimic-iv_notes_training_set.csv").set_index(
            "note_id"
        )["text"]
    if annotations is None:
        annotations = pd.read_csv(data_directory / "interim" / "train_annotations_cln.csv")

    np.random.seed(12345)
    ids = list(texts.index)
    np.random.shuffle(ids)
    train_ids = ids[:train_size]
    test_ids = ids[train_size:]
    if test_size is not None:
        test_ids = ids[-test_size:]

    pred, d, uc_d, scores = do_train_test(
        texts, annotations, headers, run_name, train_ids, test_ids
    )

    return pred, d, uc_d


def cross_validation(
    texts=None, annotations=None, n_folds=5, run_name="cv", headers=common_headers
):
    if texts is None:
        texts = pd.read_csv(data_directory / "raw" / "mimic-iv_notes_training_set.csv").set_index(
            "note_id"
        )["text"]
    if annotations is None:
        annotations = pd.read_csv(data_directory / "interim" / "train_annotations_cln.csv")

    np.random.seed(123456)
    ids = list(texts.index)
    np.random.shuffle(ids)
    fold_size = int(len(ids) / n_folds)
    scores = []
    preds = []
    fold_ids = {}
    for i in range(n_folds):
        if i < n_folds - 1:
            test_ids = ids[i * fold_size : (i + 1) * fold_size]
        else:
            test_ids = ids[i * fold_size :]
        train_ids = [tid for tid in ids if tid not in test_ids]
        fold_ids[i] = {"test": test_ids, "train": train_ids}
        p, d, uc_d, s = do_train_test(
            texts, annotations, headers, f"{run_name}_{i}", train_ids, test_ids
        )
        scores.append(s)
        preds.append(p)
    with open(f"../debug/{run_name}_fold_ids.pkl", "wb") as f:
        pickle.dump(fold_ids, f)

    return pd.concat(preds), scores


def do_train_test(texts, annotations, headers, run_name, train_ids, test_ids):
    d, uc_d = train(texts[texts.index.isin(train_ids)], annotations, headers, run_name)

    pred = make_predictions(texts[texts.index.isin(test_ids)], d, uc_d, run_name=run_name)
    return pred, d, uc_d, print_scores(pred, annotations)


def do_predict(d, uc_d={}, texts=None, annotations=None, test_size=54, run_name="default"):
    if texts is None:
        texts = pd.read_csv(data_directory / "raw" / "mimic-iv_notes_training_set.csv").set_index(
            "note_id"
        )["text"]
    if annotations is None:
        annotations = pd.read_csv(data_directory / "intermi" / "train_annotations_cln.csv")

    np.random.seed(12345)
    ids = list(texts.index)
    np.random.shuffle(ids)

    test_ids = ids[-test_size:]
    pred = make_predictions(texts[texts.index.isin(test_ids)], d, uc_d, run_name=run_name)
    print_scores(pred, annotations)
    return pred


def print_scores(pred, annotations):
    annotated = annotations[annotations["note_id"].isin(pred["note_id"].unique())]
    ious_c = pd.Series(iou_per_class(pred, annotated))
    ious_n = pd.Series(iou_per_note(pred, annotated))
    print("score by concept", ious_c.mean(), "; score by note", ious_n.mean())
    return ious_c.mean(), ious_n.mean()


def make_submission(submission_number="", no_check=False, submission_path: Optional[Path] = None):
    if submission_path is None:
        submission_path = submission_dir / f"submission{submission_number}"

    if not submission_path.exists():
        print(f"{submission_path} does not exist")
        return None

    texts = pd.read_csv(data_directory / "raw" / "mimic-iv_notes_training_set.csv").set_index(
        "note_id"
    )["text"]
    annotations = pd.read_csv(data_directory / "interim" / "train_annotations_cln.csv")
    kiri_dicts = make_kiri_dicts(texts, annotations, submission_path)
    print("size of dict", len(kiri_dicts[0]), len(kiri_dicts[1]), " (should be around 1M)")
    shutil.copyfile(src_directory / "mimic_submission_main.py", submission_path / "main.py")
    shutil.copyfile(src_directory / "mimic_predict.py", submission_path / "mimic_predict.py")
    shutil.copyfile(src_directory / "mimic_common.py", submission_path / "mimic_common.py")
    shutil.copyfile(
        src_directory / "mimic_postprocess_attributes.py",
        submission_path / "mimic_postprocess_attributes.py",
    )
    shutil.copyfile(
        data_directory / "interim" / "abbr_dict.pkl", submission_path / "assets" / "abbr_dict.pkl"
    )
    shutil.copyfile(
        data_directory / "interim" / "term_extension.csv",
        submission_path / "assets" / "term_extension.csv",
    )
    if no_check:
        return

    cwd = os.getcwd()
    os.chdir(submission_path)
    print("running submission main")
    run_main()
    os.chdir(cwd)
    submission = pd.read_csv(submission_path / "submission.csv")
    print("number of predictions", len(submission), " (should be around 60k)")
    overlaps = check_for_overlaps(submission)
    if overlaps is not None:
        df, i, j = overlaps
        print(f"overlaps found, rows {i} and {j} in overlaps.csv")
        df.to_csv(submission_path / "overlaps.csv")

    ious_c = pd.Series(iou_per_class(submission, annotations))
    ious_n = pd.Series(iou_per_note(submission, annotations))
    print("score by concept", ious_c.mean(), "; score by note", ious_n.mean())


def spans_overlap(s1, s2):
    return s1["start"] <= s2["start"] < s1["end"]


def check_for_overlaps(pred):
    for ni in tqdm(pred["note_id"].unique()):
        df = pred.query(f'note_id == "{ni}"')
        df = df.sort_values("start")
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if spans_overlap(df.iloc[i], df.iloc[j]):
                    return df, i, j
    print("No overlaps")
    return None


def make_kiri_dicts(texts, annotations, submission_path: Path, headers: list[str] = common_headers):
    texts = texts.str.lower()
    headers = [h.lower() for h in headers]
    if "source orig" not in annotations:
        annotations["source orig"] = annotations["source"]
    annotations["source"] = annotations["source"].str.lower()
    kiri_dicts = train(texts, annotations, common_headers)
    with (submission_path / "assets" / "kiri_dicts.pkl").open("wb") as f:
        pickle.dump(kiri_dicts, f)
    return kiri_dicts


if __name__ == "__main__":
    pred, d, d_uc = mimic_train_test()
    data = pd.read_pickle("../debug/default.pkl")
    for k in data:
        globals()[k] = data[k]
