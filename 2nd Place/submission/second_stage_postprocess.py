import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def add_text(df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    notes_dict = dict(zip(notes_df.note_id, notes_df.text))
    for i, row in df.iterrows():
        note = notes_dict[row.note_id]
        df.loc[i, "text"] = note[row.start : row.end]
    return df


def static_preprocess(text):
    t = text.lower()
    t = t.replace("\n", "")
    t = re.sub("[^a-z]", "", t)
    t = re.sub("\s+", " ", t)
    t = t.strip()
    return t


def get_static_dict(trdf):
    static_dict_all = defaultdict(list)
    for _, row in trdf.iterrows():
        t = row.text
        t = static_preprocess(t)
        static_dict_all[t].append(row.concept_id)

    static_dict = {}
    for k, v in static_dict_all.items():
        if len(set(v)) > 1:
            cc = Counter(v)
            static_dict[k] = cc.most_common(1)[0][0]
        else:
            static_dict[k] = v[0]

    return static_dict


def choose_concepts(df, note_df, data_path: Path, annotations_path: Path):
    df = df.reset_index(drop=True)
    df.loc[:, "sap_concept_id"] = df.sap_cids.apply(lambda x: x[0])

    df.loc[:, "sapscore"] = df.sap_scores.apply(lambda x: x[0])

    dfn = pd.read_csv(data_path)
    trdf = pd.read_csv(annotations_path)

    trdf = add_text(trdf, dfn)
    train_set = set(trdf.concept_id.unique())

    df = add_text(df, note_df)
    df["is_train_sap"] = df.sap_concept_id.isin(train_set)
    df["len"] = df.end - df.start

    static_dict = get_static_dict(trdf)

    thr1 = 0.90

    xx_all = df.copy()
    sap_train = xx_all.is_train_sap
    xx_all.loc[sap_train, "sapscore"] += 0.05

    blacklist = (
        "left right your of with in the and to or no person date for on was a is at were".split()
    )
    is_blacklisted = xx_all.text.apply(lambda x: static_preprocess(x)).isin(blacklist)
    xx_all = xx_all[~is_blacklisted]

    xx_all.loc[:, "concept_id"] = -1
    xx_all.loc[:, "score"] = xx_all.loc[:, "sapscore"]

    for i, row in xx_all.iterrows():
        t = row.text
        t = static_preprocess(t)
        if t in static_dict and xx_all.loc[i, "score"] < 1.0:
            xx_all.loc[i, "concept_id"] = static_dict[t]

    static = xx_all.concept_id != -1
    xx_static = xx_all[static].copy()
    xx = xx_all[~static].copy()
    xx.loc[:, "concept_id"] = xx.sap_concept_id
    all_score = xx.score > thr1
    xx = xx[all_score]
    xx = xx[xx.len > 2].copy()
    xx = pd.concat([xx, xx_static])
    xx = xx[xx.len > 1].copy()
    return xx
