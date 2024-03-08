import bisect
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def load_sctid_syn(sctid_syn_parh: Path):
    with open(sctid_syn_parh, "r") as f:
        data = json.load(f)
        return set(map(int, data.keys()))


def add_concept_class(ann_df, sctid_syn_dir: Path):
    p_cids = load_sctid_syn(Path(sctid_syn_dir) / "proc_sctid_syn.json")
    p_cids.add(71388002)
    f_cids = load_sctid_syn(Path(sctid_syn_dir) / "find_sctid_syn.json")
    b_cids = load_sctid_syn(Path(sctid_syn_dir) / "body_sctid_syn.json")
    snomed_class = []
    for i, r in ann_df.iterrows():
        cid = r.concept_id
        if cid in p_cids:
            label = "proc"
        elif cid in b_cids:
            label = "body"
        elif cid in f_cids:
            label = "find"
        else:
            label = "???"
            raise ValueError(f"unknown concept_id: {cid}")
        snomed_class.append(label)

    ann_df["cls"] = snomed_class
    return ann_df


def get_labels(starts, ends, spans):
    """Convert offsets to sequence labels in BIO format."""
    labels = ["O"] * len(starts)
    spans = sorted(spans)
    for s, e, l in spans:
        li = bisect.bisect_left(starts, s)
        ri = bisect.bisect_left(starts, e)
        ni = len(labels[li:ri])
        labels[li] = f"B-{l}"
        labels[li + 1 : ri] = [f"I-{l}"] * (ni - 1)
    return labels


class Labeler:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def fix_annotation(self, text, ann_df):
        trail = False
        idxs = []
        for i, c in enumerate(text):
            if c == " ":
                if trail:
                    idxs.append(i)
                trail = True
            else:
                trail = False

        label_char = np.zeros(len(text))
        label_char[idxs] = 1
        label_char = np.cumsum(label_char).astype(int)

        char_spans = []
        for i, r in ann_df.iterrows():
            span = [r.start, r.end, r.cls]
            char_spans.append(span)

        # fix spans, delete trailing spaces from text
        for i, span in enumerate(char_spans):
            s, e, c = span
            s -= label_char[s]
            e -= label_char[e]
            char_spans[i] = [s, e, c]
        return char_spans

    def preprocess_text(self, note_ann_df, text):
        text = re.sub(r"[^a-zA-Z0-9\s.,:\/]", " ", text)
        char_spans = []
        for _, r in note_ann_df.iterrows():
            span = [r.start, r.end, r.cls]
            char_spans.append(span)

        encoded = self.tokenizer(
            text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        input_ids = encoded["input_ids"]
        off = np.array(encoded["offset_mapping"])
        starts = off[:, 0]
        ends = off[:, 1]

        labels = get_labels(starts, ends, char_spans)
        return text, input_ids, labels


def convert_labels_tokens(tokenizer, note_df, ann_df):
    preproc = Labeler(tokenizer)

    agg = ann_df.groupby("note_id")
    # to dict
    anns = {}
    for note_id, r in tqdm(agg):
        anns[note_id] = r

    res = defaultdict(list)
    for i, r in tqdm(note_df.iterrows(), total=len(note_df)):
        note_id = r.note_id
        text = r.text
        ann_note_df = anns[note_id]
        try:
            t, i, l = preproc.preprocess_text(ann_note_df, text)
            res["note_id"].append(note_id)
            res["text"].append(t)
            res["input_ids"].append(i)
            res["labels"].append(l)
            res["fold"].append(r.fold)
        except Exception as e:
            print(e)
            print(note_id)

    tdf = pd.DataFrame(res)
    return tdf


def parallel_convert_labels_tokens(tokenizer, note_df, ann_df, n_jobs=4):
    from joblib import Parallel, delayed

    note_chunks = np.array_split(note_df, n_jobs)
    ann_df_gg = ann_df.groupby("note_id")
    ann_df_dict = {k: v for k, v in ann_df_gg}
    ann_chunks = []
    for chunk in note_chunks:
        note_ids = chunk.note_id
        ann_chunk = pd.concat([ann_df_dict[nid] for nid in note_ids])
        ann_chunks.append(ann_chunk)

    res = Parallel(n_jobs=n_jobs)(
        delayed(convert_labels_tokens)(tokenizer, note_chunk, ann_chunk)
        for note_chunk, ann_chunk in zip(note_chunks, ann_chunks)
    )
    tdf = pd.concat(res)
    return tdf


class PreprocessedDataset(Dataset):
    def __init__(self, cfg, tokenizer, df, train=False, fold=None, repeat=1, feature=None):
        self.cfg = cfg
        if fold is not None:
            if isinstance(fold, int):
                df = df[df.fold == fold]
            elif isinstance(fold, list):
                df = df[df.fold.isin(fold)]

        self.df = df
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.train = train
        self.feature = feature

        self.label2id = {
            "O": 0,
            "B-find": 1,
            "I-find": 2,
            "B-proc": 3,
            "I-proc": 4,
            "B-body": 5,
            "I-body": 6,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.note_dict = df.set_index("note_id").to_dict()["text"]

        if not self.train:
            # split input_ids and label_str into 2 parts, then concatenate them
            res = []
            for i, r in df.iterrows():
                input_ids = r.input_ids
                label_str = r.labels
                input_ids1 = input_ids[: len(input_ids) // 2]
                label_str1 = label_str[: len(label_str) // 2]
                input_ids2 = input_ids[len(input_ids) // 2 :]
                label_str2 = label_str[len(label_str) // 2 :]
                row1 = [r.note_id, r.text, input_ids1, label_str1, r.fold]
                row2 = [r.note_id, r.text, input_ids2, label_str2, r.fold]
                res.append(row1)
                res.append(row2)
            self.df = pd.DataFrame(res)
            self.df.columns = ["note_id", "text", "input_ids", "labels", "fold"]
        print("Dataset len:", len(self.df))
        print("Average token len:", self.df.input_ids.apply(len).mean())

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise StopIteration
        idx = idx % len(self.df)

        _, text, input_ids, label_str, _ = self.df.iloc[idx]

        max_len = self.cfg.max_len - 2

        if len(input_ids) > max_len:
            if self.train:
                offset_fixed = np.random.randint(0, len(input_ids) - max_len)
            else:
                offset_fixed = 0

            input_ids = input_ids[offset_fixed : offset_fixed + max_len]
            label_str = label_str[offset_fixed : offset_fixed + max_len]
        else:
            offset_fixed = 0

        labels_int = [self.label2id[l] for l in label_str]
        labels_int = [-100] + labels_int + [-100]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_int, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class ChunkedDataset:
    def __init__(self, tokenizer, fold, df, max_len, repeat=1):
        self.label2id = {
            "O": 0,
            "B-find": 1,
            "I-find": 2,
            "B-proc": 3,
            "I-proc": 4,
            "B-body": 5,
            "I-body": 6,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        if fold is not None:
            if isinstance(fold, int):
                df = df[df.fold == fold]
            elif isinstance(fold, list):
                df = df[df.fold.isin(fold)]

        _max_len = max_len - 2
        # split notes into chunks of max_len
        chunked_rows = []
        for i, row in df.iterrows():
            ids = row["input_ids"]
            labels = row["labels"]

            for i in range(0, len(ids), _max_len):
                chunked_rows.append(
                    {
                        "fold": row["fold"],
                        "ids": ids[i : i + _max_len],
                        "labels": labels[i : i + _max_len],
                    }
                )
        df = pd.DataFrame(chunked_rows)
        print(f"chunked into {len(df)} rows")
        self.df = df
        self.tokenizer = tokenizer
        self.repeat = repeat
        self.max_len = max_len

    def __len__(self):
        return len(self.df) * self.repeat

    def __getitem__(self, idx):
        if idx > self.__len__():
            raise StopIteration
        idx = idx % len(self.df)

        row = self.df.iloc[idx]
        input_ids = row["ids"]
        label_str = row["labels"]

        labels_int = [self.label2id[l] for l in label_str]
        labels_int = [-100] + labels_int + [-100]
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels_int, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
