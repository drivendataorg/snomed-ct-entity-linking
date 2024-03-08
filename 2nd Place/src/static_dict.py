import pickle
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


def get_true_header_indices(text, true_headers):
    text = re.sub("\n", " ", text.lower())
    true_header_indices = {}
    for true_header in true_headers:
        pos = text.find(true_header)
        if pos != -1:
            true_header_indices[true_header] = pos
    true_header_indices = dict(sorted(true_header_indices.items(), key=lambda item: item[1]))
    true_header_indices = OrderedDict(true_header_indices)
    return true_header_indices


def calc_header_span(df):
    true_headers = [
        "past medical history:",
        "allergies:",
        "history of present illness:",
        "physical exam:",
        "admission date:  discharge date:",
        "attending:",
        "major surgical or invasive procedure:",
        "family history:",
        "discharge disposition:",
        "discharge condition:",
        "discharge instructions:",
        "name:  unit no:",
        "social history:",
        "chief complaint:",
        "pertinent results:",
        "discharge medications:",
        "medications on admission:",
        "___ on admission:",
        "discharge diagnosis:",
        "followup instructions:",
        "brief hospital course:",
        "facility:",
        "impression:",
    ]
    res = {}
    for i, row in df.iterrows():
        text = row["text"]
        headers = get_true_header_indices(text.lower(), true_headers)
        headers_spans = {}
        for header, start in headers.items():
            i = list(headers).index(header)
            if i == len(headers) - 1:
                end = len(text)
            else:
                next_header = list(headers)[i + 1]
                end = headers[next_header]
            headers_spans[header] = (start, end)
        res[row.note_id] = headers_spans
    return res


class StaticDict:
    ignore_headers = [
        "medications on admission:",
        "___ on admission:",
        "discharge medications:",
    ]
    ignore_term_list = []

    def __init__(self, data: pd.DataFrame, ann: pd.DataFrame, filt_headers: bool = False):
        self.filt_headers = filt_headers
        self.static_dict = self.prepare_concepts_dict(data, ann)
        self.terms = self.term_to_concepts(self.static_dict)
        self.concepts = self.concepts_to_term(self.static_dict)
        self.most_common_concept = self.term_to_most_common_concept(self.static_dict)

    @staticmethod
    def preprocess_text(text: str, keep_len: bool = False):
        t = text.lower()
        t = t.replace("\n", " ")
        t = re.sub("[^a-z0-9]", " ", t)
        if not keep_len:
            t = re.sub("\s+", " ", t)
            t = t.strip()
        return t

    def ignore_by_headers(self, span, header_spans):
        for ignore_header in self.ignore_headers:
            header_span = header_spans.get(ignore_header, None)
            if (header_span is not None) and max(span[0], header_span[0]) <= min(
                span[1], header_span[1]
            ):
                return True
        return False

    def prepare_concepts_dict(self, notes: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
        notes = notes.set_index("note_id")
        note_ids = annotation.note_id.unique()
        concepts = []
        for note_id in tqdm(note_ids, desc="Processing notes", total=len(note_ids)):
            note = notes.loc[note_id].text
            for i, row in annotation[annotation.note_id == note_id].iterrows():
                term = self.preprocess_text(note[row.start : row.end], keep_len=False)
                if term in self.ignore_term_list:
                    continue
                concepts.append([note_id, row.concept_id, term])
        concepts = pd.DataFrame(concepts, columns=["note_id", "concept_id", "term"])
        # concepts = concepts.drop_duplicates()
        return concepts

    @staticmethod
    def term_to_most_common_concept(concepts: pd.DataFrame) -> dict:
        return (
            concepts.drop(columns="note_id")
            .groupby("term")
            .concept_id.apply(lambda x: Counter(list(x)).most_common(1)[0][0])
            .to_dict()
        )

    @staticmethod
    def term_to_concepts(concepts: pd.DataFrame) -> dict:
        return (
            concepts.drop(columns="note_id")
            .groupby("term")
            .apply(lambda x: x.concept_id.unique().tolist())
            .to_dict()
        )

    @staticmethod
    def concepts_to_term(concepts: pd.DataFrame) -> dict:
        return (
            concepts.drop(columns="note_id")
            .groupby("concept_id")
            .apply(lambda x: x.term.unique().tolist())
            .to_dict()
        )

    def get_submission_from_static_dict(
        self, notes: pd.DataFrame, most_common_concept: Optional[dict] = None
    ) -> pd.DataFrame:
        spans = []
        if most_common_concept is None:
            most_common_concept = self.most_common_concept
        patterns_list = [w.replace(" ", r"\b\s+\b") for w in most_common_concept.keys()]
        patterns = r"\b(" + r"|".join(patterns_list) + r")\b"
        header_span = calc_header_span(notes)
        for i, note in tqdm(notes.iterrows(), total=len(notes), desc="notes"):
            matches = re.finditer(patterns, self.preprocess_text(note.text, keep_len=True))
            for match in matches:
                full_term = match.group()
                term = self.preprocess_text(full_term, keep_len=False)
                concept_id = most_common_concept[term]
                if self.ignore_by_headers((match.start(), match.end()), header_span[note.note_id]):
                    continue
                spans.append(
                    (note.note_id, term, match.start(), match.end(), concept_id, full_term)
                )
        spans = pd.DataFrame(
            spans, columns=["note_id", "term", "start", "end", "concept_id", "full_term"]
        )
        return spans

    def calc_ratio(self, notes: pd.DataFrame) -> pd.DataFrame:
        static_submission = self.get_submission_from_static_dict(notes)
        all_positive = (
            static_submission.drop(columns=["note_id", "start", "end", "full_term"])
            .groupby("term")
            .count()
        )
        true_positive = self.static_dict.drop(columns=["note_id"]).groupby("term").count()
        ratio = (true_positive / all_positive).dropna()
        return ratio

    @staticmethod
    def filt_by_ratio(concepts, ratio, threshold=0.25):
        ratio = ratio[ratio > threshold].dropna()
        concepts = concepts[concepts.term.isin(ratio.index)]
        return concepts


def get_most_common_concept(static_dict_path: Path, notes: pd.DataFrame, annotation: pd.DataFrame):
    static_dict = StaticDict(notes, annotation, True)
    precalc_ratio = static_dict.calc_ratio(notes)
    concepts = static_dict.filt_by_ratio(static_dict.static_dict, precalc_ratio, threshold=0.9)
    most_common_concept = static_dict.term_to_most_common_concept(concepts)
    with open(static_dict_path, "wb") as f:
        pickle.dump(most_common_concept, f)
