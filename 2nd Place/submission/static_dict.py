import pickle
import re
from pathlib import Path

import pandas as pd
from first_stage_postprocess import calc_header_span
from tqdm.auto import tqdm

ignore_headers = [
    "medications on admission:",
    "___ on admission:",
    "discharge medications:",
]


def preprocess_text(text: str, keep_len: bool = False):
    t = text.lower()
    t = t.replace("\n", " ")
    t = re.sub("[^a-z0-9]", " ", t)
    if not keep_len:
        t = re.sub("\s+", " ", t)
        t = t.strip()
    return t


def ignore_by_headers(span, header_spans):
    for ignore_header in ignore_headers:
        header_span = header_spans.get(ignore_header, None)
        if (header_span is not None) and max(span[0], header_span[0]) <= min(
            span[1], header_span[1]
        ):
            return True
    return False


def get_submission_from_static_dict(notes: pd.DataFrame, most_common_concept: dict) -> pd.DataFrame:
    spans = []
    patterns_list = [w.replace(" ", r"\b\s+\b") for w in most_common_concept.keys()]
    patterns = r"\b(" + r"|".join(patterns_list) + r")\b"
    header_span = calc_header_span(notes)
    for i, note in tqdm(notes.iterrows(), total=len(notes), desc="get_submission"):
        matches = re.finditer(patterns, preprocess_text(note.text, keep_len=True))
        for match in matches:
            full_term = match.group()
            term = preprocess_text(full_term, keep_len=False)
            concept_id = most_common_concept[term]
            if ignore_by_headers((match.start(), match.end()), header_span[note.note_id]):
                continue
            spans.append((note.note_id, term, match.start(), match.end(), concept_id, full_term))
    spans = pd.DataFrame(
        spans, columns=["note_id", "term", "start", "end", "concept_id", "full_term"]
    )
    return spans


def merge_submissions(submission, submission_static):
    new_submission = []
    submission = submission.copy()
    submission_static = submission_static.copy()
    submission["is_static"] = False
    submission_static["is_static"] = True
    for note_id in tqdm(
        submission.note_id.unique(),
        desc="merge_submissions",
        total=len(submission.note_id.unique()),
    ):
        submission_note = submission[submission.note_id == note_id]
        submission_static_note = submission_static[submission_static.note_id == note_id]
        filt_note = submission_static_note.start < 0
        for i, row in submission_note.iterrows():
            filt_note |= submission_static_note.start.apply(
                lambda x: max(row.start, x)
            ) <= submission_static_note.end.apply(lambda x: min(row.end, x))
        new_submission.append(submission_static_note[~filt_note])
    new_submission = pd.concat(new_submission, axis=0)
    new_submission = pd.concat([submission, new_submission], axis=0)
    return new_submission


def add_static_dict(submission: pd.DataFrame, most_common_concept_path: Path, notes: pd.DataFrame):
    with open(most_common_concept_path, "rb") as fp:
        most_common_concept = pickle.load(fp)
    submission_static = get_submission_from_static_dict(notes, most_common_concept)
    submission = merge_submissions(submission, submission_static)
    return submission
