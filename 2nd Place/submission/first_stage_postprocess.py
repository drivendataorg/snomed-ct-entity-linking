import re
from collections import OrderedDict

import pandas as pd
from loguru import logger
from tqdm import tqdm


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


def add_spans(dfa, dfn):
    logger.info(f"Adding headers to annotations.. {dfa.shape}")
    notes_headers = calc_header_span(dfn)
    gg = dfa.groupby("note_id")
    res = []
    for note_id, group in tqdm(gg, desc="Adding spans.."):
        note_headers_spans = notes_headers[note_id]
        for i, row in group.iterrows():
            start, end = row.start, row.end
            for header, (hstart, hend) in note_headers_spans.items():
                if start >= hstart and start <= hend:
                    row["header"] = header
                    break
            res.append(row)
    dfa = pd.DataFrame(res)
    return dfa


def clean_by_header(dfa, dfn):
    cut_headers = [
        "medications on admission:",
        "___ on admission:",
        "discharge medications:",
    ]
    dfa = add_spans(dfa, dfn)
    dfa = dfa[~dfa.header.isin(cut_headers)]
    return dfa


def _clean_spans(dfa, dfn):
    logger.info(f"Pre-cleaned spans.. {dfa.shape}")
    spans_to_delete = calc_header_span(dfn)
    gg = dfa.groupby("note_id")
    res = []
    for note_id, group in tqdm(gg, desc="Cleaning spans.."):
        dspans = spans_to_delete[note_id]
        for i, row in group.iterrows():
            start, end = row.start, row.end
            for dstart, dend in dspans:
                if start >= dstart and start <= dend:
                    row.start = -1
                    break
                if end >= dstart and end <= dend:
                    row.end = -1
                    break
            res.append(row)
    dfa = pd.DataFrame(res)
    dfa = dfa[(dfa.start != -1) & (dfa.end != -1)]
    logger.info(f"Cleaned spans.. {dfa.shape}")
    return dfa
