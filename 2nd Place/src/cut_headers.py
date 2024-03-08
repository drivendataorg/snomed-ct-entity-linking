import re
from collections import OrderedDict

import pandas as pd
from tqdm.auto import tqdm

true_headers_list = [
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


cut_headers_list = [
    "medications on admission:",
    "___ on admission:",
    "discharge medications:",
]


def get_true_header_indices(text, true_headers_list):
    text = re.sub("\n", " ", text.lower())
    true_header_indices = {}
    for true_header in true_headers_list:
        pos = text.find(true_header)
        if pos != -1:
            true_header_indices[true_header] = pos
    true_header_indices = dict(sorted(true_header_indices.items(), key=lambda item: item[1]))
    true_header_indices = OrderedDict(true_header_indices)
    return true_header_indices


def cut_headers(df, ann_df):
    cut_notes = {}
    cut_ann = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row["text"]
        adf = ann_df[ann_df.note_id == row.note_id]
        adf = adf.copy()
        for cut_header in cut_headers_list:
            headers = get_true_header_indices(text.lower(), true_headers_list)
            if cut_header in headers:
                cut_start = headers[cut_header]
                i = list(headers).index(cut_header)
                next_header = list(headers)[i + 1]
                cut_end = headers[next_header]
                diff = cut_end - cut_start

                # move spans to the left
                for j, r in adf.iterrows():
                    if r.start >= cut_end:
                        adf.at[j, "start"] -= diff
                        adf.at[j, "end"] -= diff
                    elif r.start >= cut_start:
                        # delete
                        adf.drop(j, inplace=True)
                adf.reset_index(drop=True, inplace=True)

                text = text[:cut_start] + text[cut_end:]
        cut_notes[row.note_id] = text
        cut_ann.append(adf)
    cut_notes_df = pd.DataFrame(cut_notes.items(), columns=["note_id", "text"])
    cut_ann = pd.concat(cut_ann)
    return cut_notes_df, cut_ann
