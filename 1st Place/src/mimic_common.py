# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:10:31 2024

@author: Yonatan
"""

import re
from collections import Counter

import pandas as pd
from tqdm import tqdm

common_headers = [
    "Allergies",
    "History of Present Illness",
    "Family History",
    "Name",
    "Major Surgical or Invasive Procedure",
    "Admission Date",
    "Discharge Disposition",
    "Past Medical History",
    "Attending",
    "Service",
    "Date of Birth",
    "Discharge Instructions",
    "Discharge Condition",
    "Chief Complaint",
    "Physical Exam",
    "Pertinent Results",
    "Discharge Medications",
    "Social History",
    "Followup Instructions",
    "Medications on Admission",
    "Discharge Diagnosis",
]

internal_blacklist = [
    "other",
    "negative",
    "follow up",
    "mild",
    "normal",
    "inr",
    "changes",
    "change",
    "iv",
]

pattern_cache = {}


def get_pattern(s):
    if s in pattern_cache:
        return pattern_cache[s]

    p = " ".join(s.split())
    for c in "+(){}[]*":
        p = p.replace(c, f"\\{c}")
    p = p.replace(" ", "\\s+")
    p = p.replace("-", "[- ]")
    p = p.replace("/", "[/ ]")
    p = p + "s*"

    try:
        r = re.compile(p)
    except Exception:
        print(p)
        return None
    pattern_cache[s] = r
    return r


def is_in_header(text, pos):
    next_nl = text[pos:].index("\n") + pos
    if text[:next_nl].strip().endswith(":"):
        return True
    return False


def get_header_by_pos(pos, headers_position, pos_header, legal_headers):
    prev_pos = [p for p in headers_position if p <= pos]
    if len(prev_pos) == 0:
        return None
    header_pos = max(prev_pos)
    h = pos_header[header_pos]
    if h not in legal_headers and h[:-1] not in legal_headers:
        return "other"
    if h[-1] != ":":
        return h + ":"
    return h


def get_sections(text, headers):
    pos_header = {text.find(h + ":"): h + ":" for h in headers if h + ":" in text}
    add_break_lines(pos_header, text)
    positions = list(pos_header.keys())
    positions.sort()
    positions.append(len(text))
    return positions, pos_header


def add_break_lines(pos_header, text):
    prev_line_is_header = False
    lines = text.split("\n")
    prev_pos = None
    for i, l in enumerate(lines):
        l = l.strip()
        if len(l) == 0:
            continue
        pos = sum([len(line) + 1 for line in lines[:i]])
        assert text[pos : pos + len(lines[i])] == lines[i]
        if l.endswith(":") and i < len(lines) - 1 and len(lines[i + 1].strip()) == 0:
            pos_header[pos] = l
            prev_line_is_header = True
        elif (
            all([c in "-=_:" for c in l])
            and not prev_line_is_header
            and len(lines[i - 1].strip()) > 0
        ):
            pos_header[prev_pos] = lines[i - 1].strip()
        else:
            prev_line_is_header = False
        prev_pos = pos


def annotate_with_dict(text, d, headers, note_id, keep_overlaps=False):
    ann = pd.DataFrame(columns=["note_id", "start", "end", "concept_id", "section", "dict_entry"])
    h_positions, pos_header = get_sections(text, headers)
    for (section, source_text), cid in d.items():

        p = get_pattern(source_text)
        if p is None:
            continue
        for match in p.finditer(text):
            i = match.start()
            j = match.end()
            if i < 100:
                continue
            if text[i - 1].isalnum() or text[j].isalnum():
                continue
            if is_in_header(text, i) and not keep_overlaps:
                continue
            h = get_header_by_pos(i, h_positions, pos_header, headers)
            if h is None:
                return
            if "medication" in h.lower() or "service" in h.lower() or "date of birth" in h.lower():
                continue

            if h == section or h in section or section == "any":
                r = len(ann)
                if type(cid) == Counter:
                    for k in cid:
                        vals = [note_id, i, j, k, section, source_text]
                else:
                    vals = [note_id, i, j, cid, section, source_text]
                ann.loc[r] = vals

    if keep_overlaps:
        return ann
    return remove_overlaps(ann)


def shorter_span(i, j, l):
    if l.iloc[i] < l.iloc[j]:
        return i
    return j


def remove_overlaps(df, verbose=False):
    log = []
    df = df.sort_values("start").reset_index(drop=True)
    length = df["end"] - df["start"]
    length = length.astype(float)
    section_any = [type(s) == tuple or s == "any" for s in df["section"]]
    length[
        section_any
    ] -= 0.1  # if the section is "other" then we prefer the same span with a section-based annotation
    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            remove_index = shorter_span(i, j, length)
            if verbose:
                log.append(
                    f'overalpping segments {df.iloc[i]["start"]}-{df.iloc[i]["end"]} and {df.iloc[j]["start"]}-{df.iloc[j]["end"]}'
                )
            to_remove.add(df.index[remove_index])
            if remove_index == i:
                break

    df2 = df.drop(to_remove)
    for i in to_remove:
        s, e = df.loc[i, ["start", "end"]].values
        overlaps = ((df2["start"] <= s) & (df2["end"] > s)) | (
            (df2["start"] <= e) & (df2["end"] > e)
        )
        if overlaps.sum() == 0:
            if verbose:
                log.append(f'returning segment {df.loc[i]["start"]}-{df.loc[i]["end"]}')
            df2.loc[i] = df.loc[i]

    if verbose:
        return df2, log
    return df2


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
