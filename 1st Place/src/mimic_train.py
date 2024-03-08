# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 09:19:31 2024

@author: Yonatan
"""

import pickle
from collections import Counter
from itertools import permutations
from pathlib import Path

import pandas as pd
from mimic_common import (
    annotate_with_dict,
    common_headers,
    get_header_by_pos,
    get_sections,
    internal_blacklist,
)
from tqdm import tqdm

data_directory = Path(__file__).parent.parent / "data"
(debug_directory := Path(__file__).parent.parent / "debug").mkdir(exist_ok=True)
correct_frac_for_dict = 0.2
correct_frac_for_any = 0.3
yuvals_method_ratio = 1
snomed_min_len = 2
snomed_max_len = 5
blacklist_thresh = 2000
train_size = 150
test_size = None
words_counter = Counter()


def get_blacklist():
    if len(words_counter) == 0:
        print("word counter not initiated")
        return None
    bl = [v for v in words_counter if words_counter[v] > blacklist_thresh]
    bl.extend(internal_blacklist)
    return bl


def build_dict(text, text_annotations, headers, blacklist):
    d = {}
    length = text_annotations["end"] - text_annotations["start"]
    h_positions, pos_header = get_sections(text, headers)
    section_blacklist = {}
    for bl in blacklist:
        if type(bl) == tuple:
            section_blacklist.setdefault(bl[0], set()).add(bl[1])

    rows = (length > 1) & ~text_annotations["source"].isin(blacklist)
    for i in text_annotations.index[rows]:
        mention = text_annotations["source"][i]
        h = get_header_by_pos(text_annotations["start"][i], h_positions, pos_header, headers)
        if h in section_blacklist and mention in section_blacklist[h]:
            continue
        d.setdefault((h, mention), Counter())[text_annotations["concept_id"][i]] += 1
        d.setdefault(("any", mention), Counter())[text_annotations["concept_id"][i]] += 1

    return d


def score_dict(text, ref, d, headers):
    scores = {}
    ann = annotate_with_dict(text, d, headers, None, keep_overlaps=True)
    ann["start"] = ann["start"].astype(int)
    ann["end"] = ann["end"].astype(int)
    ann["concept_id"] = ann["concept_id"].astype(int)
    compare_ref_pred(ref.copy(), ann)
    scores_counter = Counter()
    for i in ann.index:
        h = ann["section"][i]
        source = ann["dict_entry"][i]
        k = (h, source)
        if k in d:
            if type(d[k]) == Counter:
                k = (k, ann["concept_id"][i][-1])
            scores.setdefault(k, []).append(ann["score"][i])
            scores_counter[(k, ann["score"][i])] += 1
        else:
            print("key not in dict:", k)

    return scores, scores_counter


def compare_ref_pred(ref, ann):
    ref.sort_values("start", inplace=True)
    ann.sort_values("start", inplace=True)
    i_ref = 0
    n_ref = len(ref)
    ann["score"] = None
    for i in ann.index:
        while i_ref + 1 < n_ref - 1 and ref.iloc[i_ref + 1]["start"] <= ann.loc[i, "start"]:
            i_ref = i_ref + 1

        score = overlap_score(
            ref.iloc[i_ref]["start"],
            ref.iloc[i_ref]["end"],
            ann.loc[i, "start"],
            ann.loc[i, "end"],
            ref.iloc[i_ref]["concept_id"],
            ann.loc[i, "concept_id"],
            ann.loc[i, "dict_entry"],
        )

        if score == 0 and i_ref + 1 < n_ref - 1:
            score = overlap_score(
                ref.iloc[i_ref + 1]["start"],
                ref.iloc[i_ref + 1]["end"],
                ann.loc[i, "start"],
                ann.loc[i, "end"],
                ref.iloc[i_ref + 1]["concept_id"],
                ann.loc[i, "concept_id"],
                ann.loc[i, "dict_entry"],
            )
        if score == 0:
            score = -1
        ann.loc[i, "score"] = score


def overlap_score(ref_start, ref_end, ann_start, ann_end, ref_concept, ann_concept, mention):
    if ref_start > ann_start:
        return -1
    elif ref_end < ann_start:
        return 0  # it might overlap the next one
    elif ref_concept == ann_concept:
        if (ref_start == ann_start and ref_end == ann_end) or " " in mention:
            return 1
        return -1
    else:
        return -1


def add_snomed_syn(d, c_id, c_name, min_len, max_len):
    n = len(c_name.split())
    if len(c_name) < 3:
        return
    if "machine translation" in c_name:
        return
    if "]" in c_name and c_name.index("[") > 5:
        return
    pt = process_term(c_name)
    n = len(pt.split())
    if n > max_len or n < min_len:
        return
    if not pt[0].isalnum():
        return
    if len(pt) > 1:
        d[("any", pt)] = c_id


def get_snomed_synonyms(min_len=snomed_min_len, max_len=snomed_max_len, fsn_only=False):
    snomed_syns = pd.read_csv(
        data_directory / "interim" / "flattened_terminology_syn_snomed+omop_v5.csv"
    ).drop_duplicates("concept_name", keep="first")

    sno_fsn = (
        pd.read_csv(data_directory / "interim" / "flattened_terminology.csv")
        .drop_duplicates("concept_name")
        .set_index("concept_id")["concept_name"]
    )
    replacements = {
        "procedure": "procedure",
        "body structure": "body structure",
        "disorder": "finding",
        "finding": "finding",
        "morphologic abnormality": "body structure",
        "cell structure": "body structure",
        "regime/therapy": "finding",
    }
    cid_to_type = sno_fsn.apply(lambda x: x.split("(")[-1][:-1]).replace(replacements)

    d = {}
    if not fsn_only:
        for c_id, c_name in snomed_syns[["concept_id", "concept_name"]].values:
            add_snomed_syn(d, c_id, c_name, min_len, max_len)
    for c_id in sno_fsn.index:
        add_snomed_syn(d, c_id, sno_fsn[c_id], min_len, max_len)

    sno_fsn = sno_fsn.apply(lambda t: process_term(t))

    return d, sno_fsn, cid_to_type


def process_term(t):
    t = t.lower()
    if "(" in t:
        t = t[: t.rindex("(") - 1]
    if "]" in t:
        t = t[t.index("]") + 1 :]

    return t.strip()


def get_permutations(d, blacklist):
    permuted = {}
    for k in d:
        words = k[1].split()
        new_mentions = []
        n = len(words)
        if n < 3 or n > 4:
            continue
        if n == 3 and words[1] == "of":
            new_mentions = [f"{words[2]} {words[0]}"]
        elif n == 4:
            if words[1] == "of":
                new_mentions = [f"{words[2]} {words[3]} {words[0]}"]
            elif words[2] == "of":
                new_mentions = [
                    f"{words[3]} {words[0]} {words[1]}",
                    f"{words[0]} {words[3]} {words[1]}",
                ]
        elif (n == 3 or n == 4) and all([w not in blacklist for w in words]):
            new_mentions = [" ".join(p) for p in permutations(words)]
        for new_mention in new_mentions:
            new_key = (k[0], new_mention)
            if new_key not in d:
                permuted[new_key] = d[k]
    return permuted


def get_word_replacements(d):
    wr = {}
    replacements = {
        ",": "",
        " and ": " with ",
        " with ": " and ",
        " valve ": " ",
        " of ": " of the ",
    }
    for k in d:
        mention = k[1]
        for s1, s2 in replacements.items():
            if s1 in mention:
                wr[(k[0], mention.replace(s1, s2))] = d[k]

    return wr


def remove_bad_keys(d, scores, scores_include_cid=False):
    bad_keys = []
    for k in scores:
        if scores_include_cid and (k[0] not in d or d[k[0]] != k[1]):
            continue

        if is_naive_key_remove(count_correct(scores[k]), k, scores_include_cid):
            if scores_include_cid:
                bad_keys.append(k[0])
            else:
                bad_keys.append(k)

    print(
        "number of bad keys:",
        len(bad_keys),
        "in d:",
        len(set(bad_keys).intersection(d.keys())),
    )
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


def yuvals_key_selection(d, scores_by_mention, scores_by_note, annotations):
    cids = annotations["concept_id"].unique()
    bad_keys = []
    print("removing bad keys")
    for cid in tqdm(cids):
        keys = [k for k in d if d[k] == cid]
        t_scores = {k: count_correct(scores_by_mention[k]) for k in keys if k in scores_by_mention}
        n_annotations = (annotations["concept_id"] == cid).sum()
        bad_keys.extend(get_bad_keys_for_concept(t_scores, n_annotations))

    print(
        "number of bad keys (yuval method):",
        len(bad_keys),
        "in d:",
        len(set(bad_keys).intersection(d.keys())),
    )
    for k in bad_keys:
        d.pop(k, None)
    return bad_keys


def count_correct(l):
    s = pd.Series(l)
    return (s == 1).sum(), (s == -1).sum()


def get_bad_keys_for_concept(scores, n):
    assert n > 0
    bad_keys = []
    key_to_ratio = pd.Series(
        [scores[k][0] / (scores[k][1] + 0.01) for k in scores], index=scores.keys()
    )
    correct = 0
    incorrect = 0
    key_to_ratio.sort_values(ascending=False, inplace=True)
    for i, k in enumerate(key_to_ratio.index):
        curr_score = correct / (incorrect + n)

        if curr_score < key_to_ratio[k] or not is_naive_key_remove(
            scores[k], k, double_thr=(i > 2)
        ):
            correct += scores[k][0]
            incorrect += scores[k][1]
        else:
            bad_keys.append(k)
    return bad_keys


def is_naive_key_remove(counts, k, scores_include_cid=False, double_thr=False):
    section = k[0] if not scores_include_cid else k[0][0]
    th = correct_frac_for_any if section == "any" else correct_frac_for_dict
    if double_thr:
        th = th * 2

    correct = counts[0]
    if correct == 1:
        th = 1
    incorrect = counts[1]
    return correct < th * incorrect


def mock_train(texts, annotations, headers, run_name):
    blacklist = get_blacklist()
    d, d_all = {}, {}
    print("extracting annotations")

    ids = texts.index

    d_combined = {}
    for i in tqdm(ids):
        t = build_dict(texts[i], annotations.query(f'note_id == "{i}"'), headers, blacklist)

        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])
        d_all[i] = t

    for k in d_combined:
        mc = d_combined[k].most_common(1)
        if len(mc) == 1:
            d[k] = mc[0][0]

    scores_by_note = {}
    scores_by_mention = {}
    scores_counter = {}
    print("scoring")
    for i in tqdm(ids):
        t, scores_counter[i] = score_dict(
            texts[i], annotations.query(f'note_id == "{i}"'), d, headers
        )
        for k in t:
            scores_by_mention.setdefault(k, []).extend(t[k])
            for s in [1, -1]:
                if s in t[k]:
                    scores_by_note.setdefault(k, []).append(s)

    d_full = d.copy()
    bad_keys = remove_bad_keys(d, scores_by_note)

    with (debug_directory / f"{run_name}.pkl").open("wb") as fp:
        pickle.dump(
            {
                "d_trained": d,
                "d_full": d_full,
                "d_all": d_all,
                "bad_keys": bad_keys,
                "d_combined": d_combined,
                "scores_by_note": scores_by_note,
                "scores_by_mention": scores_by_mention,
                "scores_counter": scores_counter,
            },
            fp,
        )

    return d, scores_by_note, scores_by_mention


def get_cid_type_sections_pairs(texts, annotations, headers, cid_to_type):
    pairs = set()
    for nid in annotations["note_id"].unique():
        if nid not in texts.index:
            continue
        df = annotations.query(f'note_id == "{nid}"')
        h_positions, pos_header = get_sections(texts[nid], headers)
        for i, cid in df[["start", "concept_id"]].values:
            h = get_header_by_pos(i, h_positions, pos_header, headers)
            pairs.add((h, cid_to_type[cid]))
    return pairs


def get_allowed_sections(texts, annotations, headers, cid_to_type):
    act = {}
    for section, ct in get_cid_type_sections_pairs(texts, annotations, headers, cid_to_type):
        act.setdefault(ct, set()).add(section)
    return act


def limit_any_to_allowed_sections(d, allowed_sec, cid_to_type):
    any_keys = [k for k in d if k[0] == "any"]
    for k in any_keys:
        cid = d[k]
        if cid not in cid_to_type.index:
            print(f"CID {cid} not in flat snomed, skipping")
            continue
        ct = cid_to_type[cid]
        d[(tuple(allowed_sec[ct]), k[1])] = cid
        d.pop(k, None)


def cond_update(d, d2, sno_fsn, blacklist):
    for k, v in d2.items():
        if k[1] in blacklist:
            continue
        if k not in d or k[1] == sno_fsn.loc[v].lower():
            d[k] = v


def extract_uppercase_mentions(d, annotations):
    uc_d = {}
    to_remove = []
    for k in d:
        section, mention = k
        mention_source = annotations.loc[annotations["source"] == mention, "source orig"]
        if (mention_source == mention_source.str.upper()).mean() > 0.99:
            uc_d[(capitalize_section(section), mention.upper())] = d[k]
            to_remove.append(k)
    for k in to_remove:
        d.pop(k, None)
    return uc_d


def capitalize_section(section):
    if section in ["other", "any"]:
        return section
    for h in common_headers:
        if section == h.lower() + ":":
            return h + ":"
    print("section not found", section)
    return "any"


def add_external_dicts(d, sno_syns, sno_fsn, blacklist):
    print("initial dict size", len(d))

    cond_update(d, sno_syns, sno_fsn, blacklist)
    print("after adding snomed", len(d))

    with open(
        data_directory / "interim" / "snomed_unigrams_annotation_dict_3k_v4_new.pkl", "rb"
    ) as fp:
        d_unigrams = pickle.load(fp)
    cond_update(d, d_unigrams, sno_fsn, blacklist)
    print("after adding snomed unigrams", len(d))

    with open(
        data_directory / "interim" / "snomed_unigrams_annotation_dict_20k_v4_fsn.pkl", "rb"
    ) as fp:
        d_unigrams = pickle.load(fp)
    cond_update(d, d_unigrams, sno_fsn, blacklist)
    print("after adding FSN snomed unigrams", len(d))

    wr = get_word_replacements(d)
    cond_update(d, wr, sno_fsn, blacklist)
    print("after doing word replacements", len(d))

    permuted = get_permutations(d, blacklist)
    cond_update(d, permuted, sno_fsn, blacklist)
    print("after adding permutations", len(d))


def train(texts, annotations, headers=common_headers, run_name="debug"):
    texts_lc = texts.str.lower()
    words = "\n".join(texts_lc).split()
    if len(words_counter) == 0:
        words_counter.update(Counter(words))

    headers = [h.lower() for h in headers]
    if "source orig" not in annotations:
        annotations["source orig"] = annotations["source"]
        annotations["source"] = annotations["source"].str.lower()

    sno_syns, sno_fsn, cid_to_type = get_snomed_synonyms()
    allowed_sec = get_allowed_sections(texts_lc, annotations, headers, cid_to_type)

    d, scores_by_note, scores_by_mention = mock_train(texts_lc, annotations, headers, run_name)
    uc_d = extract_uppercase_mentions(d, annotations)
    print("number of entries moved to uc dict", len(uc_d))

    add_external_dicts(d, sno_syns, sno_fsn, get_blacklist())

    limit_any_to_allowed_sections(d, allowed_sec, cid_to_type)

    with (debug_directory / f"{run_name}_full.pkl").open("wb") as fp:
        pickle.dump(d, fp)

    return d, uc_d
