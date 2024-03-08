import re

import pandas as pd


def postprocess_annotations(
    data_df, ann_df, att_df, submission=False, do_before=True, do_after=False
):
    ann_list = []
    max_words_before = 2
    max_words_after = 2
    max_chars = 20

    for id, txt in data_df.items():
        ann = ann_df[ann_df.note_id == id]

        for i in range(0, len(ann)):
            a = ann.iloc[i]
            a_txt = txt[a.start : a.end]
            next_a = None
            prev_a = None
            if i < len(ann) - 1:
                next_a = ann.iloc[i + 1]
            if i > 0:
                prev_a = ann.iloc[i - 1]
            found = False
            cand = att_df[att_df.generalId == a.concept_id]
            if len(cand) > 0:
                reg_str = r"(?:\b\S+\s*){{,{0}}}{1}\b(?:\s*\S+\s*?){{,{2}}}".format(
                    max_words_before, re.escape(a_txt), max_words_after
                )
                if prev_a is not None:
                    left_lim = max(prev_a.end + 1, a.start - max_chars)
                else:
                    left_lim = max(0, a.start - max_chars)
                if next_a is not None:
                    right_lim = min(next_a.start - 1, a.end + max_chars)
                else:
                    right_lim = min(a.end + max_chars, len(txt) - 1)

                words = re.findall(reg_str, txt[left_lim:right_lim])
                if len(words) > 0:
                    words = words[0].split(a_txt)
                    txt_before = words[0]
                    txt_after = words[1]
                    words_before = txt_before.split()
                    words_after = txt_after.split()

                    for i, c in cand.iterrows():
                        attribute = c.additionalWord
                        new_concept = c.specificId
                        new_start = a.start
                        new_end = a.end
                        side_words = [attribute, attribute.title(), attribute.upper()]
                        if attribute in ["left", "right"]:
                            side_words = side_words + [
                                attribute[0].upper(),
                                attribute + "-sided",
                            ]

                        if do_before:
                            ind_before = [
                                ind for (ind, w) in enumerate(words_before) if w in side_words
                            ]
                            if len(ind_before) > 0:
                                if ind_before[0] == len(words_before) - 1:
                                    delta = len(txt_before) - len(txt_before.rstrip())
                                    new_start = new_start - len(words_before[-1]) - delta
                                found = True
                                break

                        if do_after:
                            ind_after = [
                                ind for (ind, w) in enumerate(words_after) if w in side_words
                            ]
                            if len(ind_after) > 0:
                                if ind_after[0] == 0:
                                    delta = len(txt_after) - len(txt_after.lstrip())
                                    new_end = new_end + len(words_after[0]) + delta
                                found = True
                                break

            if found:
                # update annotation
                ann_list.append(
                    {
                        "note_id": id,
                        "start": new_start,
                        "end": new_end,
                        "concept_id": new_concept,
                    }
                )
            else:
                # keep original annotation
                ann_list.append(
                    {
                        "note_id": id,
                        "start": a.start,
                        "end": a.end,
                        "concept_id": a.concept_id,
                    }
                )

    post_ann_df = pd.DataFrame(ann_list)
    return post_ann_df


if __name__ == "__main__":
    dataset = "../../dev/mimic-iv_notes_training_set.csv"
    ann_file = "../../dev/predicted annotations/base_ann_vcv13.csv"
    out_ann_file = "./base_ann_vcv_post.csv"
    gt_ann_file = "../../dev/train_annotations_cln.csv"
    att_fn = "../../dev/term_extension.csv"

    ann_df = pd.read_csv(ann_file)
    data_df = pd.read_csv(dataset)
    data_df = data_df[data_df.note_id.isin(ann_df["note_id"])]

    att_df = pd.read_csv(att_fn)

    post_ann_df = postprocess_annotations(data_df, ann_df, att_df)
    post_ann_df.to_csv(out_ann_file, index=False)

    # run scoring on original and post-processed results
    gt_ann_df = pd.read_csv(gt_ann_file)
    gt_ann_df = gt_ann_df[gt_ann_df.note_id.isin(ann_df["note_id"])]

    exit(0)
