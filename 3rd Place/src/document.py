import difflib
import math
import os
import re
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()
warnings.filterwarnings("ignore")

headings = [
    "Name:",
    "Unit No:",
    "Admission Date:",
    "Discharge Date:",
    "Date of Birth:",
    "Sex:",
    "Service:",
    "Allergies:",
    "Attending:",
    "Chief Complaint:",
    "History of Present Illness:",
    "Past Medical History:",
    "Social History:",
    "Family History:",
    "Physical Exam:",
    "Pathology:",
    "Brief Hospital Course:",
    "Medications on Admission:",
    "Discharge Medications:",
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Discharge:",
    "Pertinent Results:",
    "Studies:",
    "Pending Results:",
    "Transitional Issues:",
    "PAST SURGICAL HISTORY:",
    "ADMISSION PHYSICAL EXAM:",
    "DISCHARGE PHYSICAL EXAM:",
    "PERTINENT LABS:",
    "DISCHARGE LABS:",
    "MICROBIOLOGY:",
    "IMAGING:",
    "ACTIVE ISSUES:",
    "CHRONIC ISSUES:",
    "Review of Systems:",
    "Major Surgical or Invasive Procedure:",
    "ADMISSION CXR:",
    "FOLLOW UP CXR:",
    "VASCULAR SURGERY ADMISSION EXAM:",
    "ADMISSION LABS:",
    "DEATH EXAM:",
    "CXR:",
    "CXR ___:",
    "SECONDARY:",
    "LABS:",
]
toIgnore = []
toIgnore_2 = [
    "Name:",
    "Unit No:",
    "Admission Date:",
    "Discharge Date:",
    "Date of Birth:",
    "Sex:",
    "Service:",
    "Attending:",
    "Medications on Admission:",
    "Discharge Medications:",
    "Followup Instructions:",
]
is_submission = True


def load_notes(NOTES_PATH):
    # Check if the data root directory exists
    if not os.path.exists(NOTES_PATH):
        print("NOTES_PATH does not exist: " + str(NOTES_PATH))
        sys.exit()

    # Define the file paths for different versions of the data
    file_paths_notes = Path(NOTES_PATH)

    df_notes = pd.read_csv(file_paths_notes, sep=",")
    df_notes["text"] = df_notes.progress_apply(lambda x: x["text"].replace("<br>", " "), axis=1)

    return df_notes


def load_annotations(ANNOTATIONS_PATH):
    if not os.path.exists(ANNOTATIONS_PATH):
        print("ANNOTATIONS_PATH does not exist: " + str(ANNOTATIONS_PATH))
        sys.exit()
    file_paths_annotations = Path(ANNOTATIONS_PATH)
    df_annotations = pd.read_csv(file_paths_annotations, sep=",")

    return df_annotations


def add_annotations_to_notes(df_notes, df_annotations):
    df_notes["annotated_words"] = pd.Series([[]] * len(df_notes))

    for index, row in df_annotations.iterrows():
        note_id = row["note_id"]
        if len(df_notes.loc[df_notes["note_id"] == note_id, "text"].values) < 1:
            continue
        start = row["start"]
        end = row["end"]
        concept_id = row["concept_id"]
        text = df_notes.loc[df_notes["note_id"] == note_id, "text"].values[0]
        substring = text[start:end]
        df_notes.loc[df_notes["note_id"] == note_id, "annotated_words"] = df_notes.loc[
            df_notes["note_id"] == note_id, "annotated_words"
        ].apply(
            lambda x: x
            + [[note_id, str(start), str(end), str(concept_id), substring.replace("\n", "")]]
        )

    return df_notes


def minmax_text_tokens(df_notes, tokenizer):
    df_notes["tokens_document"] = df_notes.progress_apply(
        lambda x: get_num_tokens(x["text"], tokenizer=tokenizer), axis=1
    )
    print("min document tokens:", df_notes["tokens_document"].min())
    print("max document tokens:", df_notes["tokens_document"].max())


def minmax_chunck_tokens(df_notes, tokenizer):
    # df_notes['tokens_chunks'] = df_notes.progress_apply(lambda x: max(get_num_tokens(ch, tokenizer=tokenizer_mistral) for ch in x['text_chunks']), axis=1)
    df_notes["tokens_chunks_mini_sections"] = df_notes.progress_apply(
        lambda x: max(
            get_num_tokens(ch, tokenizer=tokenizer) for ch in x["text_chunks_mini_sections"]
        ),
        axis=1,
    )
    print("min sections chuncks tokens:", df_notes["tokens_chunks_mini_sections"].min())
    print("max sections chuncks tokens:", df_notes["tokens_chunks_mini_sections"].max())


def get_num_tokens(x, tokenizer):
    res = x
    res = len(tokenizer.tokenize(x))

    return res


def split_document_2Lines(document):
    return re.split(r"(?<=\n\n\w)|(?<=\n\s\w)", document)


def split_document_at_section(document):
    pattern = r"\n(?<=\w+.*:)"
    return re.split(pattern, document)


def split_document_phrases(document):
    # pattern = r'(?<=\.\n)'
    pattern = r"(?<=\n)(?=\n|\s{2,}|\_\_\_ |\[\])|(?<=[.!?)])(?=\s+[A-Z]|\s+\d+\.)|(?<=[a-zA-Z])(\s+\n)(?=[A-Z][a-z])|(?=\n\s*\w{5,}:\s+)|(?=\n\s*-{1,2}\s*\w+)"
    docs = re.split(pattern, document)

    final_docs = []
    buffer = ""
    for doc in docs:
        if not doc:
            continue
        if re.match(r"^\s*$", doc):
            if final_docs and len(final_docs) > 0:
                final_docs[-1] += doc
            else:
                buffer += doc
        else:
            if buffer:
                final_docs.append(buffer + doc)
                buffer = ""
            else:
                final_docs.append(doc)
    if buffer:
        final_docs.append(buffer)

    return final_docs


def find_longest_string(string_list, tokenizer_mistral):
    if not string_list:  # Check if the list is empty
        return None  # Return None if the list is empty
    longest_string = string_list[0]  # Initialize with the first string in the list
    for string in string_list:
        if get_num_tokens(string, tokenizer_mistral) > get_num_tokens(
            longest_string, tokenizer_mistral
        ):  # Compare lengths
            longest_string = string  # Update longest string
    return longest_string


def extract_sections(text):
    sections = sorted(headings, key=len, reverse=True)
    section_dict = {}
    # init dictionary and set start index
    for section in sections:  # init start index
        pattern = r"(^|\s\s+)" + re.escape(section)
        match = re.search(pattern, text)
        if match:
            section_dict[section] = [match.start(), None, ""]  # [start index, end index, text]
    # search for end index where a new section starts
    for section in sections:
        if section not in section_dict:
            continue
        next_section = None
        for key, value in section_dict.items():
            if section_dict[section][0] < value[0] and (
                next_section == None or value[0] < next_section[0]
            ):
                next_section = value
        if next_section != None:
            section_dict[section][1] = next_section[0]
        else:
            section_dict[section][1] = len(text)
        section_dict[section][2] = text[section_dict[section][0] : section_dict[section][1]]

    for ignore in toIgnore:
        if ignore in section_dict:
            del section_dict[ignore]

    for key, value in section_dict.items():
        section_dict[key][0] = str(section_dict[key][0])
        section_dict[key][1] = str(section_dict[key][1])

    section_dict = dict(sorted(section_dict.items(), key=lambda x: int(x[1][0])))

    return section_dict


def extract_sections_2(text):
    sections = sorted(headings, key=len, reverse=True)
    section_dict = {}
    # init dictionary and set start index
    for section in sections:  # init start index
        pattern = r"(^|\s\s+)" + re.escape(section)
        match = re.search(pattern, text)
        if match:
            section_dict[section] = [match.start(), None, ""]  # [start index, end index, text]
    # search for end index where a new section starts
    for section in sections:
        if section not in section_dict:
            continue
        next_section = None
        for key, value in section_dict.items():
            if section_dict[section][0] < value[0] and (
                next_section == None or value[0] < next_section[0]
            ):
                next_section = value
        if next_section != None:
            section_dict[section][1] = next_section[0]
        else:
            section_dict[section][1] = len(text)
        section_dict[section][2] = text[section_dict[section][0] : section_dict[section][1]]

    for ignore in toIgnore_2:
        if ignore in section_dict:
            del section_dict[ignore]

    for key, value in section_dict.items():
        section_dict[key][0] = str(section_dict[key][0])
        section_dict[key][1] = str(section_dict[key][1])

    section_dict = dict(sorted(section_dict.items(), key=lambda x: int(x[1][0])))

    return section_dict


def split_documents(df_notes, tokenizer, max_tokens):
    df_notes["dict_sections"] = df_notes.progress_apply(
        lambda x: extract_sections(x["text"]), axis=1
    )
    df_notes["text_chunks_mini_sections"] = pd.Series([[]] * len(df_notes))
    df_notes["text_chunks_mini_sections_indices"] = pd.Series([[]] * len(df_notes))
    for index, row in df_notes.iterrows():
        # SPLIT SECTIONS IN MINI CHUNKS WITH MAX TOKENS
        new_text_chunks = []
        new_text_chunks_indices = []
        for section, value_array in sorted(
            list(row["dict_sections"].items()), key=lambda x: int(x[1][0])
        ):
            section_chunks = split_chunks_in_mini_chunks(
                value_array[2], max_tokens, tokenizer, merge=""
            )
            section_chunks = [s for s in section_chunks if len(s) > 0]
            real_section_text = row["text"][int(value_array[0]) : int(value_array[1])]
            for ch in section_chunks:
                local_indices = real_section_text.index(ch)
                index_mini = local_indices + int(value_array[0])
                real_section_text = replace_with_asterisks(
                    real_section_text, local_indices, local_indices + len(ch)
                )
                new_text_chunks_indices.append(
                    [str(index_mini), str(index_mini + len(ch)), section]
                )
            new_text_chunks = new_text_chunks + section_chunks
        df_notes.at[index, "text_chunks_mini_sections"] = new_text_chunks
        df_notes.at[index, "text_chunks_mini_sections_indices"] = new_text_chunks_indices
    return df_notes


def split_documents_2(df_notes, tokenizer, max_tokens):
    df_notes["dict_sections"] = df_notes.progress_apply(
        lambda x: extract_sections_2(x["text"]), axis=1
    )
    df_notes["text_chunks_mini_sections"] = pd.Series([[]] * len(df_notes))
    df_notes["text_chunks_mini_sections_indices"] = pd.Series([[]] * len(df_notes))
    for index, row in df_notes.iterrows():
        # SPLIT SECTIONS IN MINI CHUNKS WITH MAX TOKENS
        new_text_chunks = []
        new_text_chunks_indices = []
        for section, value_array in sorted(
            list(row["dict_sections"].items()), key=lambda x: int(x[1][0])
        ):
            section_chunks = split_chunks_in_mini_chunks(
                value_array[2], max_tokens, tokenizer, merge=""
            )
            section_chunks = [s for s in section_chunks if len(s) > 0]
            real_section_text = row["text"][int(value_array[0]) : int(value_array[1])]
            for ch in section_chunks:
                local_indices = real_section_text.index(ch)
                index_mini = local_indices + int(value_array[0])
                real_section_text = replace_with_asterisks(
                    real_section_text, local_indices, local_indices + len(ch)
                )
                new_text_chunks_indices.append(
                    [str(index_mini), str(index_mini + len(ch)), section]
                )
            to_add_mini_chunks = [
                s_ch.replace("\n", " ").replace("  ", " ").strip() for s_ch in section_chunks
            ]
            new_text_chunks = new_text_chunks + section_chunks
        df_notes.at[index, "text_chunks_mini_sections"] = new_text_chunks
        df_notes.at[index, "text_chunks_mini_sections_indices"] = new_text_chunks_indices
    return df_notes


def replace_with_asterisks(original, start, end):
    # Sort the indices list to ensure we replace from the beginning to the end
    result = original
    offset = 0
    # Adjust the start and end based on the current offset
    adjusted_start = start + offset
    adjusted_end = end + offset
    # Replace the specified segment with asterisks
    result = result[:adjusted_start] + "*" * (adjusted_end - adjusted_start) + result[adjusted_end:]
    # Update the offset based on the difference in length caused by the replacement
    offset += (adjusted_end - adjusted_start) - (adjusted_end - adjusted_start)
    return result


def split_chunks_in_mini_chunks(chunk, max_tokens, tokenizer_mistral, merge=""):
    curr_chunck_tokens = get_num_tokens(chunk, tokenizer_mistral)
    if curr_chunck_tokens > max_tokens:
        # check in how many parts we need to split the chunk
        max_len_parts = max_tokens / math.ceil(curr_chunck_tokens / max_tokens)
        new_mini_chunks = split_document_phrases(chunk)

        new_chunks_to_append, _ = merge_chuncks_target_max_tokens(
            new_mini_chunks, None, max_len_parts, tokenizer_mistral, merge
        )

        return new_chunks_to_append

    else:
        return [chunk]


def merge_chuncks(df_notes, tokenizer, max_tokens, merge=""):
    for index, row in df_notes.iterrows():
        new_text_chunks, new_text_chunks_inndices = merge_chuncks_target_max_tokens(
            row["text_chunks_mini_sections"],
            row["text_chunks_mini_sections_indices"],
            max_tokens,
            tokenizer,
            "",
        )
        df_notes.at[index, "text_chunks_mini_sections"] = new_text_chunks
        df_notes.at[index, "text_chunks_mini_sections_indices"] = new_text_chunks_inndices

    return df_notes


def merge_chuncks_target_max_tokens(chunks, chunks_indices, max_tokens, tokenizer, merge=""):
    new_text_chunks = []
    new_text_chunks_indices = []
    new_chunck = """"""
    new_chunck_indices = []
    new_chunck_tokens = 0
    for i_ch, chunk in enumerate(chunks):
        curr_chunck_tokens = get_num_tokens(chunk, tokenizer)
        if new_chunck_tokens + curr_chunck_tokens > max_tokens:
            new_text_chunks.append(new_chunck)
            if chunks_indices != None:
                new_text_chunks_indices.append(
                    [
                        new_chunck_indices[0][0],
                        new_chunck_indices[len(new_chunck_indices) - 1][1],
                        new_chunck_indices[0][2],
                    ]
                )
            new_chunck = chunk
            if chunks_indices != None:
                new_chunck_indices = [chunks_indices[i_ch]]
            new_chunck_tokens = curr_chunck_tokens
        else:
            new_chunck = new_chunck + merge + chunk
            if chunks_indices != None:
                new_chunck_indices.append(chunks_indices[i_ch])
            new_chunck_tokens = new_chunck_tokens + curr_chunck_tokens

    if new_chunck != "":
        new_text_chunks.append(new_chunck)
        if chunks_indices != None:
            new_text_chunks_indices.append(
                [
                    new_chunck_indices[0][0],
                    new_chunck_indices[len(new_chunck_indices) - 1][1],
                    new_chunck_indices[0][2],
                ]
            )

    if chunks_indices != None:
        return new_text_chunks, new_text_chunks_indices
    else:
        return new_text_chunks, None


def extract_result_csv(text):
    # first tentative
    pattern = r"```csv(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return split_result_csv(match.group(1).strip())

    # second tentative
    pattern = r"```csv(.*?)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return split_result_csv(match.group(1).strip())

    # third tentative
    pattern = r".*\n(.*?)$"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return split_result_csv(match.group(1).strip())

    return text


def split_result_csv(text):
    return [s.replace(",", "").strip() for s in text.split("\n")]


def extract_result_simple(text):
    text_clean = text.strip()
    if text_clean.lower().startswith("none"):
        return []
    return [clean_sections_from_inferences(s.strip()) for s in text_clean.split(",")]


def get_annotations_from_inference_new(note_id, text, dict_sections, chunk_index, target):
    """Runs entity recognition on a note chunk.

    Returns:
        A tuple with two items, extracted terms and items to check.

        Extracted terms is a list of lists where the outer list has an entry per extracted term and
        the inner list has 11 entries corresponding to:

            0: note_id
            1: start index of the term in terms of the text
            2: end index of the term in terms of the text
            3: concept_id
            4: text of the term, i.e., text[start_index:end_index] (with newlines removed)
            5: context (added by src.snomedctentitylinking.add_context at a later step)
            6: top_concept_id
            7: top_concept_score
            8: top_concept_doc
            9: section
            10: term format extraction

        The second item returned is a list of chunks to check.
    """
    target_no_annotations = target.replace("<t>", "").replace("</t>", "")
    chunk_text = text[int(chunk_index[0]) : int(chunk_index[1])]

    base_indices = find_annotation_indices(target)
    mappings = find_matching_sequences_difflib(chunk_text, target_no_annotations)

    if (
        not is_submission
        and len(chunk_text) - sum([int(m[0][1]) - int(m[0][0]) for m in mappings]) > 0
    ):
        print(
            f"Text length: {len(chunk_text)},({chunk_index[0]}, {chunk_index[1]})  Missing parts in chunk: {len(chunk_text) - sum([int(m[0][1]) - int(m[0][0]) for m in mappings])}"
        )

    chunks_to_check = []
    gold_indices = []
    for indices in base_indices:
        start_in_mapping = None
        prev_mapping = None
        for mappinng in mappings:
            gold, inf, _ = mappinng
            if inf[0] <= indices[0] and indices[1] <= inf[1]:
                index_in_gold_start = int(chunk_index[0]) + indices[0] - inf[0] + gold[0]
                index_in_gold_end = int(chunk_index[0]) + indices[1] - inf[1] + gold[1]
                gold_indices.append(
                    [
                        note_id,
                        str(index_in_gold_start),
                        str(index_in_gold_end),
                        None,
                        text[index_in_gold_start:index_in_gold_end].replace("\n", ""),
                        "",
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                )
                break
            if inf[0] <= indices[0] < inf[1] and indices[1] > inf[1]:
                index_in_gold_start = int(chunk_index[0]) + indices[0] - inf[0] + gold[0]
                start_in_mapping = index_in_gold_start
                prev_mapping = mappinng
                continue
            if start_in_mapping != None and inf[0] <= indices[1] <= inf[1]:
                index_in_gold_end = int(chunk_index[0]) + indices[1] - inf[1] + gold[1]
                chunks_to_check.append([start_in_mapping, index_in_gold_end, "0"])
                start_in_mapping = None
                break
            if start_in_mapping and not inf[0] <= indices[1] <= inf[1]:
                start_in_mapping = None
                break

            if inf[0] > indices[1]:
                break

    min_len_to_check = 10
    # check sequences that may need an inference by it self
    last_mapping_end_index = None
    for mappinng in mappings:
        start_mapping = mappinng[0][0] + int(chunk_index[0])
        end_mapping = mappinng[0][1] + int(chunk_index[0])

        if last_mapping_end_index == None:
            if mappinng[0][0] > min_len_to_check:
                if len(gold_indices) > 0:
                    if start_mapping < int(gold_indices[0][1]):
                        chunks_to_check.append([int(chunk_index[0]), int(gold_indices[0][1]), "1"])
                else:
                    chunks_to_check.append([int(chunk_index[0]), start_mapping, "2"])
            last_mapping_end_index = end_mapping
        elif start_mapping - last_mapping_end_index > min_len_to_check:
            # check if there is a previous annotation to use the end as start, and if there is a next annotation to use the start as end
            prev_gi_end = None
            next_gi_start = None
            for gi in gold_indices:
                if int(gi[2]) <= start_mapping:
                    prev_gi_end = int(gi[2])
                    continue
                if int(gi[1]) >= last_mapping_end_index:
                    next_gi_start = int(gi[1])
                    break
            if prev_gi_end == None:
                prev_gi_end = last_mapping_end_index
            if next_gi_start == None:
                next_gi_start = start_mapping
            chunks_to_check.append([prev_gi_end, next_gi_start, "3"])
    if (
        len(mappings) > 1
        and (int(chunk_index[1]) - int(chunk_index[0])) - mappings[-1][0][1] > min_len_to_check
    ):
        if len(gold_indices) > 0 and int(gold_indices[-1][2]) < int(chunk_index[1]):
            chunks_to_check.append([int(gold_indices[-1][2]), int(chunk_index[1]), "4"])
        else:
            chunks_to_check.append(
                [int(chunk_index[0]) + mappings[-1][0][1], int(chunk_index[1]), "5"]
            )

    for i in gold_indices:
        for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
            if int(i[1]) >= int(value_array[0]) and int(i[2]) <= int(value_array[1]):
                if section != None:
                    i[9] = section
                else:
                    i[9] = "Somewhere"

    filtered_indices_sorted = remove_dupplicates(gold_indices)

    return filtered_indices_sorted, chunks_to_check


def remove_dupplicates(indices):
    indices_sorted = sorted(indices, key=lambda x: [int(x[1]), int(x[2])])
    filtered_indices_sorted = []
    current_end = -1
    removed_overlaps = []
    for interval in indices_sorted:
        start = int(interval[1])
        end = int(interval[2])

        # Check if the current interval overlaps with the previous one
        if start >= current_end:
            # No overlap, add the interval to the filtered list
            filtered_indices_sorted.append(interval)
            current_end = end
        else:
            # Overlap detected, but we only add the first (longest) of overlapping intervals due to sorting
            if (
                int(filtered_indices_sorted[-1][2]) - int(filtered_indices_sorted[-1][1])
                < end - start
            ):
                removed_overlaps.append(filtered_indices_sorted[-1])
                filtered_indices_sorted[-1] = interval
                current_end = end
    return filtered_indices_sorted


def find_matching_sequences_difflib(text1, text2):
    """
    This function finds all matching sequences between two given texts using difflib.SequenceMatcher.
    If the texts are the same, it will return one match containing the entire text.

    Args:
    text1 (str): The first text.
    text2 (str): The second text, typically generated by an LLM to annotate the first text.

    Returns:
    list of tuples: Each tuple contains the start and end indices of the matching sequence in text1,
                    and the actual matching text.
    """
    # Initialize the SequenceMatcher
    sm = difflib.SequenceMatcher(None, text1, text2)

    # Get matching blocks
    matching_blocks = sm.get_matching_blocks()

    # Initialize the list to store matching sequences pairs
    matches = []

    # Iterate over matching blocks and store the details
    for match in matching_blocks[:-1]:  # Exclude the last dummy match
        start1, start2, length = match
        if length > 0:  # Ensure that the match length is non-zero
            end1 = start1 + length
            end2 = start2 + length
            matches.append(((start1, end1), (start2, end2), text1[start1:end1]))

    return matches


def get_annotations_from_inference(note_id, text, dict_sections, chunk_index, target):
    target_no_annotations = target.replace("<t>", "").replace("</t>", "")
    # align start between text and target
    idx_remove, newText = match_and_copy(
        text[int(chunk_index[0]) : int(chunk_index[1])], target_no_annotations
    )
    new_target = target
    if idx_remove != -1:
        new_target = newText + target[idx_remove:]
    # allign end between text and target
    text_idx_mismatch, targer_idx_mismatch = find_mismatch_index(
        text[int(chunk_index[0]) : int(chunk_index[1])], new_target
    )
    targer_idx_mismatch_backup = len(new_target) - targer_idx_mismatch
    new_target = new_target[:targer_idx_mismatch]
    # target aligned with no tags
    target_no_annotations = new_target.replace("<t>", "").replace("</t>", "")

    print(
        "Start added from text: ",
        len(newText),
        "End mismatch in text: ",
        int(chunk_index[1]) - int(chunk_index[0]) - text_idx_mismatch,
    )
    print("Start removed: ", idx_remove, "End removed: ", targer_idx_mismatch_backup)

    base_indices = find_annotation_indices(new_target)
    final_indices = []
    for s, e in base_indices:
        full_text_start_index = int(chunk_index[0]) + int(s)
        full_text_end_index = int(chunk_index[0]) + int(e)
        annotation_section = None
        for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
            if int(full_text_start_index) >= int(value_array[0]) and int(
                full_text_end_index
            ) <= int(value_array[1]):
                annotation_section = section
                break
        final_indices.append(
            [
                note_id,
                str(full_text_start_index),
                str(full_text_end_index),
                None,
                text[full_text_start_index:full_text_end_index].replace("\n", ""),
                "",
                None,
                None,
                None,
                annotation_section,
                None,
            ]
        )

    indices_sorted = sorted(final_indices, key=lambda x: int(x[1]))

    filtered_indices_sorted = []
    current_end = -1
    removed_overlaps = []
    for interval in indices_sorted:
        start = int(interval[1])
        end = int(interval[2])

        # Check if the current interval overlaps with the previous one
        if start >= current_end:
            # No overlap, add the interval to the filtered list
            filtered_indices_sorted.append(interval)
            current_end = end
        else:
            # Overlap detected, but we only add the first (longest) of overlapping intervals due to sorting
            if (
                int(filtered_indices_sorted[-1][2]) - int(filtered_indices_sorted[-1][1])
                < end - start
            ):
                removed_overlaps.append(filtered_indices_sorted[-1])
                filtered_indices_sorted[-1] = interval
                current_end = end
    if len(removed_overlaps) > 0 and not is_submission:
        print("Overlaps removed:", removed_overlaps)
        print("Current indices:", filtered_indices_sorted)

    return filtered_indices_sorted


def find_full_word_occurrences(substring, text, note_id, start_index):
    # Split the substring into words and escape each word individually
    words = [re.escape(word) for word in substring.split()]
    # Join the escaped words with a pattern that matches any whitespace, including newlines
    pattern = r"\s+".join(words)
    # Use lookahead and lookbehind to ensure the match is surrounded by non-word characters or start/end of string
    pattern = r"(?<!\w)" + pattern + r"(?!\w)"
    matches = list(re.finditer(pattern, text, re.DOTALL))
    # Extract the start and end indices of each match
    indices = [
        [
            note_id,
            str(start_index + match.start()),
            str(start_index + match.end()),
            None,
            substring,
            "",
            None,
            None,
        ]
        for match in matches
    ]
    return indices


def map_strings_in_text(note_id, text, target):
    # Example usage
    # result_text = "The quick brown fox jumps over the lazy dog. The fox was very quick.".lower()
    # target = ["quick", "fox", "The", "very", "he"]
    # result = Document.map_strings_in_text(result_text, target)

    text_processed = text.lower()
    target_processed = [x.lower() for x in target if x != ""]
    # Sort the target list by length of strings, longest first
    target_sorted = sorted(target_processed, key=len, reverse=True)

    # Dictionary to store the indices of each string
    string_indices = {}

    for string in target_sorted:
        start = 0
        indices = []

        while True:
            # Find the string in the text
            index = text_processed.find(string, start)

            if index == -1:
                # No more occurrences found
                break

            # Add the index to the list and update the start position
            indices.append(index)
            start = index + 1

        # Record the indices for this string
        string_indices[string] = indices

        # Replace occurrences of this string with a placeholder
        # to prevent overlapping with shorter strings
        text_processed = text_processed.replace(string, "*" * len(string))

    return create_vector_from_index(note_id, string_indices)


def create_vector_from_index(note_id, dict):
    tuples = []
    for key, value in dict.items():
        for v in value:
            tuples.append([note_id, str(v), str(v + len(key)), None, key, ""])

    sorted_tuples = sorted(tuples, key=lambda x: int(x[1]))

    return sorted_tuples


def extract_context_for_terms_NEW(curr_item, dict_sections):
    curr_section = None
    for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
        if int(curr_item[1]) >= int(value_array[0]) and int(curr_item[2]) <= int(value_array[1]):
            curr_section = section
            break

    if curr_section == None:
        return curr_item[4]
    elif curr_section == "Allergies:":
        return "Allergy to " + curr_item[4] + " finding"
    elif curr_section == "Chief Complaint:":
        return "Disorder of " + curr_item[4]
    elif curr_section == "Major Surgical or Invasive Procedure:":
        return curr_item[4] + " procedure"
    else:
        text_to_use = dict_sections[curr_section][2]
        phrase_start, phrase_end = find_phrase_indices_with_regex(
            text_to_use,
            int(curr_item[1]) - int(dict_sections[curr_section][0]),
            int(curr_item[2]) - int(dict_sections[curr_section][0]),
        )
        phrase = text_to_use[phrase_start:phrase_end].replace("\n", "").strip()
        print(curr_item[4] + " in context of " + phrase)
        return curr_item[4] + " in context of " + phrase


def extract_context_for_terms(text, curr_item, all_terms, dict_sections):
    """Adds context to the the extracted terms for later entity linking/classification stage."""
    curr_section = None
    for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
        if int(curr_item[1]) >= int(value_array[0]) and int(curr_item[2]) <= int(value_array[1]):
            curr_section = section
            break
    if curr_section == None:
        return curr_item[4]
    elif curr_section == "Allergies:":
        return "Allergy to " + curr_item[4] + " finding"
    elif curr_section == "Chief Complaint:":
        return "Disorder of " + curr_item[4]
    elif curr_section == "Major Surgical or Invasive Procedure:":
        return curr_item[4] + " procedure"
    else:
        return curr_item[4]
        terms_to_use = list(
            filter(
                lambda x: x[1] != curr_item[1]
                and int(x[1]) > int(dict_sections[curr_section][0])
                and int(x[2]) < int(dict_sections[curr_section][1]),
                all_terms,
            )
        )
        text_to_use = text
        for i in terms_to_use:
            text_to_use = (
                text_to_use[: int(i[1])]
                + (" " * (int(i[2]) - int(i[1])))
                + text_to_use[int(i[2]) :]
            )
        text_to_use = text_to_use[
            int(dict_sections[curr_section][0]) : int(dict_sections[curr_section][1])
        ]

        phrase_start, phrase_end = find_phrase_indices_with_regex(
            text_to_use, int(curr_item[1]) - int(dict_sections[curr_section][0])
        )
        phrase = (
            re.sub(r"\s+", " ", text_to_use[phrase_start:phrase_end])
            .replace(curr_section, "")
            .strip()
        )

        return phrase


def find_phrase_indices_with_regex(text, index_start, index_end):
    # Define the regex pattern for splitting the text into phrases
    pattern = r"[.!?]\s+[A-Z]|\n[A-Z]]"

    # Find all matches of the pattern in the text and get their indices
    matches = list(re.finditer(pattern, text))
    phrases = []
    phrases_text = []
    start = 0

    # Split the text into phrases based on the matches
    for match in matches:
        end = match.start() + 1  # Include the punctuation in the phrase
        phrases.append((start, end))
        phrases_text.append(text[start:end])
        start = match.end() - 1  # Start after the space following the punctuation
    for s in phrases_text:
        print(s.replace("\n", ""))
    exit()
    # Add the last phrase if it exists
    if start < len(text):
        phrases.append((start, len(text)))

    start_found = None
    # Find the phrase that contains the given index
    for i, (phrase_start, phrase_end) in enumerate(phrases):
        if phrase_start <= index_start < phrase_end:
            start_found = phrase_start
            if index_end <= phrase_end:
                return (
                    phrase_start,
                    phrase_end,
                )  # Return the index of the phrase and its start and end indices
        if start_found != None and index_end <= phrase_end:
            return (
                phrase_start,
                phrase_end,
            )  # Return the index of the phrase and its start and end indices

    return None, None  # If the index is not in any phrase


def notes_to_submision(df_notes):
    spans = []
    headers = ["note_id", "start", "end", "concept_id"]
    for index_o, row in df_notes.iterrows():
        for term in row["result_chunks_inst"]:
            spans.append([row["note_id"], int(term[1]), int(term[2]), int(term[3])])
    spans_df = pd.DataFrame(columns=headers, data=spans)
    return spans_df


def notes_to_submision_extended(df_notes):
    spans = []
    headers = ["note_id", "start", "end", "concept_id", "top_concept_id"]
    for index_o, row in df_notes.iterrows():
        for term in row["result_chunks_inst"]:
            term = list(term)
            spans.append([row["note_id"], int(term[1]), int(term[2]), int(term[3]), term[6]])
    spans_df = pd.DataFrame(columns=headers, data=spans)
    return spans_df


def clean_sections_from_inferences(chunk):
    sections = sorted(headings, key=len, reverse=True)

    for section in sections:
        # Check if the chunk starts with the section
        if chunk.startswith(section):
            # Remove the section from the start of the chunk
            return chunk[len(section) :].strip()

    return chunk


def annotatate_notes(note_chunk, start_idx, annotations):
    # Sort the index pairs in reverse based on start indices to avoid index shift issues
    sorted_annotations_inv = sorted(annotations, key=lambda x: int(x[1]), reverse=True)

    # Iterate over each pair and insert the annotation tags
    for vect in sorted_annotations_inv:
        start = int(vect[1]) - int(start_idx)
        end = int(vect[2]) - int(start_idx)
        note_chunk = note_chunk[:end] + "</t>" + note_chunk[end:]
        note_chunk = note_chunk[:start] + "<t>" + note_chunk[start:]
    return note_chunk


def get_indices_from_annotated_text(annotated_text):
    start_tag = "<t>"
    end_tag = "</t>"
    pairs = []
    current_pos = 0

    while True:
        start_index = annotated_text.find(start_tag, current_pos)
        if start_index == -1:
            break  # No more start tags found

        end_index = annotated_text.find(end_tag, start_index)

        # Adjust indices to account for the text without tags and add the pair
        pairs.append((start_index, end_index - len(start_tag)))

        # Update current_pos to search for the next pair
        current_pos = end_index + len(end_tag)

    # Return the list of index pairs without tags
    return pairs


def find_annotation_indices(
    annotated_text, start_tag="<t>", end_tag="</t>", include_tags=False
) -> list[tuple[int, int]]:
    """Gets the start and stop character indices for annotated portions of text."""
    pairs = []
    current_pos = 0
    tag_adjustment = 0  # Adjusts for the length of tags already processed

    while True:
        start_index = annotated_text.find(start_tag, current_pos)
        if start_index == -1:
            break  # No more start tags found

        # Adjust start index to not include previous tags
        if include_tags:
            start_index_adjusted = start_index
        else:
            start_index_adjusted = start_index - tag_adjustment

        # Move past the start tag to look for the end tag
        if include_tags:
            current_pos = start_index
        else:
            current_pos = start_index + len(start_tag)

        end_index = annotated_text.find(end_tag, current_pos)
        if end_index == -1:
            break  # Corresponding end tag not found (shouldn't happen if tags are well-formed)

        # Adjust end index to not include the start tag and any previous tags
        if include_tags:
            end_index_adjusted = end_index - len(start_tag)
        else:
            end_index_adjusted = end_index - tag_adjustment - len(start_tag)

        # Update tag_adjustment to account for this set of tags
        tag_adjustment += len(start_tag) + len(end_tag)

        # Update current_pos to search for the next pair
        current_pos = end_index + len(end_tag)

        # Add the adjusted pair
        pairs.append((start_index_adjusted, end_index_adjusted))

    return pairs


def match_and_copy(s1, s2):
    """
    This function finds the first common substring of length 10 between two strings (s1 and s2).
    It then copies what comes before this common substring in s1 to the beginning of s2, making sure
    that both strings start with the same sequence up to the found common substring.

    If no common substring of length 10 is found, the function returns the original strings.
    """

    # Helper function to generate all substrings of a specific length
    def generate_substrings(s, length):
        return [s[i : i + length] for i in range(len(s) - length + 1)]

    # Generate all substrings of length 10 for both strings
    substrings_s1 = generate_substrings(s1, 10)
    substrings_s2 = generate_substrings(s2, 10)

    # Find the first common substring of length 10
    common_substring = None
    for sub in substrings_s1:
        if sub in substrings_s2:
            common_substring = sub
            break

    # If a common substring is found, modify s2
    if common_substring is not None:
        index_in_s1 = s1.index(common_substring)
        index_in_s2 = s2.index(common_substring)

        # Copy everything before the common substring in s1 to s2
        modified_s2 = s1[:index_in_s1] + s2[index_in_s2:]

        return index_in_s2, s1[:index_in_s1]
    else:
        # Return original strings if no common substring is found
        return -1, None


def find_mismatch_index(s1: str, s2: str) -> int:
    i, j = 0, 0  # i for s1, j for s2
    while i < len(s1) and j < len(s2):
        while s2[j : j + 3] == "<t>":
            j += 3  # Skip the content within the tags
        while s2[j : j + 4] == "</t>":
            j += 4  # Skip the content within the tags
        if j >= len(s2):
            break
        if s1[i] == s2[j]:
            i += 1
            j += 1
        else:
            return i, j
    return (
        i,
        j,
    )  # Return i as the index where s1 is no more equal to s2 from the start and j as the index where s2 is no more equal to s1 from the start


def find_full_word_occurrences_test(
    substring, text, note_id, sections_indices, chunk, terms_to_remove, extract_with_context=False
):
    # \b\w*[a-zA-Z]\w*\b  [^a-zA-Z]

    pattern, left, term, right = compose_pattern_from_term(substring, extract_with_context)
    if term == None:
        return []

    # empty strings remove
    if remove_annotations_if(term, substring, terms_to_remove):
        return []

    matches = list(re.finditer(pattern, text, re.DOTALL))
    if len(matches) == 0 and extract_with_context:
        if left != None:
            pattern, _, _, _ = compose_pattern_from_term(left + "{" + term + "}")
            matches = list(re.finditer(pattern, text, re.DOTALL))
        if len(matches) == 0:
            if right != None:
                pattern, _, _, _ = compose_pattern_from_term("{" + term + "}" + right)
                matches = list(re.finditer(pattern, text, re.DOTALL))
            if len(matches) == 0 and (left == None or right == None):
                pattern, _, _, _ = compose_pattern_from_term("{" + term + "}")
                matches = list(re.finditer(pattern, text, re.DOTALL))

    # Extract the start and end indices of each match
    indices = []
    for match in matches:
        previous_section = 0
        for section_len, start_index, end_index in sections_indices:
            if match.start(1) < section_len:
                indices.append(
                    [
                        note_id,
                        str(start_index + match.start(1) - previous_section),
                        str(start_index + match.end(1) - previous_section),
                        None,
                        term,
                        "",
                        None,
                        None,
                        None,
                        None,
                        substring,
                    ]
                )
                # 0: note_id, 1: start, 2: end, 3: concept_id, 4: term, 5: context, 6: top_concept_id, 7: top_concept_score, 8: top_concept_doc, 9: section, 10: term format extraction
                break
            previous_section = section_len
    if len(matches) != 0 and len(indices) == 0 and not is_submission:
        print("INDICES LEN 0", term, indices, matches, sections_indices)
    if len(matches) == 0 and not is_submission:
        print(f"Term has {len(matches)} matches", substring)
    return indices


def compose_pattern_from_term(substring, extract_with_context=False):
    left, term, right, is_standalone = split_term_with_context(substring)
    if term == None:
        return None, None, None, True
    if extract_with_context == False:
        left = None
        right = None

    # Split the substring into words and escape each word individually
    words = [re.escape(word) for word in term.split()]
    # Join the escaped words with a pattern that matches any whitespace, including newlines
    pattern = r"(" + r"\s+".join(words) + r")"
    if left != None:
        if is_standalone:
            pattern = re.escape(left) + r"[^a-zA-Z]+" + pattern
        else:
            pattern = re.escape(left) + r"[^a-zA-Z]*" + pattern
    if right != None:
        if is_standalone:
            pattern = pattern + r"[^a-zA-Z]+" + re.escape(right)
        else:
            pattern = pattern + r"[^a-zA-Z]*" + re.escape(right)
    # Use lookahead and lookbehind to ensure the match is surrounded by non-word characters or start/end of string
    pattern = r"(?<![a-zA-Z])" + pattern + r"(?![a-zA-Z])"
    return pattern, left, term, right


def remove_annotations_if(term, substring, terms_to_remove):
    if term == "":
        if not is_submission:
            print("term is empty:", substring)
        return True
    # remove also when term has no letter in it
    if not any(c.isalpha() for c in term):
        if not is_submission:
            print("term has no letters:", substring)
        return True
    if term in terms_to_remove:
        if not is_submission:
            print("term in terms_to_remove:", substring)
        return True
    return False


def split_term_with_context(substring, mode="term"):
    left = None
    term = None
    right = None
    is_standalone = True

    term_right = None
    if mode == "term":
        term = substring.strip()
    elif mode == "w$term":
        split_left = substring.split("$")
        if len(split_left) > 1:
            left = split_left[0].strip()
            term = split_left[1].strip()
        else:
            term = split_left[0].strip()
    elif mode == "complex":
        if "$" in substring and ("{" in substring or "}" in substring) and not is_submission:
            print("Both $ and {} in complex mode:", substring)
            split_left = substring.split("$")
            substring = split_left[1]
        if "$" in substring:
            split_left = substring.split("$")
            if len(split_left) > 1:
                left = split_left[0].strip()
                term = split_left[1].strip()
            else:
                term = split_left[0].strip()
        elif "{" in substring:
            is_standalone = False
            split_left = substring.split("{")
            if len(split_left) > 1:
                left = split_left[0].strip()
                term_right = split_left[1]
            else:
                term_right = split_left[0]

            split_right = term_right.split("}")
            term = split_right[0]
            if len(split_right) > 1:
                right = split_right[1].strip()
        else:
            term = substring.strip()
    if not term:
        return None, None, None, True
    term = term.replace("\n", " ").replace("  ", " ").strip()
    return left, term, right, is_standalone
