import os
import sys
import numpy as np
import pandas as pd
import re

import src.document as Document
import src.metrics as metrics
from src.scoring import iou_per_class
import src.vectorDB as LoadVectorize

PROGRAM_PATH = os.path.abspath(os.getcwd())
ASSETS_PATH = os.path.join(PROGRAM_PATH, "assets")
DATA_ROOT = os.path.join(PROGRAM_PATH, "data")
cache_folder = os.path.join(PROGRAM_PATH, "backup")

# notes parquet path from the inference on the full training notes with training annotations set
NOTES_PARQUET = os.path.join(cache_folder, "df_notes_v4_v6.gzip")
# train annotations path
ANNOTATIONS_PATH = os.path.join(ASSETS_PATH, "train_annotations.csv")
# submission path of the inference on the full training notes with training annotations set
path_inf = os.path.join(PROGRAM_PATH, "backup", "submission_v4_v6.csv")

# sentence-transformers model for faiss embeddings
model_path_faiss = "sentence-transformers/all-MiniLM-L12-v2"
# cache folder for sentence-transformers model
model_path_faiss_cache = None

remove_list_path = os.path.join(ASSETS_PATH, "terms_to_remove_BIG.csv")
add_list_path = os.path.join(ASSETS_PATH, "terms_to_add.csv")

faiss_index = os.path.join(ASSETS_PATH, "faiss_index_constitution_all-MiniLM-L12-v2_finetuned")
terminologies_path_extended = os.path.join(ASSETS_PATH, "newdict_snomed_extended.txt")


# The first element of sys.argv is the notes parquet path from the inference on the full training notes with training annotations set
if len(sys.argv) > 1:
    NOTES_PARQUET = sys.argv[1]
# The second element of sys.argv is the train annotations path
if len(sys.argv) > 2:
    ANNOTATIONS_PATH = sys.argv[2]
# The third element of sys.argv is the submission path of the inference on the full training notes with training annotations set
if len(sys.argv) > 3:
    path_inf = sys.argv[3]
# The forth element of sys.argv is the sentence-transformers model name or path for faiss embeddings
if len(sys.argv) > 4:
    model_path_faiss = sys.argv[4]
# The fifth element of sys.argv is the faiss index path
if len(sys.argv) > 5:
    faiss_index = sys.argv[5]
# The sixth element of sys.argv is the terminologies path
if len(sys.argv) > 6:
    terminologies_path_extended = sys.argv[6]


df_notes = pd.read_parquet(NOTES_PARQUET)
for index, row in df_notes.iterrows():
    df_notes.at[index, "result_chunks_inst"] = [inst for inst in row["result_chunks_inst"]]

df_annotations = pd.read_csv(ANNOTATIONS_PATH, sep=",")
df_annotations_inf = pd.read_csv(path_inf)

df_annotations["term"] = pd.Series([[]] * len(df_annotations))
df_annotations_inf["term"] = pd.Series([[]] * len(df_annotations))


def merge_and_extract(df_annotations, df_notes):
    # Merge the dataframes on 'note_id'
    merged_df = pd.merge(
        df_annotations, df_notes[["note_id", "text", "dict_sections"]], on="note_id"
    )

    # Extract the text segment based on 'start' and 'end' into a new column 'term'
    merged_df["term"] = merged_df.apply(
        lambda row: row["text"][row["start"] : row["end"]].replace("\n", ""), axis=1
    )

    return merged_df


df_annotations = merge_and_extract(df_annotations, df_notes)
df_annotations_inf = merge_and_extract(df_annotations_inf, df_notes)


# add len of text as column and number of words in the term
def count_words(text):
    return len(text.split())


df_annotations["len_text"] = df_annotations["term"].apply(len)
df_annotations["nr_words"] = df_annotations["term"].apply(count_words)

df_annotations_inf["len_text"] = df_annotations_inf["term"].apply(len)
df_annotations_inf["nr_words"] = df_annotations_inf["term"].apply(count_words)

df_annotations["section"] = pd.Series([""] * len(df_annotations))
df_annotations_inf["section"] = pd.Series([""] * len(df_annotations_inf))


def extract_section(row):
    dict_sections_ordered = {k: v for k, v in row["dict_sections"].items() if v is not None}
    for section, value_array in sorted(
        list(dict_sections_ordered.items()), key=lambda x: int(x[1][0])
    ):
        if int(row["start"]) >= int(value_array[0]) and int(row["end"]) <= int(value_array[1]):
            return section


df_annotations["section"] = df_annotations.apply(lambda row: extract_section(row), axis=1)
df_annotations_inf["section"] = df_annotations_inf.apply(lambda row: extract_section(row), axis=1)

dict_annotations_span = {}
dict_annotations_inf_span = {}
set_terms = set()

for index, row in df_annotations.iterrows():
    if row["note_id"] not in dict_annotations_span:
        dict_annotations_span[row["note_id"]] = []
    dict_annotations_span[row["note_id"]].append([int(row["start"]), int(row["end"]), row["term"]])
    set_terms.add(row["term"])

for index, row in df_annotations_inf.iterrows():
    if row["note_id"] not in dict_annotations_inf_span:
        dict_annotations_inf_span[row["note_id"]] = []
    dict_annotations_inf_span[row["note_id"]].append(
        [int(row["start"]), int(row["end"]), row["term"]]
    )
    set_terms.add(row["term"])


def pattern_gen(term):
    # Split the substring into words and escape each word individually
    words = [re.escape(word) for word in term.split()]
    # Join the escaped words with a pattern that matches any whitespace, including newlines
    pattern = r"(" + r"\s+".join(words) + r")"
    # Use lookahead and lookbehind to ensure the match is surrounded by non-word characters or start/end of string
    pattern = r"(?<![a-zA-Z])" + pattern + r"(?![a-zA-Z])"

    return pattern


def get_all_occurences(pattern, text):
    matches = list(re.finditer(pattern, text, re.DOTALL))
    return len(matches)


def count_substring_occurrences(term, strings):
    return sum(term in string for string in strings)


dict_term_occurrences = {}

for term in set_terms:
    pattern = pattern_gen(term)
    total_count = df_notes["text"].apply(lambda x: get_all_occurences(pattern, x)).sum()
    dict_term_occurrences[term] = total_count


dict_terms_in_annotations = {}


def check_indices(pair, pairs_list):
    # Iterate through each pair in the list
    for p in pairs_list:
        if pair[1] <= p[0] or pair[0] >= p[1]:
            continue
        # Check if the pair is identical to any pair in the list
        if pair[0] == p[0] and pair[1] == p[1]:
            return "same"
        # Check if the pair is included in any pair in the list
        elif (p[0] <= pair[0] and p[1] >= pair[1]) or (pair[0] <= p[0] and pair[1] >= p[1]):
            return "included"
        # Check if only one index is included in any pair in the list
        elif (p[0] <= pair[0] <= p[1]) != (p[0] <= pair[1] <= p[1]):
            return "intersect"
    # If none of the conditions apply
    return "no"


for note_id, pairs_list in dict_annotations_span.items():
    for pair in pairs_list:
        if pair[2] not in dict_terms_in_annotations:
            dict_terms_in_annotations[pair[2]] = {
                "both": 0,
                "df_annotations": 0,
                "df_annotations_inf": 0,
                "same": 0,
                "included": 0,
                "intersect": 0,
            }

        res = check_indices(pair, dict_annotations_inf_span[note_id])

        if res == "same":
            dict_terms_in_annotations[pair[2]]["same"] += 1
            dict_terms_in_annotations[pair[2]]["both"] += 1
        elif res == "included":
            dict_terms_in_annotations[pair[2]]["included"] += 1
            dict_terms_in_annotations[pair[2]]["both"] += 1
        elif res == "intersect":
            dict_terms_in_annotations[pair[2]]["intersect"] += 1
            dict_terms_in_annotations[pair[2]]["both"] += 1
        else:
            dict_terms_in_annotations[pair[2]]["df_annotations"] += 1

for note_id, pairs_list in dict_annotations_inf_span.items():
    for pair in pairs_list:
        if pair[2] not in dict_terms_in_annotations:
            dict_terms_in_annotations[pair[2]] = {
                "both": 0,
                "df_annotations": 0,
                "df_annotations_inf": 0,
                "same": 0,
                "included": 0,
                "intersect": 0,
            }

        res = check_indices(pair, dict_annotations_span[note_id])

        if res == "included":
            dict_terms_in_annotations[pair[2]]["included"] += 1
            dict_terms_in_annotations[pair[2]]["both"] += 1
        elif res == "intersect":
            dict_terms_in_annotations[pair[2]]["intersect"] += 1
            dict_terms_in_annotations[pair[2]]["both"] += 1
        elif res == "no":
            dict_terms_in_annotations[pair[2]]["df_annotations_inf"] += 1

# Convert the dictionary to a DataFrame
df_terms_summary = pd.DataFrame.from_dict(dict_terms_in_annotations, orient="index").reset_index()
df_terms_summary.columns = [
    "term",
    "both",
    "df_annotations",
    "df_annotations_inf",
    "same",
    "included",
    "intersect",
]

df_terms_summary["occurrences"] = df_annotations["term"].apply(
    lambda term: dict_term_occurrences[term]
)

df_terms_summary_extended_desc = df_terms_summary.sort_values(
    by="df_annotations_inf", ascending=False
)

# Add new column based on the formula
df_terms_summary["to_remove"] = df_terms_summary["both"] / df_terms_summary["df_annotations_inf"]
df_terms_summary["to_remove2"] = (
    df_terms_summary["occurrences"] - df_terms_summary["both"]
) / df_terms_summary["df_annotations_inf"]

df_terms_summary["to_add"] = df_terms_summary["both"] / df_terms_summary["df_annotations"]
df_terms_summary["to_add2"] = df_terms_summary["occurrences"] / (
    df_terms_summary["df_annotations"] + df_terms_summary["both"]
)
df_terms_summary["term_length"] = df_terms_summary["term"].apply(len)

# Sort by 'Age' in descending order
df_terms_summary_desc = df_terms_summary.sort_values(by="occurrences", ascending=False)

print(df_terms_summary_desc)


# compute IoU
def get_iou(df_notes):
    df_annotations_inf = Document.notes_to_submision_extended(df_notes)
    df_annotations["start"] = df_annotations["start"].astype(int)
    df_annotations["end"] = df_annotations["end"].astype(int)

    df_annotations_inf["start"] = df_annotations_inf["start"].astype(int)
    df_annotations_inf["end"] = df_annotations_inf["end"].astype(int)

    df_annotations_inf.drop(["top_concept_id"], axis=1, inplace=True)

    iou = np.mean(iou_per_class(df_annotations_inf, df_annotations))
    return iou


# compute IoU
def get_iou_with_accuracy(df_notes):
    df_annotations_inf = Document.notes_to_submision_extended(df_notes)
    accuracy_1, _, _, _ = metrics.compute_metrics_on_classificatin(
        df_annotations_inf, df_annotations
    )
    df_annotations["start"] = df_annotations["start"].astype(int)
    df_annotations["end"] = df_annotations["end"].astype(int)

    df_annotations_inf["start"] = df_annotations_inf["start"].astype(int)
    df_annotations_inf["end"] = df_annotations_inf["end"].astype(int)

    df_annotations_inf.drop(["top_concept_id"], axis=1, inplace=True)
    iou = np.mean(iou_per_class(df_annotations_inf, df_annotations))
    return iou, accuracy_1


# To remove terms
df_terms_summary_desc = df_terms_summary.sort_values(by=["df_annotations_inf"], ascending=[False])
# df_terms_summary_desc = df_terms_summary_desc[((df_terms_summary_desc['to_remove'] < 0.15) & (df_terms_summary_desc['df_annotations_inf'] > 5)) | ((df_terms_summary_desc['to_remove'] < 0.15) & (df_terms_summary_desc['df_annotations_inf'] > 1) & (df_terms_summary_desc['to_remove2'] > 20))]
df_terms_summary_desc = df_terms_summary_desc[
    (
        (df_terms_summary_desc["to_remove"] < 0.15)
        & (df_terms_summary_desc["df_annotations_inf"] > 5)
    )
    | (
        (df_terms_summary_desc["to_remove"] < 0.15)
        & (df_terms_summary_desc["df_annotations_inf"] < 5)
        & (df_terms_summary_desc["to_remove2"] > 20)
    )
]

print(df_terms_summary_desc)

terms_to_remove = df_terms_summary_desc["term"].tolist()

# Convert the list to a comma-separated string
terms_string = ",".join(terms_to_remove)

# Write the string to a file
file_path = remove_list_path  # './assets/terms_to_remove_big.csv'  # Define your file name and path
with open(file_path, "w") as file:
    file.write(terms_string)

# for each text remove from the remove terms list one by one and score to see if it improves
# Making a deep copy of the original DataFrame
# final terms to remove
final_terms_to_remove = []
dict_final_terms_to_remove = {}
terms_to_remove_BAD = []
dict_terms_to_remove_BAD = {}

ious = get_iou(df_notes)
print(f"macro-averaged character IoU metric: {ious}")

df_deep_with_remove = df_notes.copy(deep=True)
for term in terms_to_remove:  # final_terms_to_remove
    for index, row in df_deep_with_remove.iterrows():
        # Filtering out the lists where the fourth element is "term"
        filtered_lists = [lst for lst in row["result_chunks_inst"] if lst[4] != term]

        # Updating the row with the filtered list
        df_deep_with_remove.at[index, "result_chunks_inst"] = filtered_lists

ious_with_remove = get_iou(df_deep_with_remove)
print(f"macro-averaged character IoU metric with remove: {ious_with_remove}")


def get_original_term(term):
    return term.split("|", 1)[0]


def do_search(FAISS_DB, df_dict, term):
    # docs = FAISS_DB.max_marginal_relevance_search(term[5], k=20)
    docs = FAISS_DB.similarity_search_with_score(term, k=20)
    # print(docs[0])
    top_docs = [(str(df_dict[get_original_term(d[0].page_content)]), d[1]) for d in docs]
    return top_docs[0][0]


FAISS_DB = LoadVectorize.load_db(faiss_index, model_path_faiss, model_path_faiss_cache)
docs = FAISS_DB.similarity_search_with_score("test", k=20)

df_concepts = LoadVectorize.load_dataset_synonyms(terminologies_path_extended)
df_dict = df_concepts.set_index("concept_name_clean")["concept_id"].to_dict()

df_terms_summary = df_terms_summary.sort_values(by=["df_annotations"], ascending=[False])

df_terms_summary_desc = df_terms_summary.sort_values(by=["df_annotations"], ascending=[False])

df_terms_summary_desc = df_terms_summary_desc[
    (
        df_terms_summary_desc["occurrences"]
        > (df_terms_summary_desc["both"] + df_terms_summary_desc["df_annotations"]) * 0.5
    )
    & (df_terms_summary_desc["df_annotations"] > 5)
    & (df_terms_summary_desc["df_annotations"] > df_terms_summary_desc["df_annotations_inf"])
    & (df_terms_summary_desc["to_add2"] < 1.5)
]

# To add terms
# df_terms_summary_desc = df_terms_summary.sort_values(by=['df_annotations', 'df_annotations_inf'], ascending=[False, False])
df_terms_summary_desc = df_terms_summary.sort_values(by=["to_add"], ascending=[False])
df_terms_summary_desc = df_terms_summary_desc[
    (
        df_terms_summary_desc["occurrences"]
        > (df_terms_summary_desc["both"] + df_terms_summary_desc["df_annotations"]) * 0.5
    )
    & (df_terms_summary_desc["df_annotations"] > 5)
    & (df_terms_summary_desc["df_annotations"] > df_terms_summary_desc["df_annotations_inf"])
    & (df_terms_summary_desc["to_add2"] < 1.5)
]

print(df_terms_summary_desc.head(5))

# Convert the list to a comma-separated string
terms_to_add = []
# Filter terms
for term in df_terms_summary_desc["term"].tolist():
    if len(term) <= 1:
        continue
    if term == "mg":  # not term.isupper():
        continue

    terms_to_add.append(term)

terms_string = ",".join(terms_to_add)

# Write the string to a file
file_path = "./assets/terms_to_add.csv"  # Define your file name and path
with open(file_path, "w") as file:
    file.write(terms_string)

# for each text remove from the remove terms list one by one and score to see if it improves
# Making a deep copy of the original DataFrame
# final terms to remove
final_terms_to_add = []
dict_final_terms_to_add = {}
terms_to_add_BAD = []
dict_terms_to_add_BAD = {}

ious = get_iou(df_notes)
print(f"macro-averaged character IoU metric: {ious}")
Document.mode = "term"

for term in terms_to_add:
    term_concept = do_search(FAISS_DB, df_dict, term)

    if term in terms_to_remove:
        continue
    df_deep_test = df_notes.copy(deep=True)

    for index, row in df_deep_test.iterrows():
        # Add all terms from the add list in the text
        indices = row["result_chunks_inst"]
        sections_indices = [[len(row["text"]), 0, len(row["text"])]]
        add_term_indices = Document.find_full_word_occurrences_test(
            term, row["text"], row["note_id"], sections_indices, "", [], extract_with_context=False
        )

        # if len(add_term_indices) > 0:
        #     print(f"Add term added: {term} nr. times: {len(add_term_indices)}")

        for i, indice in enumerate(add_term_indices):
            add_term_indices[i][3] = str(term_concept)

        # if len(add_term_indices) > 0:
        #     print(add_term_indices)
        #     break

        indices = indices + add_term_indices

        # remove overlaps
        overlaps = []
        indices_sorted = sorted(indices, key=lambda x: int(x[1]))
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

        # Updating the row with the filtered list
        df_deep_test.at[index, "result_chunks_inst"] = filtered_indices_sorted

    ious_with_add = get_iou(df_deep_test)
    print(f"macro-averaged character IoU metric with remove: {ious-ious_with_add}")

    if ious < ious_with_add:
        final_terms_to_add.append(term)
        dict_final_terms_to_add[term] = ious_with_add
    else:
        terms_to_add_BAD.append(term)
        dict_terms_to_add_BAD[term] = ious_with_add
print(len(final_terms_to_add), final_terms_to_add)
print(len(terms_to_add_BAD), terms_to_add_BAD)


terms_string = ",".join(final_terms_to_add)

# Write the string to a file
file_path = add_list_path  # './assets/terms_to_add.csv'  # Define your file name and path
with open(file_path, "w") as file:
    file.write(terms_string)
