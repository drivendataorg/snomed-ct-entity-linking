import concurrent.futures
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

from src.config import cache_dir
import src.document as Document
import src.vectorDB as LoadVectorize

PROGRAM_PATH = os.path.abspath(os.getcwd())
ASSETS_PATH = os.path.join(PROGRAM_PATH, "assets")
DATA_ROOT = os.path.join(PROGRAM_PATH, "data")
cache_folder = os.path.join(PROGRAM_PATH, "backup")

# training notes path
NOTES_PATH = os.path.join(DATA_ROOT, "mimic-iv_notes_training_set.csv")

# train annotations path
ANNOTATIONS_PATH = os.path.join(ASSETS_PATH, "train_annotations.csv")

# add faiss docs path
terminologies_path_extended = os.path.join(ASSETS_PATH, "newdict_snomed_extended-150.txt")

# faiss index path
faiss_index = os.path.join(ASSETS_PATH, "faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150")

# OUTPUT FILE used for fine fune the classification model
ANNOTATIONS_train_classification_PATH = os.path.join(
    cache_folder, "annotations_extended_for_classification.gzip"
)

# base mistral model used for the fine tune
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# sentence-transformers model for faiss embeddings
model_path_faiss = "sentence-transformers/all-MiniLM-L12-v2"
# cache folder for sentence-transformers model
model_path_faiss_cache = cache_dir / "hf"

# The first element of sys.argv is the notes path
if len(sys.argv) > 1:
    NOTES_PATH = sys.argv[1]
# The second element of sys.argv is the train annotations path
if len(sys.argv) > 2:
    ANNOTATIONS_PATH = sys.argv[2]
# The third element of sys.argv faiss docs path
if len(sys.argv) > 3:
    terminologies_path_extended = sys.argv[3]
# The fourth element of sys.argv is the faiss index path
if len(sys.argv) > 4:
    faiss_index = sys.argv[4]
# The fifth element of sys.argv is the output file path
if len(sys.argv) > 5:
    ANNOTATIONS_train_classification_PATH = sys.argv[5]
# The sixth element of sys.argv is the base model_id path or name
if len(sys.argv) > 6:
    model_id = sys.argv[6]
# The seventh element of sys.argv is the sentence transformers path used for faiss embeddings
if len(sys.argv) > 7:
    model_path_faiss = sys.argv[7]

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=model_path_faiss_cache)

df_notes = Document.load_notes(NOTES_PATH)
df_notes = Document.split_documents(
    df_notes, tokenizer, max_tokens=100
)  # split in chuncks and split even more if nr tokens too high
df_notes = Document.merge_chuncks(
    df_notes, tokenizer, max_tokens=100
)  # merge chuncks untill you reach max tokens

df_annotations = Document.load_annotations(ANNOTATIONS_PATH)

df_annotations["term"] = pd.Series([[]] * len(df_annotations))


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

df_annotations["section"] = pd.Series([""] * len(df_annotations))

for index, row in df_annotations.iterrows():
    dict_sections = {k: v for k, v in row["dict_sections"].items() if v is not None}
    for section, value_array in sorted(list(dict_sections.items()), key=lambda x: int(x[1][0])):
        if int(row["start"]) >= int(value_array[0]) and int(row["end"]) <= int(value_array[1]):
            df_annotations.loc[index, "section"] = section


def extract_context(text, start, end, num_words_before=5, num_words_after=5):
    # Split the text into words
    words = text.split()

    # Find the start and end indices in terms of words
    words_before = text[:start].split()
    words_after = text[end:].split()
    start_word_index = len(words_before)
    end_word_index = len(words) - len(words_after) - 1

    # Calculate the indices for the words before and after
    start_context_index = max(0, start_word_index - num_words_before)
    end_context_index = min(len(words), end_word_index + num_words_after + 1)

    # Extract the words before, the target, and the words after
    words_before_target = words[start_context_index:start_word_index]
    target_words = words[start_word_index : end_word_index + 1]
    words_after_target = words[end_word_index + 1 : end_context_index]

    # Join and return the context
    return " ".join(words_before_target + target_words + words_after_target)


df_annotations["term_context"] = df_annotations.apply(
    lambda row: extract_context(row["text"], int(row["start"]), int(row["end"])), axis=1
)

# Counting occurrences of each unique value in Column_A
value_counts = df_annotations["section"].value_counts()

# Converting the Series to a dictionary
value_counts_dict = value_counts.to_dict()


def set_training_text(term, section):
    if section == "Allergies:":
        return ("Allergy to " + term + " finding").lower().capitalize()
    elif section == "Chief Complaint:":
        return ("Disorder of " + term).lower().capitalize()
    elif section == "Major Surgical or Invasive Procedure:":
        return (term + " procedure").lower().capitalize()
    elif len(term) <= 4 and term.isupper():
        return term
    else:
        return term.lower().capitalize()


df_annotations["concept_name_clean"] = df_annotations.apply(
    lambda row: set_training_text(row["term"], row["section"]), axis=1
)

print("Dictionary prepared...")

df_annotations_unique = df_annotations
df_annotations_unique.reset_index(drop=True, inplace=True)
df_annotations_unique["inf_concept_id"] = pd.Series([""] * len(df_annotations_unique))
df_annotations_unique["inf_concept_id_score"] = pd.Series([""] * len(df_annotations_unique))
df_annotations_unique["inf_concept_id_20"] = pd.Series([[]] * len(df_annotations_unique))
df_annotations_unique["inf_concept_id_score_20"] = pd.Series([[]] * len(df_annotations_unique))
df_annotations_unique["inf_concept_id_docs_20"] = pd.Series([[]] * len(df_annotations_unique))


def get_original_term(term):
    return term.split("|", 1)[0]


def do_search(FAISS_DB, df_dict, index, term):
    docs = FAISS_DB.similarity_search_with_score(term, k=20)
    top_docs = [
        (str(df_dict[get_original_term(d[0].page_content)]), d[1], d[0].page_content) for d in docs
    ]
    return [index, top_docs]


def concurrent_calls(FAISS_DB, df_dict, chuncks_pairs):
    results = []
    executor = concurrent.futures.ThreadPoolExecutor()
    futures = [
        executor.submit(do_search, FAISS_DB, df_dict, idx, term) for idx, term in chuncks_pairs
    ]
    done, not_done = concurrent.futures.wait(futures)

    if not_done:
        print("Not all futures have completed.")

    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"An error occurred: {e}")

    return sorted(results, key=lambda x: x[0])


FAISS_DB = LoadVectorize.load_db(faiss_index, model_path_faiss, model_path_faiss_cache)
df_concepts_extended = LoadVectorize.load_dataset_synonyms(terminologies_path_extended)
df_dict = df_concepts_extended.set_index("concept_name_clean")["concept_id"].to_dict()
df_dict_sections = df_concepts_extended.set_index("concept_name_clean")["dict_sections"].to_dict()

docs = FAISS_DB.similarity_search_with_score("test", k=20)

nr_threads = os.cpu_count()
multiprocessing = False

df_annotations_unique_train = pd.merge(
    df_annotations_unique,
    df_notes.head(150).reset_index(drop=True)[["note_id"]],
    on="note_id",
    how="inner",
)
df_annotations_unique_test = pd.merge(
    df_annotations_unique,
    df_notes.iloc[150:].reset_index(drop=True)[["note_id"]],
    on="note_id",
    how="inner",
)
df_annotations_unique_test.reset_index(drop=True, inplace=True)

if multiprocessing:
    # Define the step size
    step_size = 100
    print("length of df_annotations_unique: ", len(df_annotations_unique_test))
    # Iterate through the DataFrame in chunks of 10 rows
    for start in range(0, len(df_annotations_unique_test), step_size):
        # Select a chunk of 10 rows
        chunk = df_annotations_unique_test.iloc[start : start + step_size]
        chuncks_pairs = list(zip(chunk.index, chunk["concept_name_clean"]))
        futures = concurrent_calls(FAISS_DB, df_dict, chuncks_pairs)
        for future in futures:
            index = future[0]
            top_docs = future[1]
            df_annotations_unique_test.at[index, "inf_concept_id"] = top_docs[0][0]
            df_annotations_unique_test.at[index, "inf_concept_id_score"] = top_docs[0][1]
            df_annotations_unique_test.at[index, "inf_concept_id_20"] = [
                val for val, score, doc in top_docs
            ]
            df_annotations_unique_test.at[index, "inf_concept_id_score_20"] = [
                score for val, score, doc in top_docs
            ]
            df_annotations_unique_test.at[index, "inf_concept_id_docs_20"] = [
                doc for val, score, doc in top_docs
            ]
        if start + step_size % 1000 == 0:
            print(start + step_size)

else:
    for index, row in df_annotations_unique_test.iterrows():
        _, top_docs = do_search(FAISS_DB, df_dict, index, row["concept_name_clean"])
        df_annotations_unique_test.at[index, "inf_concept_id"] = top_docs[0][0]
        df_annotations_unique_test.at[index, "inf_concept_id_score"] = top_docs[0][1]
        df_annotations_unique_test.at[index, "inf_concept_id_20"] = [
            val for val, score, doc in top_docs
        ]
        df_annotations_unique_test.at[index, "inf_concept_id_score_20"] = [
            score for val, score, doc in top_docs
        ]
        df_annotations_unique_test.at[index, "inf_concept_id_docs_20"] = [
            doc for val, score, doc in top_docs
        ]
        if index % 100 == 0:
            print(index, "out of", len(df_annotations_unique_test))


print("length", len(df_annotations_unique_test))
print(df_annotations_unique_test.head(5))
df_annotations_unique_test.to_parquet(ANNOTATIONS_train_classification_PATH, compression="gzip")
