import concurrent.futures
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from langchain import PromptTemplate
from transformers import AutoTokenizer

import src.document as Document
import src.metrics as metrics
import src.model.vLLM as LLM
import src.vectorDB as LoadVectorize
from src.scoring import iou_per_class


def pipe(
    file_paths_notes,
    file_paths_submission,
    file_paths_annotations,
    model_path,
    model_classification_path,
    model_path_cache,
    model_path_faiss,
    model_path_faiss_cache,
    PROGRAM_PATH,
    SRC_PATH,
    model_path_2,
    faiss_index,
    terminologies,
):
    use_remove_list = True
    use_add_list = True
    use_classification = True

    tokenizer_mistral = AutoTokenizer.from_pretrained(model_path, use_fast=True, truncation=False)

    prompt_path = os.path.join(SRC_PATH, "prompts", "prompt.txt")
    prompt_classification_path = os.path.join(SRC_PATH, "prompts", "prompt_classification_v1.txt")

    df_notes_parquet_backup = None
    if PROGRAM_PATH is not None:
        cache_folder = os.path.join(PROGRAM_PATH, "backup")
        # Check if cache_folder exists and create it if not
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        df_notes_parquet_backup = os.path.join(cache_folder, "df_notes_v4_v6.gzip")
        print(df_notes_parquet_backup)
    else:
        cache_folder = None

    if file_paths_annotations is None:
        is_submission = True
    else:
        is_submission = False
        use_remove_list = False
        use_add_list = False

    terms_to_remove = []
    terms_to_add = []
    file_path_terms_to_remove = os.path.join(SRC_PATH, "terms_to_remove_BIG.csv")
    file_path_terms_to_add = os.path.join(SRC_PATH, "terms_to_add.csv")

    if use_remove_list and os.path.exists(file_path_terms_to_remove):
        print("Extra terms to remove loaded from file...")
        # Read list of terms to always add or remove
        with open(file_path_terms_to_remove, "r") as file:
            terms_string = (
                file.readline().strip()
            )  # Read the first line and strip any trailing newline characters
            terms_to_remove = terms_string.split(",")
    else:
        print("No extra terms to remove found")

    if os.path.exists(file_path_terms_to_add):
        # Read list of terms to always add or remove
        with open(file_path_terms_to_add, "r") as file:
            terms_string = (
                file.readline().strip()
            )  # Read the first line and strip any trailing newline characters
            terms_to_add = sorted(terms_string.split(","), key=len, reverse=True)
    else:
        print("No extra terms to add found")

    start_time = time.time()
    sys_msg = load_prompt_template(prompt_path)
    classification_template = load_prompt_template(prompt_classification_path)

    if is_submission:
        df_notes = load_and_process_data(
            file_paths_notes,
            tokenizer_mistral,
            df_notes_parquet_backup,
            sys_msg=sys_msg,
            is_submission=is_submission,
            max_tokens_split=100,
            max_tokens_merge=100,
        )
        df_notes2 = load_and_process_data(
            file_paths_notes,
            tokenizer_mistral,
            df_notes_parquet_backup,
            sys_msg=sys_msg,
            is_submission=is_submission,
            max_tokens_split=500,
            max_tokens_merge=400,
        )
    else:
        df_notes = load_and_process_data(
            file_paths_notes,
            tokenizer_mistral,
            df_notes_parquet_backup,
            sys_msg=sys_msg,
            is_submission=is_submission,
            max_tokens_split=100,
            max_tokens_merge=100,
        )
        df_notes2 = load_and_process_data(
            file_paths_notes,
            tokenizer_mistral,
            df_notes_parquet_backup,
            sys_msg=sys_msg,
            is_submission=is_submission,
            max_tokens_split=500,
            max_tokens_merge=400,
        )
        df_notes.reset_index(drop=True, inplace=True)
        df_notes2.reset_index(drop=True, inplace=True)

    # Add annotations to notes during training
    if not is_submission:
        df_notes, df_annotations = add_annotations_to_notes(file_paths_annotations, df_notes)
        df_notes2, df_annotations = add_annotations_to_notes(file_paths_annotations, df_notes2)

    # Run entity recognition on the notes
    compute(
        model_path,
        model_path_cache,
        df_notes,
        sys_msg,
        terms_to_remove,
        terms_to_add,
        df_notes_parquet_backup,
        is_submission=is_submission,
    )
    compute(
        model_path_2,
        model_path_cache,
        df_notes2,
        sys_msg,
        terms_to_remove,
        terms_to_add,
        df_notes_parquet_backup,
        is_submission=is_submission,
    )

    df_notes = merge_dataframes_inf_annotations(df_notes, df_notes2)

    if use_remove_list and len(terms_to_remove) > 0:
        for index_o, row in df_notes.iterrows():
            df_notes.at[index_o, "result_chunks_inst"] = [
                inst
                for inst in df_notes.at[index_o, "result_chunks_inst"]
                if inst[4] not in terms_to_remove
            ]

    if use_add_list and len(terms_to_add) > 0:
        for index_o, row in df_notes.iterrows():
            for add_term in terms_to_add:
                sections_indices = [[len(row["text"]), 0, len(row["text"])]]
                add_term_indices = Document.find_full_word_occurrences_test(
                    add_term,
                    row["text"],
                    row["note_id"],
                    sections_indices,
                    "",
                    [],
                    extract_with_context=False,
                )
                df_notes.at[index_o, "result_chunks_inst"] = (
                    df_notes.at[index_o, "result_chunks_inst"] + add_term_indices
                )
            df_notes.at[index_o, "result_chunks_inst"] = Document.remove_dupplicates(
                df_notes.at[index_o, "result_chunks_inst"]
            )

    pretty_printer_timer(start_time)

    if not is_submission:
        df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

    if not is_submission:
        df_notes["metrics"] = df_notes.progress_apply(
            lambda x: metrics.compute_metrics(
                x["note_id"],
                [term[4] for term in x["annotated_words"]],
                [term[4] for term in x["result_chunks_inst"]],
            ),
            axis=1,
        )

    if not is_submission:
        df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

    add_context(df_notes)
    if not is_submission:
        df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

    assign_condition(
        df_notes,
        faiss_index,
        terminologies,
        model_path_faiss,
        model_path_faiss_cache,
        multiprocessing=False,
    )

    pretty_printer_timer(start_time)

    if use_classification:
        if not is_submission:
            df_annotations_inf = save_submision(
                df_notes, file_paths_submission, is_submission=is_submission
            )
            metrics.compute_metrics_on_classificatin(df_annotations_inf, df_annotations)
            filtered_df_annotations, _ = metrics.filter_matching_rows_2(
                df_annotations, df_annotations_inf
            )
            df_annotations_inf["start"] = df_annotations_inf["start"].astype(int)
            df_annotations_inf["end"] = df_annotations_inf["end"].astype(int)
            filtered_df_annotations["start"] = filtered_df_annotations["start"].astype(int)
            filtered_df_annotations["end"] = filtered_df_annotations["end"].astype(int)

            ious = iou_per_class(df_annotations_inf, filtered_df_annotations)
            print(f"macro-averaged character IoU metric: {np.mean(ious):0.4f}")
            df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

        improve_assign_condition(
            model_classification_path,
            model_path_cache,
            df_notes,
            classification_template,
            is_submission=is_submission,
        )

    df_annotations_inf = save_submision(
        df_notes, file_paths_submission, is_submission=is_submission
    )

    if not is_submission:
        metrics.compute_metrics_on_classificatin(df_annotations_inf, df_annotations)
        filtered_df_annotations, _ = metrics.filter_matching_rows_2(
            df_annotations, df_annotations_inf
        )
        df_annotations_inf["start"] = df_annotations_inf["start"].astype(int)
        df_annotations_inf["end"] = df_annotations_inf["end"].astype(int)
        filtered_df_annotations["start"] = filtered_df_annotations["start"].astype(int)
        filtered_df_annotations["end"] = filtered_df_annotations["end"].astype(int)

        ious = iou_per_class(df_annotations_inf, filtered_df_annotations)
        print(f"macro-averaged character IoU metric: {np.mean(ious):0.4f}")
        df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

    # Convert execution_time to h:m:s format

    pretty_printer_timer(start_time)
    print("length of df_notes: " + str(len(df_notes)))


# ******************************************************************************************************************************
#                                                       Prompt
# ******************************************************************************************************************************


def load_prompt_template(prompt_path):
    # read file
    prompt_file = open(prompt_path, "r")

    sys_msg = prompt_file.read()
    return sys_msg


def instruction_format(sys_msg: str, query: str):
    # note, don't "</s>" to the end
    template = """[INST] {sys_msg}
    
# Hospital discharge note:
{query} [/INST]"""

    prompt_template = PromptTemplate(input_variables=["sys_msg", "query"], template=template)
    prompt = prompt_template.format(sys_msg=sys_msg, query=query)
    return prompt


# ******************************************************************************************************************************
#                                                       Data
# ******************************************************************************************************************************


def load_and_process_data(
    NOTES_PATH,
    tokenizer,
    df_notes_parquet_backup,
    sys_msg,
    is_submission,
    max_tokens_split=100,
    max_tokens_merge=100,
):
    if False and os.path.exists(df_notes_parquet_backup) and not is_submission:
        df_notes = pd.read_parquet(df_notes_parquet_backup)
    else:
        df_notes = Document.load_notes(NOTES_PATH)

        df_notes = Document.split_documents(
            df_notes, tokenizer, max_tokens=max_tokens_split
        )  # split in chuncks and split even more if nr tokens too high

        df_notes = Document.merge_chuncks(
            df_notes, tokenizer, max_tokens=max_tokens_merge
        )  # merge chuncks untill you reach max tokens

    if not is_submission:
        Document.minmax_text_tokens(df_notes, tokenizer)
        Document.minmax_chunck_tokens(df_notes, tokenizer)
    return df_notes


def add_annotations_to_notes(ANNOTATIONS_PATH, df_notes):
    df_annotations = Document.load_annotations(ANNOTATIONS_PATH)

    df_notes = Document.add_annotations_to_notes(df_notes, df_annotations)
    return df_notes, df_annotations


# ******************************************************************************************************************************
#                                                       Computation
# ******************************************************************************************************************************
def compute(
    model_path,
    model_path_cache,
    df_notes,
    sys_msg,
    terms_to_remove,
    terms_to_add,
    df_notes_parquet_backup,
    is_submission,
):
    """Runs an entity recognition model on notes.

    Args:
        model_path (str): Path to the entity recognition model, e.g., models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4
        model_path_cache (str): Path to the cache for the entity recognition model.
        df_notes (pd.DataFrame): The notes to process.
        sys_msg (str): The system message to use in the prompts.
        terms_to_remove (list): A list of terms to remove from the results.
        terms_to_add (list): A list of terms to add to the results.
        df_notes_parquet_backup (str): The path to the backup parquet file for the notes.
        is_submission (bool): If true, print more debugging information and save the result to df_notes_parquet_backup.
    """
    start_time = time.time()
    llm = LLM.instantiate(model_path, model_path_cache)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=model_path_cache, use_fast=True, truncation=False
    )

    df_notes["text_chunks_inst"] = [[] for _ in range(len(df_notes))]
    df_notes["result_chunks_inst"] = [[] for _ in range(len(df_notes))]
    length = len(df_notes) - 1

    for index_o, row in df_notes.iterrows():
        if not is_submission:
            print(
                "******************************************************************************************************************************"
            )
            print("\t\t\t\tINDEX: " + str(index_o))
            print(
                "--                       --                              --                     --                     --                   --"
            )
            print([c[4] for c in df_notes.iloc[index_o]["annotated_words"]])
            print(
                "******************************************************************************************************************************"
            )

        df_notes.iloc[index_o]["text_chunks_inst"] = []

        res_list = []
        prompts = []
        for index_i, item in enumerate(row["text_chunks_mini_sections"]):
            prompt = instruction_format(sys_msg, item)
            prompts.append(prompt)

        prompts_inf = LLM.inference(llm, prompts)
        for index_i, item in enumerate(row["text_chunks_mini_sections"]):
            df_notes.iloc[index_o]["text_chunks_inst"].append(prompts_inf[index_i])
            annotations_chunk, chunks_to_check = Document.get_annotations_from_inference_new(
                note_id=row["note_id"],
                text=row["text"],
                dict_sections=row["dict_sections"],
                chunk_index=row["text_chunks_mini_sections_indices"][index_i],
                target=prompts_inf[index_i],
            )

            # redo inferences for chunks that are not the same as the original
            prompts_redo = []
            if len(chunks_to_check) > 0:
                if not is_submission:
                    print("There are some prompts to redo...")
                for ctc in chunks_to_check:
                    prompt = instruction_format(sys_msg, row["text"][ctc[0] : ctc[1]])
                    prompts_redo.append(prompt)
                prompts_redo_inf = LLM.inference(llm, prompts_redo)
                for i_ctc, ctc in enumerate(chunks_to_check):
                    annotations_chunk_redo, chunks_to_check_bis = (
                        Document.get_annotations_from_inference_new(
                            note_id=row["note_id"],
                            text=row["text"],
                            dict_sections=row["dict_sections"],
                            chunk_index=ctc,
                            target=prompts_redo_inf[i_ctc],
                        )
                    )
                    if not is_submission and len(chunks_to_check_bis) > 0:
                        print(
                            f"There are still some chunks no perfect {len(chunks_to_check_bis)}..."
                        )
                    annotations_chunk = annotations_chunk + annotations_chunk_redo
                    annotations_chunk = sorted(annotations_chunk, key=lambda x: x[1])

            df_notes.at[index_o, "result_chunks_inst"] = (
                df_notes.at[index_o, "result_chunks_inst"] + annotations_chunk
            )
            df_notes.at[index_o, "result_chunks_inst"] = Document.remove_dupplicates(
                df_notes.at[index_o, "result_chunks_inst"]
            )

        if not is_submission:
            df_notes.to_parquet(df_notes_parquet_backup, compression="gzip")

        if not is_submission:
            print([chunck_inst[4] for chunck_inst in df_notes.at[index_o, "result_chunks_inst"]])
            print(
                "--                       --                              --                     --                     --                   --"
            )
            # Convert execution_time to h:m:s format
            pretty_printer_timer(start_time)
            print("Nr of chunks: " + str(len(df_notes.iloc[index_o]["text_chunks_mini_sections"])))
            print(
                "******************************************************************************************************************************"
            )
            print(f"Index: {index_o} completed out of {length}.")
        else:
            print(f"Index: {index_o} completed out of {length}.")


# ******************************************************************************************************************************
#                                                       Merge annotations inf in dataframes and remove duplicates
# ******************************************************************************************************************************


def merge_dataframes_inf_annotations(df1, df2):
    for index_o, row in df1.iterrows():
        df1.at[index_o, "result_chunks_inst"] = Document.remove_dupplicates(
            df1.at[index_o, "result_chunks_inst"] + df2.at[index_o, "result_chunks_inst"]
        )

    return df1


# ******************************************************************************************************************************
#                                                       Add context to each term
# ******************************************************************************************************************************
def add_context(df_notes: pd.DataFrame) -> Optional:
    """Adds context to each term in the notes. The input DataFrame is updated in place."""
    for index_o, row in df_notes.iterrows():
        prev_item = None
        for index_i, item in enumerate(row["result_chunks_inst"]):
            df_notes.at[index_o, "result_chunks_inst"][index_i][5] = (
                Document.extract_context_for_terms(
                    row["text"], item, row["result_chunks_inst"], row["dict_sections"]
                )
            )


# ******************************************************************************************************************************
#                                                       Assign condition for each term
# ******************************************************************************************************************************
def assign_condition(
    df_notes,
    faiss_index,
    terminologies_path,
    model_path_faiss,
    model_path_faiss_cache,
    multiprocessing=False,
):
    faiss_db = LoadVectorize.load_db(faiss_index, model_path_faiss, model_path_faiss_cache)
    df_concepts = LoadVectorize.load_dataset_synonyms(terminologies_path)
    df_dict = df_concepts.set_index("concept_name_clean")["concept_id"].to_dict()

    for index_o, row in df_notes.iterrows():
        if multiprocessing:
            futures = concurrent_calls(faiss_db, df_dict, row["result_chunks_inst"])
            for index_i, res_pair in enumerate(futures):
                top_docs = res_pair[1]
                df_notes.at[index_o, "result_chunks_inst"][index_i] = list(
                    df_notes.at[index_o, "result_chunks_inst"][index_i]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][3] = top_docs[0][0]
                df_notes.at[index_o, "result_chunks_inst"][index_i][6] = ",".join(
                    [str(val) for val, score, doc in top_docs]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][7] = ",".join(
                    [str(score) for val, score, doc in top_docs]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][8] = ",".join(
                    [doc for val, score, doc in top_docs]
                )
        else:
            for index_i, term in enumerate(row["result_chunks_inst"]):
                top_docs = do_search(faiss_db, df_dict, term)[1]
                df_notes.at[index_o, "result_chunks_inst"][index_i] = list(
                    df_notes.at[index_o, "result_chunks_inst"][index_i]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][3] = top_docs[0][0]
                df_notes.at[index_o, "result_chunks_inst"][index_i][6] = ",".join(
                    [str(val) for val, score, doc in top_docs]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][7] = ",".join(
                    [str(score) for _, score, _ in top_docs]
                )
                df_notes.at[index_o, "result_chunks_inst"][index_i][8] = ",".join(
                    [doc for _, _, doc in top_docs]
                )
        print("Document nr: " + str(index_o) + " assigned")


def gen_faiss_list(faiss: list):
    result = []
    for i, data in enumerate(faiss):
        id, score, docs = data
        result.append(f"{i}: {id} - {docs}")
    return "\n".join(result)


def instruction_format_classification(
    template: str, term: str, section: str, context: str, faiss_list: list
):
    prompt_template = PromptTemplate(
        input_variables=["term", "section", "context", "faiss"], template=template
    )
    prompt = prompt_template.format(
        term=term, section=section, context=context.strip(), faiss=gen_faiss_list(faiss_list)
    )
    return prompt


def extract_context(
    text: str, start: int, end: int, num_words_before: int = 5, num_words_after: int = 5
) -> str:
    """Gets the context around a target span of text.

    Args:
        text (str): The complete text containing the span
        start (int): the start character of the span
        end (int): the end character of the span
        num_words_before (int): The number of words to include before the target span
        num_words_after (int): The number of words to include after the target span

    Returns:
        str: The text starting num_words_before and ending num_words_after the target span.
    """
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


def is_integer(s):
    try:
        int(s)  # Try converting the string to an integer
        return True
    except ValueError:
        return False


def improve_assign_condition(
    model_path, model_path_cache, df_notes, classification_template, is_submission
):
    """Classifies the extracted terms using an LLM. Updates the input DataFrame in place."""
    llm = LLM.instantiate(model_path, model_path_cache)

    for index_o, row in df_notes.iterrows():
        prompts = []
        for inst in row["result_chunks_inst"]:
            term_context = extract_context(row["text"], int(inst[1]), int(inst[2]))
            top_concepts_id = inst[6].split(",")
            top_concepts_score = inst[7].split(",")
            top_concepts_text = inst[8].split(",")
            prompt = instruction_format_classification(
                template=classification_template,
                term=inst[5],
                section=inst[9],
                context=term_context,
                faiss_list=list(
                    zip(top_concepts_id[:10], top_concepts_score[:10], top_concepts_text[:10])
                ),
            )
            prompts.append(prompt)
        prompts_inf = LLM.inference(llm, prompts)

        list_to_remove = []
        for index_i, item in enumerate(row["result_chunks_inst"]):
            if is_integer(prompts_inf[index_i]):
                pos_value = int(prompts_inf[index_i])
                if 0 <= pos_value <= 9:
                    concept_id_top = row["result_chunks_inst"][index_i][6].split(",")

                    df_notes.at[index_o, "result_chunks_inst"][index_i][3] = concept_id_top[
                        pos_value
                    ]
                elif pos_value == -1:
                    list_to_remove.append(item)
        if not is_submission:
            print("List to remove:", [t[4] for t in list_to_remove])
        df_notes.at[index_o, "result_chunks_inst"] = [
            inst
            for inst in df_notes.at[index_o, "result_chunks_inst"]
            if inst not in list_to_remove
        ]


def get_original_term(term):
    return term.split("|", 1)[0]


def do_search(FAISS_DB, df_dict, term) -> Tuple[int, List[Tuple[str, float, str]]]:
    """Searches for a term + the surrounding context in the FAISS database.

    Args:
        df_dict (dict): A dictionary mapping clean concept names to concept IDs
        term (list): A list containing data about the extracted term withh the following items:

            0: note_id
            1: start index of the term in terms of the text
            2: end index of the term in terms of the text
            3: Concept ID of the top match
            4: Text of the term, i.e., text[start_index:end_index] (with newlines removed)
            5: Context (added by src.snomedctentitylinking.add_context at a later step)
            6: Concept IDs of the top matching documents, comma-separated (added later)
            7: Match scores of the top matching documents, comma-separated (added later)
            8: Text of the top matching documents, comma-separated (added later)
            9: Section
            10: Term format extraction

    Returns:
        A list with two items, the start index of the match and a list of the matching documents
        from the vector store where each item in the list has the format:

            (concept ID of matched document, similarity score, matched document text)
    """
    # term[5] is the extracted term plus a few words before and after
    docs = FAISS_DB.similarity_search_with_score(term[5], k=20)
    top_docs = [
        (str(df_dict[get_original_term(doc.page_content)]), score, doc.page_content)
        for doc, score in docs
    ]
    return int(term[1]), top_docs


def concurrent_calls(FAISS_DB, df_dict, chuncks):
    results = []
    # Create a ThreadPoolExecutor
    executor = concurrent.futures.ThreadPoolExecutor()
    # Submit tasks to the thread pool
    futures = [executor.submit(do_search, FAISS_DB, df_dict, chunck) for chunck in chuncks]

    # Collect and print the results as they become available
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"An error occurred: {e}")

    return sorted(results, key=lambda x: x[0])


def save_submision(df_notes, SUBMISSION_PATH, is_submission):
    if is_submission:
        df_spans = Document.notes_to_submision(df_notes)
    else:
        df_spans = Document.notes_to_submision_extended(df_notes)
    # check duplicates
    duplicates = df_spans.duplicated(subset=["note_id", "start"], keep=False)
    if len(df_spans[duplicates]) > 0:
        print("Duplicates found")
        print(df_spans[duplicates])

    df_spans[["note_id", "start", "end", "concept_id"]].to_csv(SUBMISSION_PATH, index=False)
    if is_submission:
        print("Submission saved...")
    else:
        print("Submission extended saved...")
    return df_spans


def pretty_printer_timer(start_time):
    end_time = time.time()
    execution_time = int(end_time - start_time)
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    execution_time_formatted = f"{hours}:{minutes:02d}:{seconds:02d}"
    print(f"Computation time: {execution_time_formatted}")
