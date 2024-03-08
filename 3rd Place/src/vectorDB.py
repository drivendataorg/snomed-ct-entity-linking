import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def get_embeddings(
    model_path_faiss: Union[Path, str], model_path_faiss_cache: Optional[Union[Path, str]]
):
    embeddings = HuggingFaceEmbeddings(
        model_name=str(model_path_faiss),
        cache_folder=None if model_path_faiss_cache is None else str(model_path_faiss_cache),
    )
    return embeddings


def vectorize(faiss_index, model_path_faiss, model_path_faiss_cache, df_concepts):
    embeddings = get_embeddings(model_path_faiss, model_path_faiss_cache)
    terminologies = df_concepts["concept_name_clean"].tolist()
    db_faiss = FAISS.from_texts(terminologies, embeddings)
    db_faiss.save_local(faiss_index)
    return db_faiss


def replace_substring(text, sub):
    modified_text = re.sub(re.escape("(" + sub + ")"), "", text)
    return modified_text.strip()


def load_dataset(terminologies_path):
    if not Path(terminologies_path).exists():
        print("File does not exist at path:", terminologies_path)
        sys.exit()

    df_concepts = pd.read_csv(terminologies_path)
    df_concepts["concept_name_clean"] = df_concepts.apply(
        lambda row: replace_substring(row["concept_name"], row["hierarchy"]), axis=1
    )
    return df_concepts


def load_dataset_synonyms(terminologies_path):
    return pd.read_csv(terminologies_path, sep="\t").rename(
        columns={"term": "concept_name_clean", "code": "concept_id"}
    )


def load_db(faiss_index, model_path_faiss, model_path_faiss_cache):
    """Loads vectorstore from disk and creates BM25 retriever"""
    embeddings = get_embeddings(model_path_faiss, model_path_faiss_cache)
    return FAISS.load_local(faiss_index, embeddings)


def generate_db(faiss_index, model_path_faiss, model_path_faiss_cache, terminologies_path):
    df_concepts = load_dataset_synonyms(terminologies_path)
    FAISS_DB = vectorize(faiss_index, model_path_faiss, model_path_faiss_cache, df_concepts)
    query = "Amygdalo hippocampal epilepsy"
    docs = FAISS_DB.max_marginal_relevance_search(query, k=3)
    print(docs)


def is_gpu_available():
    try:
        return subprocess.check_call(["nvidia-smi"]) == 0

    except FileNotFoundError:
        return False


def test_faiss_n_gpus():
    import faiss

    GPU_AVAILABLE = is_gpu_available()
    print(GPU_AVAILABLE)
    if GPU_AVAILABLE:
        print(faiss.get_num_gpus())
