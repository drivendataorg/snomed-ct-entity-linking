"""This is a template for the expected code submission format.

python main.py 2>&1 | tee -a main_out.txt
"""

import os
from typing import Optional
from typer import Typer
from src import snomedctentitylinking
import src.lora_merge as lora_merge

PROGRAM_PATH = os.path.abspath(os.getcwd())
ASSETS_PATH = os.path.join(PROGRAM_PATH, "assets")

app = Typer()


@app.command()
def main(
    NOTES_PATH: str = "data/test_notes.csv",
    base_model_path: str = "mistralai/Mistral-7B-Instruct-v0.2",
    model_path_peft: str = os.path.join(
        PROGRAM_PATH, "models", "Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4"
    ),
    model_path_2_peft: str = os.path.join(
        PROGRAM_PATH, "models", "Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6"
    ),
    model_classification_path_peft: str = os.path.join(
        PROGRAM_PATH, "models", "Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2"
    ),
    model_path_faiss: str = "sentence-transformers/all-MiniLM-L12-v2",
    faiss_index: str = os.path.join(
        ASSETS_PATH, "faiss_index_constitution_all-MiniLM-L12-v2_finetuned"
    ),
    terminologies: str = os.path.join(ASSETS_PATH, "newdict_snomed_extended.txt"),
    ANNOTATIONS_PATH: Optional[str] = None,
    SUBMISSION_PATH: str = "submission.csv",
):
    """Processes notes using the trained models.

    Args:
        NOTES_PATH (str): Path to the notes data CSV file.
        base_model_path (str): Base model path for fine-tuning
        model_path_peft (str): Path to the model for named entity recognition.
        model_path_2_peft (str): Path to the second model for named entity recognition.
        model_classification_path_peft (str): Path to the model for classification.
        model_path_faiss (str): Path to model used for embedding sentences for FAISS
        faiss_index (str): Path to the faiss index.
        terminologies (str): Terminologies used as the FAISS dictionary database.
        ANNOTATIONS_PATH (Optional[str]): Path to the training annotations. Only needed for
            inference when we want to save a dataframe of annotations to then use to generate
            remove and add lists in remove-add-lists.py
        SUBMISSION_PATH (str): Path to save the submission file.

    """

    model_path = os.path.join(PROGRAM_PATH, "models", "model_1")
    model_path_2 = os.path.join(PROGRAM_PATH, "models", "model_2")
    model_classification_path = os.path.join(PROGRAM_PATH, "models", "model_classification")
    model_path_cache = None
    model_path_faiss_cache = None

    lora_merge.merge_lora(base_model_path, model_path_peft, model_path)
    lora_merge.merge_lora(base_model_path, model_path_2_peft, model_path_2)
    lora_merge.merge_lora(
        base_model_path, model_classification_path_peft, model_classification_path
    )

    snomedctentitylinking.pipe(
        NOTES_PATH,
        SUBMISSION_PATH,
        ANNOTATIONS_PATH,
        model_path,
        model_classification_path,
        model_path_cache,
        model_path_faiss,
        model_path_faiss_cache,
        None if ANNOTATIONS_PATH is None else PROGRAM_PATH,
        ASSETS_PATH,
        model_path_2,
        faiss_index,
        terminologies,
    )


if __name__ == "__main__":
    app()
