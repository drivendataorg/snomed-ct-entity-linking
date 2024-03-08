from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import typer


def iou_per_note(user_annotations: pd.DataFrame, target_annotations: pd.DataFrame) -> List[float]:
    """
    Calculate the IoU metric for each note in a set of annotations.
    """
    user_note_concepts = user_annotations.groupby("note_id")["concept_id"].agg(set)
    target_note_concepts = target_annotations.groupby("note_id")["concept_id"].agg(set)
    ious = {
        nid: iou_score(user_note_concepts[nid], target_note_concepts[nid])
        for nid in user_note_concepts.index
        if nid in target_note_concepts.index
    }

    return ious


def iou_score(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))


def main(
    user_annotations_path: Path,
    target_annotations_path: Path,
):
    """
    Calculate the macro-averaged character IoU metric for each class in a set of annotations.
    """
    user_annotations = pd.read_csv(user_annotations_path)
    target_annotations = pd.read_csv(target_annotations_path)
    ious = iou_per_note(user_annotations, target_annotations)
    print(f"macro-averaged character IoU metric (per note): {np.mean(list(ious.values())):0.4f}.")


if __name__ == "__main__":
    typer.run(main)
