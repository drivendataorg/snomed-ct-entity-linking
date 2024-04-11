import numpy as np
import pandas as pd
import scipy.sparse as sp


def iou_per_class(
    user_annotations: pd.DataFrame, target_annotations: pd.DataFrame, mean: bool = False
):
    """
    Calculate the IoU metric for each class in a set of annotations.
    """
    # Get mapping from note_id to index in array
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Identify union of categories in GT and PRED
    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))

    # Find max character index in GT or PRED
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

    # Populate matrices for keeping track of character class categorization
    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((n_rows, n_cols), dtype=np.int64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, row.start : row.end] = row.concept_id  # noqa: E203
        return mtx.tocsr()

    gt_mtx = populate_char_mtx(docs.shape[0], max_end, target_annotations)
    pred_mtx = populate_char_mtx(docs.shape[0], max_end, user_annotations)

    # Calculate IoU per category
    ious = []
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        # sparse matrices don't support bitwise operators, but the _cat matrices
        # have bool dtypes so when we multiply/add them we end up with only T/F values
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        if union.sum() == 0:
            ious.append(0)
            print(f"Category {cat} has no union")
            continue
        iou = intersection.sum() / union.sum()
        ious.append(iou)
    if mean:
        return np.mean(ious)
    return ious
