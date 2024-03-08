import numpy as np
import pandas as pd


def compute_metrics(note_id, annotated_words, inference_words):
    annotated_words_set = set([x.lower().replace("\n", "").strip() for x in annotated_words])
    inference_words_set = set([x.lower().replace("\n", "").strip() for x in inference_words])

    # dfmetric = pd.DataFrame(columns=['note_id','TP (both pred and gt)','FP (in pred but not in gt)','FN (in gt but not in pred)','Precision','Recall','F1'])

    # True Positives (TP): Items in both predicted and ground truth
    true_positives = inference_words_set.intersection(annotated_words_set)

    # False Positives (FP): Items in predicted but not in ground truth
    false_positives = inference_words_set.difference(annotated_words_set)

    # False Negatives (FN): Items in ground truth but not in predicted
    false_negatives = annotated_words_set.difference(inference_words_set)

    # Precision calculation
    precision = np.round(
        len(true_positives) / (len(true_positives) + len(false_positives))
        if (len(true_positives) + len(false_positives)) > 0
        else 0,
        3,
    )

    # Recall calculation
    recall = np.round(
        len(true_positives) / (len(true_positives) + len(false_negatives))
        if (len(true_positives) + len(false_negatives)) > 0
        else 0,
        3,
    )

    # F1-score calculation
    f1_score = np.round(
        2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 3
    )

    # [0]: 'TP (both pred and gt)', [1]:'FP (in pred but not in gt)', [2]:'FN (in gt but not in pred)', [3]:'Precision', [4]:'Recall', [5]:'F1'
    metrics = [
        note_id,
        str(len(true_positives)),
        str(len(false_positives)),
        str(len(false_negatives)),
        str(precision),
        str(recall),
        str(f1_score),
    ]
    print(metrics)
    return metrics


def compute_metrics_on_classificatin(df_annotations_inf, df_annotations):
    filtered_df_annotations, _ = filter_matching_rows_2(df_annotations, df_annotations_inf)
    filtered_df_annotations["concept_id_start_end"] = filtered_df_annotations.apply(
        lambda x: str(x["note_id"]) + "_" + str(x["start"]) + "_" + str(x["end"]), axis=1
    )
    df_annotations_inf["concept_id_start_end"] = df_annotations_inf.apply(
        lambda x: str(x["note_id"]) + "_" + str(x["start"]) + "_" + str(x["end"]), axis=1
    )
    f1 = compute_metrics(
        "All annotations F1:",
        filtered_df_annotations["concept_id_start_end"].tolist(),
        df_annotations_inf["concept_id_start_end"].tolist(),
    )

    filtered_df_annotations, filtered_df_annotations_inf = filter_matching_rows(
        df_annotations, df_annotations_inf
    )
    accuracy_1, accuracy_5, accuracy_10, accuracy_20 = compute_accuracy(
        filtered_df_annotations, filtered_df_annotations_inf
    )
    print(accuracy_1, accuracy_5, accuracy_10, accuracy_20)
    return accuracy_1, accuracy_5, accuracy_10, accuracy_20


def filter_matching_rows(df1, df2):
    # Ensuring 'note_id' and 'start' are of the same data type (e.g., string) in both dataframes
    df1["note_id"] = df1["note_id"].astype(str)
    df1["start"] = df1["start"].astype(str)
    df2["note_id"] = df2["note_id"].astype(str)
    df2["start"] = df2["start"].astype(str)

    # Merging dataframes on 'note_id' and 'start' to find matches
    merged_df = pd.merge(df1, df2, on=["note_id", "start"])

    # Extracting the matching 'note_id' and 'start' to filter the original dataframes
    matching_ids = merged_df[["note_id", "start"]].drop_duplicates()

    # Filtering original dataframes to keep only rows with matching 'note_id' and 'start'
    filtered_df1 = pd.merge(df1, matching_ids, on=["note_id", "start"], how="inner")
    filtered_df2 = pd.merge(df2, matching_ids, on=["note_id", "start"], how="inner")

    return filtered_df1, filtered_df2


def filter_matching_rows_2(df1, df2):
    # Ensuring 'note_id' and 'start' are of the same data type (e.g., string) in both dataframes
    df1["note_id"] = df1["note_id"].astype(str)
    df1["start"] = df1["start"].astype(str)
    df2["note_id"] = df2["note_id"].astype(str)
    df2["start"] = df2["start"].astype(str)

    # Merging dataframes on 'note_id' and 'start' to find matches
    merged_df = pd.merge(df1, df2, on=["note_id"])

    # Extracting the matching 'note_id' and 'start' to filter the original dataframes
    matching_ids = merged_df[["note_id"]].drop_duplicates()

    # Filtering original dataframes to keep only rows with matching 'note_id' and 'start'
    filtered_df1 = pd.merge(df1, matching_ids, on=["note_id"], how="inner")
    filtered_df2 = pd.merge(df2, matching_ids, on=["note_id"], how="inner")

    return filtered_df1, filtered_df2


def compute_accuracy(df1, df2):
    # Ensure that the dataframes have the same length
    if len(df1) != len(df2):
        raise ValueError("Dataframes must have the same number of rows")

    # Use apply along with a lambda function to check if concept_id in df1 is a substring of concept_id in df2
    matches_1 = df1.apply(
        lambda x: str(x["concept_id"]) in df2.loc[x.name, "top_concept_id"].split(",")[:1], axis=1
    )
    matches_5 = df1.apply(
        lambda x: str(x["concept_id"]) in df2.loc[x.name, "top_concept_id"].split(",")[:5], axis=1
    )
    matches_10 = df1.apply(
        lambda x: str(x["concept_id"]) in df2.loc[x.name, "top_concept_id"].split(",")[:10], axis=1
    )
    matches_20 = df1.apply(
        lambda x: str(x["concept_id"]) in df2.loc[x.name, "top_concept_id"].split(","), axis=1
    )

    # Calculating accuracy: number of matches divided by total number of rows
    accuracy_1 = matches_1.sum() / len(df1)
    accuracy_5 = matches_5.sum() / len(df1)
    accuracy_10 = matches_10.sum() / len(df1)
    accuracy_20 = matches_20.sum() / len(df1)

    return accuracy_1, accuracy_5, accuracy_10, accuracy_20


def filter_different_rows(df1, df2):
    # Merge the dataframes on 'note_id', 'start', and 'end'
    merged_df = pd.merge(
        df1,
        df2,
        on=["note_id", "start", "end"],
        suffixes=("_df1", "_df2"),
        how="outer",
        indicator=True,
    )

    # Filter rows where the combination of 'note_id', 'start', and 'end' are different
    different_rows = merged_df[merged_df["_merge"] != "both"]

    # Drop the '_merge' column
    different_rows = different_rows.drop(columns=["_merge"])

    return different_rows
