from pathlib import Path

import pandas as pd
from infer import Matcher, SapEmbedder
from loguru import logger
from tqdm.auto import tqdm


def get_matcher_sap(checkpoint_path: Path):
    embeder = SapEmbedder(checkpoint_path / "model")
    matcher = Matcher(embeder, checkpoint_path / "embeds", prefix="sapbertmean")
    return matcher


def second_stage_match(
    mentions_df: pd.DataFrame, matcher, columns, topk=10, desc="Matching spans w .."
):
    dfs = []
    gg = mentions_df.groupby("note_id")
    for k, note_mentions in tqdm(gg, desc=desc):
        # note_mention : [start, end, text, class, note_id]
        note_mentions = note_mentions[
            ["start", "end", "text", "class", "note_id", "header"]
        ].values.tolist()
        note_match = matcher.match(note_mentions, topk=topk)
        rdf = pd.DataFrame(
            [m for mm in note_match for m in mm],
            columns=["start", "end", "text", "class", "note_id", "header"] + columns,
        )
        dfs.append(rdf)
    tdf = pd.concat(dfs)
    return tdf


def second_stage(
    mentions_df: pd.DataFrame, sap_checkpoint_path: Path, topk: int = 1
) -> pd.DataFrame:
    matcher_sap = get_matcher_sap(sap_checkpoint_path)
    sap_df = second_stage_match(
        mentions_df,
        matcher_sap,
        columns=["sap_cids", "sap_scores"],
        topk=topk,
        desc="Matching spans w sapbert..",
    )
    tdf = sap_df
    logger.debug(f"{tdf.shape=}")
    logger.debug(f"{tdf.head()}")
    return tdf
