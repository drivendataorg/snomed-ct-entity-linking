from pathlib import Path

import pandas as pd
import torch
import transformers
from infer import InferDataset, predict_spans_ensemble
from loguru import logger
from model import CustomModel


class PredictModel:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        self.model = CustomModel(model_name=checkpoint, fc_dropout=0, num_classes=7)
        fc_state = torch.load(checkpoint / "fc.pth")
        self.model.load_state_dict(fc_state, strict=False)
        self.model.cuda()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
        logger.info(f"model init: {checkpoint}")

    def predict(self, encoded):
        with torch.no_grad(), torch.cuda.amp.autocast():
            input_ids = encoded["input_ids"].unsqueeze(0).to(self.model.model.device)
            pred = self.model(input_ids=input_ids)
        return pred

    def eval(self):
        self.model.eval()

    @property
    def device(self):
        return self.model.model.device


class EnsembleModel:
    ZERO_CLASS_SCALE = 0.35

    def __init__(self, models):
        self.models = models
        self.tokenizer = models[0].tokenizer
        self.label2id = {
            "O": 0,
            "B-find": 1,
            "I-find": 2,
            "B-proc": 3,
            "I-proc": 4,
            "B-body": 5,
            "I-body": 6,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def predict(self, encoded):
        preds = []
        for model in self.models:
            pred = model.predict(encoded)[0]
            preds.append(pred)
        res = torch.stack(preds)
        pred = res.mean(0)
        pred[:, 0] = pred[:, 0] * self.ZERO_CLASS_SCALE
        pred = pred.softmax(-1)
        return pred.cpu()

    def eval(self):
        for model in self.models:
            model.eval()


def init_models(*checkpoints):
    models = [PredictModel(c) for c in checkpoints]
    ensemble = EnsembleModel(models)
    return ensemble


def fisrt_stage(checkpoints: list[Path], note_df: pd.DataFrame) -> pd.DataFrame:
    ensemble = init_models(*checkpoints)
    ensemble.eval()
    ds = InferDataset(note_df, ensemble.tokenizer, 512)
    mentions, _ = predict_spans_ensemble(ensemble, ds, ensemble.id2label)
    res = []
    for k, note_mentions in mentions.items():
        for m in note_mentions:
            res.append([m[0], m[1], m[2], m[3], k])
    first_stage_df = pd.DataFrame(res)
    first_stage_df.columns = ["start", "end", "text", "class", "note_id"]
    return first_stage_df
