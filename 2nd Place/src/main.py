import os
import shutil
from datetime import datetime
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import timm
import torch
import transformers
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


from data import (
    ChunkedDataset,
    add_concept_class,
    parallel_convert_labels_tokens,
)
from model import CustomModel, get_optimizer_params, get_scheduler
from train import train_loop


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def copy_src(cfg, src, dst):
    shutil.copytree(src / "src", str(dst / "src"))


def create_output_dir(cfg):
    if cfg.IS_MASTER:
        output_folder = Path(cfg.OUTPUTS)
        monthf = datetime.now().strftime("%m-%d")
        timef = datetime.now().strftime("%H_%M_%S")
        output_folder = output_folder / monthf / timef
        output_folder.mkdir(exist_ok=True, parents=True)
        (output_folder / "models").mkdir(exist_ok=True, parents=True)
    else:
        output_folder = None

    return output_folder


def init_writer(output_folder):
    if output_folder is None:
        return None
    tb_dir = output_folder / "tb"
    tb_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir, comment="Demo")
    return writer


def get_logger(output_dir):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)

    if output_dir is not None:
        filename = output_dir / "train_logs"
        handler2 = FileHandler(filename=f"{filename}.log")
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger


class Learner:
    def __init__(self, cfg, output_folder):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.output_folder = output_folder
        self.logger = get_logger(output_folder)
        self.writer = init_writer(output_folder)
        note_path = cfg.data["note_path"]
        annotations_path = cfg.data["annotations_path"]
        self.build_data(note_path, annotations_path, cfg.data["sctid_syn_path"])
        self.build_model()

    def build_data(self, note_path: Path, annotations_path: Path, sctid_syn_path: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        if self.output_folder is not None:
            self.tokenizer.save_pretrained(self.output_folder / "tokenizer/")

        note_df = pd.read_csv(note_path)
        ann_df = pd.read_csv(annotations_path)

        ann_df = add_concept_class(ann_df, sctid_syn_path)
        self.logger.debug(f"{note_df.shape=}")
        self.logger.debug(f"{ann_df.shape=}")

        Fold = KFold(n_splits=self.cfg.n_fold, shuffle=True, random_state=42)
        X = np.array(range(len(note_df)))
        for n, (_, val_index) in enumerate(Fold.split(X)):
            note_df.loc[val_index, "fold"] = int(n)
        note_df["fold"] = note_df["fold"].astype(int)

        tdf = parallel_convert_labels_tokens(
            self.tokenizer, note_df, ann_df, n_jobs=self.cfg.num_workers
        )

        folds = list(range(self.cfg.n_fold))
        if self.cfg.split == "all":
            val_fold = 0  # well, score will be 1.0
        else:
            val_fold = folds.pop(self.cfg.split)
        train_fold = folds

        self.tds = ChunkedDataset(
            tokenizer=self.tokenizer,
            fold=train_fold,
            df=tdf,
            max_len=self.cfg.max_len,
            repeat=self.cfg.chunked_repeat,
        )
        self.vds = ChunkedDataset(
            tokenizer=self.tokenizer,
            fold=val_fold,
            df=tdf,
            max_len=self.cfg.max_len,
            repeat=self.cfg.chunked_repeat,
        )

        collate = transformers.DataCollatorForTokenClassification(
            self.tokenizer, max_length=self.cfg.max_len
        )
        shuffle = False

        if self.cfg.PARALLEL.DDP:
            sampler = DistributedSampler(self.tds)
        else:
            sampler = None

        self.tdl = torch.utils.data.DataLoader(
            self.tds,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        self.vdl0 = torch.utils.data.DataLoader(
            self.vds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def build_model(self):
        num_classes = len(self.tds.label2id)
        self.model = CustomModel(
            model_name=self.cfg.model, fc_dropout=self.cfg.fc_dropout, num_classes=num_classes
        )
        try:
            fc_state = torch.load(Path(self.cfg.model) / "fc.pth")
            self.model.load_state_dict(fc_state, strict=False)
        except:
            pass

        if self.cfg.IS_MASTER:
            torch.save(self.model.config, self.output_folder / "models/config.pth")

        self.model.to(self.device)
        self.model_ema = timm.utils.ModelEmaV2(self.model, decay=self.cfg.model_EMA, device=None)

        optimizer_parameters = get_optimizer_params(
            self.model,
            encoder_lr=self.cfg.encoder_lr,
            decoder_lr=self.cfg.decoder_lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.opt = torch.optim.AdamW(
            optimizer_parameters, lr=self.cfg.encoder_lr, eps=self.cfg.eps, betas=self.cfg.betas
        )
        num_train_steps = int(len(self.tds) / self.cfg.batch_size * self.cfg.epochs)
        self.scheduler = get_scheduler(self.cfg, self.opt, num_train_steps)

        if self.cfg.PARALLEL.DDP:
            self.model = DDP(
                self.model, device_ids=[self.cfg.PARALLEL.LOCAL_RANK], find_unused_parameters=True
            )
        class_weights = torch.tensor(self.cfg.class_weights)
        class_weights = class_weights.half().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights, reduction="none", ignore_index=-100
        )


def parallel_init(cfg):
    cfg.IS_MASTER = cfg.PARALLEL.LOCAL_RANK == 0
    torch.cuda.set_device(cfg.PARALLEL.LOCAL_RANK)
    if cfg.PARALLEL.DDP:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = cfg.torch.benchmark
    torch.backends.cudnn.deterministic = cfg.torch.deterministic
    torch.set_anomaly_enabled(cfg.torch.detect_anomaly)
    torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(config_path="../configs", config_name="snom", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = dotdict(cfg)
    cfg.PARALLEL = dotdict(cfg.PARALLEL)
    cfg.torch = dotdict(cfg.torch)

    parallel_init(cfg)

    if cfg.IS_MASTER:
        output_folder = create_output_dir(cfg)
        src_folder = Path(os.getcwd())
        copy_src(cfg, src=src_folder, dst=output_folder)
    else:
        output_folder = None

    learner = Learner(cfg, output_folder)
    train_loop(learner)


if __name__ == "__main__":
    main()
