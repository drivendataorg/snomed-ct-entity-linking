import math
import shutil
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from tqdm.auto import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def train_fn(L, epoch):
    L.model.train()
    device = L.device
    scaler = torch.cuda.amp.GradScaler(enabled=L.cfg.apex)
    losses = AverageMeter()
    global_step = 0

    for step, inputs in enumerate(tqdm(L.tdl)):
        labels = inputs.pop("labels")

        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=L.cfg.apex):
            y_preds = L.model(**inputs)

        b, c, n = y_preds.shape
        y_preds = y_preds.view(b * c, n)
        labels = labels.flatten()

        loss = L.criterion(y_preds, labels)
        loss = torch.masked_select(loss, labels != -100).mean()

        # fix me
        if L.cfg.gradient_accumulation_steps > 1:
            loss = loss / L.cfg.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if (step + 1) % L.cfg.gradient_accumulation_steps == 0:
            scaler.step(L.opt)
            scaler.update()
            L.opt.zero_grad(set_to_none=True)

            if L.model_ema is not None:
                m = L.model.module if L.cfg.PARALLEL.DDP else L.model
                L.model_ema.update(m)

            global_step += 1
            if L.cfg.batch_scheduler:
                L.scheduler.step()

    return losses.avg


def valid_fn(L, valid_loader):
    device = L.device

    losses = AverageMeter()
    if L.model_ema is not None:
        model = L.model_ema.module
    else:
        model = L.model

    model.eval()
    preds = []
    gts = []

    for step, inputs in enumerate(valid_loader):
        labels = inputs.pop("labels")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=L.cfg.apex):
            y_preds = model(**inputs)

        b, c, n = y_preds.shape
        y_preds = y_preds.view(b * c, n)
        labels = labels.flatten()
        loss = L.criterion(y_preds, labels)
        loss = torch.masked_select(loss, labels != -100).mean()

        pr = y_preds[torch.where(labels != -100)].cpu()
        gt = torch.masked_select(labels, labels != -100).cpu()

        preds.append(pr)
        gts.append(gt)

        if L.cfg.gradient_accumulation_steps > 1:
            loss = loss / L.cfg.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

    preds = torch.vstack(preds)  # N, 7
    gts = torch.cat(gts)
    return losses.avg, gts, preds


def save_model(model, tokenizer, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    fc_state = {}
    for k, v in state.items():
        if k.startswith("fc"):
            fc_state[k] = v
    torch.save(fc_state, save_dir / "fc.pth")


def clean_checkpoint(dstf, n):
    pths = list(dstf.glob("**/model.safetensors"))
    pths = sorted(pths, key=lambda x: x.stat().st_mtime)
    for pth in pths[:-n]:
        shutil.rmtree(pth.parent)


def train_loop(learner):
    best_score = 0
    L = learner

    for epoch in range(L.cfg.epochs):
        start_time = time.time()
        try:
            L.tdl.sampler.set_epoch(epoch)
        except Exception:
            pass

        # train
        avg_loss = train_fn(L, epoch)

        # eval
        avg_val_loss, gts, preds = valid_fn(L, L.vdl0)
        preds = torch.softmax(preds.float(), 1)

        # merge I- and B- classes:
        id2label = L.tds.id2label.copy()
        for k, v in id2label.items():
            if "I-" in v:
                id2label[k] = v.replace("I-", "B-")
        gts = [id2label[i] for i in gts.tolist()]
        preds = torch.stack(
            [
                preds[:, 0],
                preds[:, 1] + preds[:, 2],
                preds[:, 3] + preds[:, 4],
                preds[:, 5] + preds[:, 6],
            ],
            dim=1,
        )

        label2id = {
            "O": 0,
            "B-find": 1,
            "B-proc": 2,
            "B-body": 3,
        }
        gts = [label2id[i] for i in gts]
        auc = roc_auc_score(gts, preds, average="macro", multi_class="ovo")
        preds = torch.argmax(preds, 1)
        iou = calc_iou(preds, torch.tensor(gts))
        labels = list(label2id.values())
        f1_mic = f1_score(gts, preds, average="micro", labels=labels)
        f1_mac = f1_score(gts, preds, average="macro", labels=labels)
        cm = confusion_matrix(gts, preds, normalize="true", labels=labels)
        cm = (cm * 100).astype(int)

        if L.cfg.IS_MASTER:
            print("\n", cm)

            L.writer.add_scalar("val/loss", avg_val_loss, epoch)
            L.writer.add_scalar("val/f1_micro", f1_mic, epoch)
            L.writer.add_scalar("val/f1_macro", f1_mac, epoch)
            L.writer.add_scalar("val/auc", auc, epoch)
            L.writer.add_scalar("val/iou", iou, epoch)

            elapsed = time.time() - start_time
            L.logger.info(
                f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
            )

            if iou > best_score:
                best_score = iou
                L.logger.info("saving model")
                model = L.model_ema.module if L.model_ema is not None else L.model

                if L.model_ema is not None:
                    model = L.model_ema.module
                elif L.cfg.PARALLEL.DDP:
                    model = L.model.module
                else:
                    model = L.model

                dstf = L.output_folder / "models" / f"S{L.cfg.split}_{epoch}_score_{iou:.4f}"
                save_model(model, L.tokenizer, dstf)
                clean_checkpoint(L.output_folder / "models", 3)


def calc_iou(preds_oh, gts):
    ious = []
    for i in range(0, 4):
        gt_cat = gts == i
        pred_cat = preds_oh == i
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        iou = intersection.sum() / union.sum()
        ious.append(iou)
    return torch.mean(torch.tensor(ious))
