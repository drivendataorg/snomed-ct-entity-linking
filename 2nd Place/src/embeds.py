import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


class SepBERTEmbedder:
    name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"

    def __init__(self, cuda: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name)
        self.cuda = cuda
        if self.cuda:
            self.model = self.model.cuda()

    def __call__(self, sentences):
        bs = 256
        all_embs = []
        for i in torch.arange(0, len(sentences), bs):
            toks = self.tokenizer.batch_encode_plus(
                sentences[i : i + bs],
                padding="max_length",
                max_length=25,
                truncation=True,
                return_tensors="pt",
            )
            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.cuda()
            with torch.no_grad(), torch.cuda.amp.autocast():
                cls_rep = self.model(**toks_cuda)[0].mean(1)
                all_embs.append(cls_rep)
        all_embs = torch.cat(all_embs, 0)
        all_embs = F.normalize(all_embs, p=2, dim=1)
        return all_embs


def simplify(xb, same_count):
    start = 0
    sxb = []
    for i in same_count:
        sub = xb[start : start + i]
        sub = sub.mean(0)
        sxb.append(sub)
        start = start + i
    return torch.stack(sxb)


def get_embeds(embedder, sctid_syn: dict, cuda: bool = False):
    labels, same_count, embeds, batch = [], [], [], []
    for k, vv in tqdm(sctid_syn.items()):
        if cuda:
            vv = vv.copy()
        labels.append(k)
        batch.extend(vv)
        same_count.append(len(vv))
        if len(batch) >= 128:
            se = embedder(batch).cpu()
            se = simplify(se, same_count)
            embeds.append(se)
            batch = []
            same_count = []
    if batch:
        se = embedder(batch).cpu()
        se = simplify(se, same_count)
        embeds.append(se)
    xb = torch.vstack(embeds)
    if cuda:
        xb = xb.cuda()
    xb = F.normalize(xb, p=2, dim=1)
    return {"labels": labels, "embeds": xb}
