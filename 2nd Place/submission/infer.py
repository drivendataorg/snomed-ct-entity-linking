import re
from collections import defaultdict

import torch
import transformers
from tqdm.auto import tqdm


def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s.,:]", " ", text)
    return text


class InferDataset:
    def __init__(self, dfn, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        self.split_notes = []
        for i, r in dfn.iterrows():
            rawtext = r.text
            text = preprocess_text(rawtext)
            chunks = self.split_note_into_chunks(text)
            chunks = [[c[0], c[1], rawtext, r.note_id] for c in chunks]
            self.split_notes.extend(chunks)

    def split_note_into_chunks(self, text):
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        tokenized_text = encoded["input_ids"]

        chunks = []
        for i in range(0, len(tokenized_text), self.block_size):
            chunk = tokenized_text[i : i + self.block_size]
            chunk = self.tokenizer.build_inputs_with_special_tokens(chunk)
            offsets_chunk = offsets[i : i + self.block_size]
            start_offset = [0, 0]
            end_offset = [0, 0]
            offsets_chunk = [start_offset] + offsets_chunk + [end_offset]
            chunks.append([chunk, offsets_chunk])
        return chunks

    def __len__(self):
        return len(self.split_notes)

    def __getitem__(self, i):
        chunk, offsets, text, note_id = self.split_notes[i]
        return {
            "input_ids": torch.tensor(chunk, dtype=torch.long),
            "attention_mask": torch.ones(len(chunk), dtype=torch.long),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "note_id": note_id,
            "text": text,
        }


def predict_spans_ensemble(ensemble, ds, id2label):
    ensemble.eval()
    spans = defaultdict(list)
    texts = defaultdict(str)

    for item in tqdm(ds, desc="Predicting spans.."):
        with torch.no_grad(), torch.cuda.amp.autocast():
            note_id = item.pop("note_id")
            offsets = item.pop("offsets")
            text = item.pop("text")

            pred = ensemble.predict(item)

            texts[note_id] = text

            classes = torch.argmax(pred, 1).cpu()
            for o, c in zip(offsets[1:-1], classes[1:-1]):
                o = o.tolist()
                cc = id2label[c.item()]
                if cc != "none":
                    spans[note_id].append((o[0], o[1], cc))

    mentions = fix_spans(spans, texts)
    return mentions, texts


def match_BIO(c1, c2):
    if c1[0] == "B" and c2[0] == "I":
        if c1[2:] == c2[2:]:
            return True
    return False


def join_classes_bio(spans):
    res = []
    for span in spans:
        s, e, c = span
        if not res:
            res.append([s, e, c])
            continue
        sp, ep, cp = res[-1]
        diff = s - ep

        if (diff == 0 or diff == 1 or diff == 2) and match_BIO(cp, c):
            res[-1] = [sp, e, cp]
        else:
            res.append([s, e, c])
    return res


def fix_spans(notes_spans, texts):
    mentions = defaultdict(list)
    for note_id, text in texts.items():
        if note_id not in notes_spans:
            continue
        spans = notes_spans[note_id]
        spans = [s for s in spans if s[2] != "O"]
        spans = join_classes_bio(spans)
        spans = [[s, e, c[2:]] for s, e, c in spans]

        for span in spans:
            s, e, c = span
            t = text[s:e]
            s = s + len(t) - len(t.lstrip())
            e = s + len(t.strip())

            mentions[note_id].append([s, e, text[s:e], c, note_id])

    return mentions


class SapEmbedder:
    def __init__(self, model_name):
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model.cuda()
        self.model.eval()

    def get_embeds(self, sentences):
        bs = 256  # batch size during inference
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
            with torch.no_grad():
                cls_rep = self.model(**toks_cuda)[0].mean(
                    1
                )  # use mean pooling representation as the embedding
                all_embs.append(cls_rep)

        all_embs = torch.cat(all_embs, 0)
        all_embs = torch.nn.functional.normalize(all_embs, p=2, dim=1)
        return all_embs


class Matcher:
    def __init__(self, embedder, embedder_path, prefix: str):
        self.embedder = embedder
        embeds = torch.load(embedder_path / f"{prefix}_body.pth")
        self.body_embeds = embeds["embeds"].cuda()
        self.body_embeds = torch.nn.functional.normalize(self.body_embeds, p=2, dim=1)
        self.body_labels = embeds["labels"]
        self.body_labels = [int(i) for i in self.body_labels]

        embeds = torch.load(embedder_path / f"{prefix}_proc.pth")
        self.procedure_embeds = embeds["embeds"].cuda()
        self.procedure_embeds = torch.nn.functional.normalize(self.procedure_embeds, p=2, dim=1)
        self.procedure_labels = embeds["labels"]
        self.procedure_labels = [int(i) for i in self.procedure_labels]

        embeds = torch.load(embedder_path / f"{prefix}_find.pth")
        self.finding_embeds = embeds["embeds"].cuda()
        self.finding_embeds = torch.nn.functional.normalize(self.finding_embeds, p=2, dim=1)
        self.finding_labels = embeds["labels"]
        self.finding_labels = [int(i) for i in self.finding_labels]

        self.embeds = [self.body_embeds, self.procedure_embeds, self.finding_embeds]
        self.labels = [self.body_labels, self.procedure_labels, self.finding_labels]

    def match_all(self, mentions):
        embeds = torch.cat(self.embeds, 0)
        labels = self.body_labels + self.procedure_labels + self.finding_labels
        labels = [int(i) for i in labels]

        mentions = [m.copy() for m in mentions]
        q_texts = [t[2] for t in mentions]
        q_embeds = self.embedder.get_embeds(q_texts)

        similarity = q_embeds @ embeds.T
        all_scores, snomed_idxs = torch.topk(similarity, 1)
        best_mentions = []
        for m, idxs, scores in zip(mentions, snomed_idxs, all_scores):
            cid = int(labels[idxs[0]])
            score = scores[0]
            best_mentions.append(m + [cid, score.item()])
        return [best_mentions]

    def match(self, mentions, topk=10):
        # copy mentions
        mentions = [m.copy() for m in mentions]
        bodys = [m for m in mentions if m[3] == "body"]
        procs = [m for m in mentions if m[3] == "proc"]
        finds = [m for m in mentions if m[3] == "find"]

        qs = [bodys, procs, finds]

        res = []
        for i, q_mentions in enumerate(qs):
            if not q_mentions:
                continue
            q_texts = [t[2] for t in q_mentions]

            q_embeds = self.embedder.get_embeds(q_texts)

            d_embeds = self.embeds[i]
            d_labels = self.labels[i]

            similarity = q_embeds @ d_embeds.T
            all_scores, snomed_idxs = torch.topk(similarity, topk)

            best_mentions = []
            for m, idxs, scores in zip(q_mentions, snomed_idxs, all_scores):
                cids = [int(d_labels[idx.item()]) for idx in idxs]
                scores = scores.tolist()
                best_mentions.append(m + [cids, scores])

            res.append(best_mentions)
        return res
