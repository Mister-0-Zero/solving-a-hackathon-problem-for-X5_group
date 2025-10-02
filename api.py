# app.py
# FastAPI-сервис, воспроизводящий инференс из code.ipynb бит-в-бит.
# Зависимости: fastapi, uvicorn, torch, transformers, torchcrf, pydantic

from __future__ import annotations
import os, re
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF

# ========= Константы & словари (как в ноутбуке) =========
LABELS = [
    "O",
    "B-BRAND","I-BRAND",
    "B-TYPE","I-TYPE",
    "B-VOLUME","I-VOLUME",
    "B-PERCENT","I-PERCENT",
]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
ID2LABEL = {i:l for l,i in LABEL2ID.items()}
ENTITY_TAGS = {"BRAND","TYPE","VOLUME","PERCENT"}

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 96
BIAS_SCALE = 0.5

# === CharCNN === (как в ноуте)
USE_CHARCNN        = True
CHAR_MAXLEN        = 16
CHAR_EMB_DIM       = 24
CHAR_CHANNELS      = 64
CHAR_KERNEL_SIZES  = (2, 3, 4)
CHAR_SCALE         = 0.5

# === Алфавит для CharCNN (как в ноуте)
CYR_LO = "абвгдеёжзийклмнопрстуфхцчшщьыъэюя"
CYR_UP = CYR_LO.upper()
LAT_LO = "abcdefghijklmnopqrstuvwxyz"
LAT_UP = LAT_LO.upper()
DIGITS = "0123456789"
PUNCT  = "-–—._'/\\%+&()[],:;!?@"
CHAR_ALPHABET = ["<pad>", "<unk>"] + list(CYR_LO + CYR_UP + LAT_LO + LAT_UP + DIGITS + PUNCT)
CHAR2ID = {c:i for i,c in enumerate(CHAR_ALPHABET)}
PAD_CHAR, UNK_CHAR = 0, 1

# === Регексы/множества из блока инференса (cell с постпроцессингом)
STOP_BRIDGE = {"и","с","в","на","по","без","для","к","от","из","под","у","со","при"}
UNITS = {
    # из bias-функции и постпроцессинга (объединено)
    "л","литр","литра","литров",
    "мл",
    "г","гр","кг",
    "шт",
    "килограммов","грамм","граммов","миллилитров",
    "пач","бут","банка","бутылка","уп","упак"
}
NUM_RE = re.compile(r"^\d+[.,]?\d*$", re.I)
VOL_TOKEN_RE = re.compile(r"^\d+[.,]?\d*(?:мл|л|г|гр|кг|шт)$", re.I)
PCT_STANDALONE_RE = re.compile(r"^\d+[.,]?\d*%$", re.I)

PCT_WORDS = {"%","процент","проц","процентов"}
ASCII_RE = re.compile(r"^[A-Za-z]+$")

# ========= Вспомогательные: токенизация по пробелам =========
WS_RE = re.compile(r"\S+")
def ws_tokens_with_offsets(text: str) -> List[Tuple[str,int,int]]:
    # end — эксклюзивный индекс
    return [(m.group(0), m.start(), m.end()) for m in WS_RE.finditer(text or "")]

# ========= Bias-фичи (как в ноуте, cell 16) =========
def token_feature_bias(words: List[str]) -> List[Dict[str,float]]:
    feats: List[Dict[str, float]] = []
    for i,w in enumerate(words):
        wl = w.lower()
        f: Dict[str,float] = {}
        is_num   = bool(NUM_RE.match(wl))
        is_ascii = bool(ASCII_RE.match(w))
        prev = words[i-1].lower() if i-1>=0 else ""
        nxt  = words[i+1].lower() if i+1<len(words) else ""
        if is_ascii and wl not in UNITS:
            f["B-BRAND"] = f.get("B-BRAND", 0.0) + 0.25
        if is_num and (nxt in PCT_WORDS or prev in PCT_WORDS or "%" in nxt or "%" in prev):
            f["B-PERCENT"] = f.get("B-PERCENT", 0.0) + 0.35
            f["I-PERCENT"] = f.get("I-PERCENT", 0.0) + 0.15
        if is_num and (nxt in UNITS):
            f["B-VOLUME"] = f.get("B-VOLUME", 0.0) + 0.35
        if wl in UNITS and bool(NUM_RE.match(prev)):
            f["I-VOLUME"] = f.get("I-VOLUME", 0.0) + 0.20
        feats.append(f)
    return feats

# ========= postprocess_bio (ровно как в твоём фрагменте) =========
_PUNKT_EDGES = re.compile(r"^\W+|\W+$")
def _norm(w: str) -> str:
    return _PUNKT_EDGES.sub("", w.lower())

def postprocess_bio(words, tags, bias_lean_O=True):
    n = len(words)
    T = tags[:]  # копия

    def set_tag(i, lab):
        if 0 <= i < n:
            T[i] = lab


    # ---------- 1) Правила для PERCENT ----------
    for i, w in enumerate(words):
        wl = w.lower()
        if PCT_STANDALONE_RE.match(wl):
            set_tag(i, "B-PERCENT"); continue
        if NUM_RE.match(wl) and i+1 < n:
            nxt = words[i+1]
            if nxt == "%":
                set_tag(i, "B-PERCENT"); set_tag(i+1, "I-PERCENT"); continue
            if nxt.lower().startswith("процент"):
                set_tag(i, "B-PERCENT"); set_tag(i+1, "I-PERCENT"); continue

    # ---------- 2) Правила для VOLUME ----------
    for i, w in enumerate(words):
        wl = w.lower()
        if VOL_TOKEN_RE.match(wl):
            set_tag(i, "B-VOLUME"); continue
        if NUM_RE.match(wl) and i+1 < n and words[i+1].lower() in UNITS:
            set_tag(i, "B-VOLUME"); set_tag(i+1, "I-VOLUME"); continue
        if T[i].endswith("VOLUME"):
            m = re.match(r"^\d+[.,]?\d*([a-zа-я]+)$", wl, re.I)
            if m and m.group(1) not in UNITS:
                T[i] = "O"

    # ---------- 3) Мягкий наклон одиноких B-* к O ----------
    if bias_lean_O:
        i = 0
        while i < n:
            if T[i].startswith("B-") and (i+1 == n or not T[i+1].startswith("I-")):
                lw = _norm(words[i])
                if (len(lw) <= 2) or (lw in STOP_BRIDGE) or (not any(ch.isalpha() for ch in lw)):
                    T[i] = "O"
            i += 1

    return T

# ========= Модель (как в твоём фрагменте TransformerCRF) =========
class TransformerCRF(nn.Module):
    """
    XLM-R/Roberta → (опц.) CharCNN → Linear → CRF
    Наклон к 'O' ДО CRF на сомнительных позициях.
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        o_id: int,
        bias_scale: float = 1.0,
        use_charcnn: bool = False,
        char_vocab_size: int = 0,
        char_emb_dim: int = 24,
        char_channels: int = 64,
        char_kernels: tuple = (2, 3, 4),
        char_scale: float = 0.5,
        lean_to_O: bool = True,
        lean_tau: float = 0.25,
        lean_delta: float = 0.35,
        lean_in_train: bool = False,
        pad_char_id: int = 0,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.use_charcnn = bool(use_charcnn)
        self.pad_char_id = int(pad_char_id)
        if self.use_charcnn:
            self.char_emb   = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=self.pad_char_id)
            self.char_convs = nn.ModuleList([nn.Conv1d(char_emb_dim, char_channels, k, padding=k//2) for k in char_kernels])
            self.char_proj  = nn.Linear(char_channels * len(char_kernels), hidden)
            self.char_scale = float(char_scale)

        self.emissions = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.bias_scale = float(bias_scale)

        self.o_id = int(o_id)
        self.lean_to_O   = bool(lean_to_O)
        self.lean_tau    = float(lean_tau)
        self.lean_delta  = float(lean_delta)
        self.lean_in_train = bool(lean_in_train)

    # --- Char features aligned to tokens [B,T,L]
    def _char_features(self, char_ids: torch.Tensor) -> torch.Tensor:
        if char_ids is None:
            return None
        B, T, L = char_ids.shape
        x = self.char_emb(char_ids)                       # [B,T,L,E]
        x = x.view(B*T, L, x.size(-1)).transpose(1, 2)    # [B*T, E, L]
        convs = [torch.relu(conv(x)).amax(dim=2) for conv in self.char_convs]  # list of [B*T,C]
        feat = torch.cat(convs, dim=1).view(B, T, -1)     # [B,T,C*K]
        feat = self.char_proj(feat)                       # [B,T,H]
        mask = (char_ids.ne(self.pad_char_id).any(dim=-1)).unsqueeze(-1)  # [B,T,1]
        return feat * mask

    @torch.no_grad()
    def _uncertain_mask(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        top2 = logits.topk(2, dim=-1).values
        return (top2[..., 0] - top2[..., 1]) < tau


    def _apply_lean_to_O(self, logits: torch.Tensor, tok_mask: torch.Tensor) -> torch.Tensor:
        if not self.lean_to_O: return logits
        if self.training and not self.lean_in_train: return logits
        with torch.no_grad():
            uncertain = self._uncertain_mask(logits, self.lean_tau) & tok_mask
        bump = uncertain.to(logits.dtype) * self.lean_delta
        logits[..., self.o_id] = logits[..., self.o_id] + bump
        return logits

    def forward(self, input_ids, attention_mask, labels=None, feat_bias=None, char_ids=None):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B,T_full,H]
        if self.use_charcnn and char_ids is not None:
            h = h + self.char_scale * self._char_features(char_ids)
        logits = self.emissions(h)  # [B,T_full,C]
        if feat_bias is not None:
            logits = logits + self.bias_scale * feat_bias.to(logits.dtype)

        logits   = logits[:, 1:-1, :]                      # Roberta-подобные: выкидываем <s>, </s>
        tok_mask = attention_mask[:, 1:-1].to(torch.bool)

        logits = self._apply_lean_to_O(logits, tok_mask)

        if labels is not None:
            gold = labels[:, 1:-1]
            tag  = gold.clone()
            tag[gold == -100] = 0
            mask_gold = (gold != -100)
            nll = -self.crf(logits, tag.long(), mask=mask_gold, reduction='mean')
            return nll
        else:
            return self.crf.decode(logits, mask=tok_mask)

# ========= Инициализация =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok_src = os.path.join("model", "tokenizer") if os.path.isdir(os.path.join("model", "tokenizer")) else MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(tok_src)

model = TransformerCRF(
    model_name=MODEL_NAME,
    num_labels=len(LABELS),
    o_id=LABELS.index("O"),
    bias_scale=BIAS_SCALE,
    use_charcnn=USE_CHARCNN,
    char_vocab_size=len(CHAR2ID),
    char_emb_dim=CHAR_EMB_DIM,
    char_channels=CHAR_CHANNELS,
    char_kernels=CHAR_KERNEL_SIZES,
    char_scale=CHAR_SCALE,
    pad_char_id=PAD_CHAR,
    lean_to_O=True,
    lean_tau=0.25,
    lean_delta=0.35,
    lean_in_train=False
).to(device)

# Загрузка весов
weights_path = os.path.join("model", "model.pt")
state = torch.load(weights_path, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

# ========= Вспомогат.: char_ids под первые сабтокены слов =========
def build_char_ids_for_first_subtokens(text: str, enc) -> torch.Tensor:
    """
    Строим [1, T_full, L] char-ids: кладём символы исходного слова
    только на первый сабтокен каждого слова, остальное — PAD.
    """
    input_ids = enc["input_ids"]
    word_ids = enc.word_ids()
    T_full = input_ids.size(1)
    out = torch.full((1, T_full, CHAR_MAXLEN), PAD_CHAR, dtype=torch.long)

    words = text.split()  # разбиение как в ws_tokens_with_offsets
    seen = set()
    for tpos, wid in enumerate(word_ids):
        if wid is None or wid in seen: 
            continue
        seen.add(wid)
        word = words[wid] if wid < len(words) else ""
        chars = [CHAR2ID.get(ch, UNK_CHAR) for ch in word[:CHAR_MAXLEN]]
        out[0, tpos, :len(chars)] = torch.tensor(chars, dtype=torch.long)
    return out

# ========= Инференс одного текста =========
@torch.no_grad()
def predict_one(text: str):
    if not text:
        return []

    toks = ws_tokens_with_offsets(text)
    words = [t for t,_,_ in toks]

    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    word_ids = enc.word_ids()
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)


    # feat_bias — только на первые сабтокены
    feat_bias = torch.zeros((1, input_ids.size(1), len(LABELS)), dtype=torch.float32, device=device)
    feats = token_feature_bias(words)
    prev = None
    for pos, wid in enumerate(word_ids):
        if wid is None: 
            continue
        if wid != prev:
            vec = [0.0] * len(LABELS)
            for k, v in feats[wid].items():
                if k in LABEL2ID:
                    vec[LABEL2ID[k]] += v
            feat_bias[0, pos, :] = torch.tensor(vec, device=device)
        prev = wid

    # char_ids (по первым сабтокенам слов)
    char_ids = None
    if USE_CHARCNN:
        char_ids = build_char_ids_for_first_subtokens(text, enc).to(device)

    # CRF-декод по сабтокенам без спецтокенов
    paths = model(input_ids=input_ids, attention_mask=attn_mask, feat_bias=feat_bias, char_ids=char_ids)
    path = paths[0]  # список индексов меток длиной по числу сабтокенов без <s>,</s>

    # маска "первый сабтокен слова"
    first = []
    prev = None
    for wid in word_ids[1:-1]:  # без <s>, </s>
        if wid is None:
            first.append(False)
        elif wid != prev:
            first.append(True)
        else:
            first.append(False)
        prev = wid

    word_labels = [ID2LABEL[p] for p, m in zip(path, first) if m]
    word_labels = postprocess_bio(words, word_labels, bias_lean_O=True)

    # под ровно число слов
    if len(word_labels) < len(toks):
        word_labels += ["O"] * (len(toks) - len(word_labels))
    elif len(word_labels) > len(toks):
        word_labels = word_labels[:len(toks)]

    # one-span-per-word с эксклюзивным end
    spans = []
    for (tok, s, e), lab in zip(toks, word_labels):
        spans.append((s, e, lab))
    return spans

# ========= FastAPI =========
class PredictIn(BaseModel):
    input: str

class PredictOutItem(BaseModel):
    start_index: int
    end_index: int
    entity: str

app = FastAPI(title="NER Service (code.ipynb-compatible)")

@app.post("/api/predict", response_model=List[PredictOutItem])
def api_predict(req: PredictIn):
    text = (req.input or "").strip()
    if text == "":
        return []
    spans = predict_one(text)
    # Возвращаем ВСЕ слова, включая 'O' (как в сабмишне ноутбука)
    return [{"start_index": s, "end_index": e, "entity": lab} for (s, e, lab) in spans]

@app.get("/health")
def health():
    return {"status": "ok"}
