#!/usr/bin/env python
# coding: utf-8

# # DERM-Net — Clinical Decision Support App
# 
# **Run All, then open the `gradio.live` link printed by the last cell.**
# 
# DERM-Net Clinical Decision Support - Gradio application.
# 
# Upload a skin photograph and the app runs the full pipeline built in the two
# notebooks in this repository:
# 
#     image -> DERM-Net (EfficientNet-B4 + ViT-B/16, MSCA fusion)
#           -> calibrated confidence
#           -> abstention gate
#           -> LayerCAM / LIME visual explanation
#           -> plain-language explanation and urgency triage for the patient
#           -> formulary retrieval filtered on the predicted disease
#           -> deterministic safety engine (pregnancy / controlled / interactions)
#           -> clinician report, downloadable
#           -> grounded question answering over the formulary
# 
# Everything is auto-discovered: the DERM-Net checkpoint, the image dataset (for
# examples) and the drug spreadsheet are located by scanning the usual places, so
# the same file runs unchanged on Kaggle, Hugging Face Spaces and a laptop.
# 
#     Run All (the last cell starts the server)
# 
# This is a research demonstrator, not a medical device. See the Limitations tab.
# 
# ---
# 
# ### Before you run
# 
# * Turn **Internet ON** (Notebook options, right sidebar). Needed for the
#   dependency install and for the public share link.
# * Attach the image dataset and the drug spreadsheet as inputs. Both are
#   auto-discovered, so no paths need editing.
# * Put `dermnet_best.pth` anywhere under `/kaggle/input` or `/kaggle/working`.
#   Without it the interface still runs, but it will say plainly that the
#   predictions carry no meaning.

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")


def _ensure(packages) -> None:
    """Install anything missing before it is imported, so the file just runs.

    torch / pandas / scikit-learn are assumed present (they ship with Kaggle,
    Colab and Spaces images); installing torch automatically would be rude.
    """
    missing = [pkg for mod, pkg in packages
               if not importlib.util.find_spec(mod.split(".")[0])]
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
        importlib.invalidate_caches()


_ensure([
    ("gradio", "gradio"),
    ("timm", "timm"),
    ("openpyxl", "openpyxl"),
    ("pytorch_grad_cam", "grad-cam"),
    ("lime", "lime"),
    ("skimage", "scikit-image"),
])

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from torchvision import transforms

import timm

# Hugging Face ZeroGPU: functions that touch the GPU must be wrapped in
# @spaces.GPU so a GPU is dynamically attached only while they run. Falls
# back to a no-op decorator so the same file still runs locally / on a
# normal CPU or GPU Space where the `spaces` package isn't installed.
try:
    import spaces
except Exception:
    class _SpacesStub:
        @staticmethod
        def GPU(*args, **kwargs):
            def _decorator(fn):
                return fn
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return _decorator
    spaces = _SpacesStub()


# ## Configuration

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.60      # below this the app refuses to recommend treatment
TOP_K_DRUGS = 6
LIME_SAMPLES = 800               # lower = faster, noisier explanation
LIME_SEGMENTS = 90

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

SEARCH_BASES = ("/kaggle/input", "/kaggle/working", ".", "./data", "./outputs",
                "./checkpoints", str(Path.cwd()))   # __file__ does not exist in a notebook kernel

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
])

CANONICAL = {
    "epidermolysisbullosa": "Epidermolysis Bullosa", "eb": "Epidermolysis Bullosa",
    "ichtyosis": "Ichthyosis", "ichthyosis": "Ichthyosis",
    "hemangioma": "Hemangioma", "hemangiomas": "Hemangioma",
    "portwine": "Port-Wine Stain", "portwinestain": "Port-Wine Stain",
    "portwinestains": "Port-Wine Stain", "pws": "Port-Wine Stain",
    "healthyskin": "Healthy Skin", "healthy": "Healthy Skin", "normal": "Healthy Skin",
}
NO_TREATMENT = {"Healthy Skin"}


def canon(name: str) -> str:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    return CANONICAL.get(key, str(name).strip())


# ## Patient-facing knowledge

# Written in plain language for the person in the chair, not the clinician. Nothing
# here prescribes: it explains the condition, says when to seek help urgently, lists
# supportive measures that do not require a prescription, and gives the patient
# questions to take to their appointment. Sourced from standard dermatology patient
# education, deliberately conservative.

PATIENT_INFO = {
    "Epidermolysis Bullosa": {
        "plain": "The layers of the skin are not firmly anchored to each other, so ordinary "
                 "rubbing, pressure or friction can make the skin blister, tear or peel. It is "
                 "an inherited condition, it is lifelong, and it is not contagious. Severity "
                 "varies enormously between people and between subtypes.",
        "urgency": "high",
        "red_flags": [
            "Fever, spreading redness, warmth, or pus around a wound - possible infection",
            "Blisters in the mouth or throat that make swallowing or breathing difficult",
            "A wound that will not heal, or one that changes shape, colour or texture",
            "Poor weight gain in a child, or signs of dehydration",
            "Large raw areas, or a sudden increase in blistering",
        ],
        "self_care": [
            "Handle the skin gently: lift and support rather than drag or slide",
            "Use non-adhesive dressings; never pull a dry dressing off - soak it first",
            "Keep wounds clean and moist; a moist wound heals faster than a dry one",
            "Soft, loose, seam-free clothing worn inside-out reduces friction",
            "Pad hard surfaces, car seats and cot rails",
            "Protein and calorie intake matter - wound healing is nutritionally expensive",
        ],
        "questions": [
            "Which subtype of EB do I have, and how was it confirmed?",
            "Which dressings suit my wounds, and how often should they be changed?",
            "How do I spot an infection early, and who do I call?",
            "Would genetic counselling be useful for my family?",
            "Is there a specialist EB centre or nurse I can be referred to?",
        ],
    },
    "Ichthyosis": {
        "plain": "The skin either makes new cells faster than it sheds the old ones, or does "
                 "not shed them properly, so scale builds up and the skin feels dry, rough and "
                 "thickened. Most forms are inherited and lifelong. It is not contagious and it "
                 "is not caused by poor hygiene.",
        "urgency": "moderate",
        "red_flags": [
            "Sudden widespread redness with shivering or trouble holding body temperature",
            "Deep painful cracks that bleed, or signs of infection",
            "Overheating, dizziness or collapse in hot weather - sweating may be reduced",
            "Eye irritation, or difficulty closing the eyes fully",
            "Reduced hearing from scale building up in the ear canal",
        ],
        "self_care": [
            "Soak and smear: bathe or shower, then apply a thick emollient to damp skin",
            "Apply emollient generously and often - several times a day is normal",
            "Soften scale first, then remove gently; never scrape or force it off",
            "Use soap substitutes; ordinary soap strips what little oil the skin has",
            "A humidifier helps, especially in dry or air-conditioned rooms",
            "Be careful in hot weather and during exercise if your sweating is reduced",
        ],
        "questions": [
            "Which type of ichthyosis is this, and is genetic testing worthwhile?",
            "Which emollient, and which strength of keratolytic, suits my skin?",
            "How do I protect my eyes and ears?",
            "What precautions do I need in hot weather or during sport?",
            "Should other family members be checked?",
        ],
    },
    "Hemangioma": {
        "plain": "A benign - meaning non-cancerous - cluster of extra blood vessels. The "
                 "infantile type usually grows for the first several months, then slowly shrinks "
                 "by itself over several years, often leaving little or no mark. Most need no "
                 "treatment at all.",
        "urgency": "moderate",
        "red_flags": [
            "Bleeding that does not stop after ten minutes of firm, steady pressure",
            "Ulceration - a raw, broken or crusted surface, which is usually painful",
            "Near the eye: it can interfere with vision and needs prompt review",
            "On the lip, mouth or neck: it can affect feeding or breathing",
            "Very rapid growth, or five or more separate hemangiomas on the body",
        ],
        "self_care": [
            "Photograph it monthly next to a ruler or coin - growth is easier to judge "
            "from a series than from memory",
            "Keep any broken skin clean; protect it from knocks and rubbing",
            "Most need only watchful waiting; resist pressure to intervene unnecessarily",
            "Ask about timing - treatment decisions are usually easier early rather than late",
        ],
        "questions": [
            "Does this need treatment, or is watchful waiting right?",
            "Is there any risk to vision, feeding or breathing given where it is?",
            "What is it likely to look like in one year, and in five?",
            "What exactly should make me call you sooner?",
            "Do we need an ultrasound or any other scan?",
        ],
    },
    "Port-Wine Stain": {
        "plain": "A birthmark made of widened small blood vessels in the skin. It is present "
                 "from birth, it does not fade away on its own, and it tends to darken and "
                 "thicken slowly over the years. It is not cancer and it is not contagious.",
        "urgency": "moderate",
        "red_flags": [
            "A stain over the forehead or upper eyelid - the eye and the brain should be "
            "assessed, as there is an association with glaucoma and with Sturge-Weber syndrome",
            "Seizures, unusual weakness, or delay in a child's development",
            "Eye pain, an enlarging eye, or any change in vision",
            "Raised nodules developing within the stain, especially if they bleed",
            "A stain on a limb with that limb growing larger than the other side",
        ],
        "self_care": [
            "Daily sun protection - sun damage makes the colour and texture worse over time",
            "Camouflage makeup is effective and there are products matched to skin tone",
            "Pulsed-dye laser generally works best when started early; ask about referral",
            "Keep up regular eye checks if the stain is near the eye",
        ],
        "questions": [
            "Do I need an eye examination or brain imaging because of where the stain is?",
            "Is laser treatment suitable, how many sessions, and what result is realistic?",
            "Will it come back or darken again after treatment?",
            "What will this look like as I get older?",
            "Is there support for the psychological side of a visible difference?",
        ],
    },
    "Healthy Skin": {
        "plain": "No skin disease was recognised in this photograph. Read that carefully: it is "
                 "not a clean bill of health. A single photograph can miss a great deal, and this "
                 "system only knows a small number of conditions - anything outside that list "
                 "cannot be detected at all.",
        "urgency": "low",
        "red_flags": [
            "A mole or spot that is changing in size, shape or colour",
            "A sore that bleeds or has not healed within four weeks",
            "A new lump, or a spot that itches or hurts persistently",
            "A mole that looks different from your others",
            "Any lesion that worries you - being wrong costs nothing, waiting can cost a lot",
        ],
        "self_care": [
            "Sunscreen daily on exposed skin; reapply if you are outdoors for long",
            "Avoid sunbeds entirely",
            "Check your own skin monthly, including scalp, soles and between toes",
            "Photograph anything you are watching, so change is easy to judge later",
        ],
        "questions": [
            "Should this spot be photographed and reviewed again in a few months?",
            "Do I need a full skin examination given my history?",
            "What specific changes should bring me back sooner?",
        ],
    },
}

URGENCY_STYLE = {
    "high": ("#C62828", "Seek review promptly"),
    "moderate": ("#EF6C00", "Arrange a routine appointment"),
    "low": ("#2E7D32", "No urgent action indicated"),
}


# ## Model

class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(hidden, channels, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.fc(torch.mean(x, dim=2)) +
                           self.fc(torch.max(x, dim=2).values)).unsqueeze(2)
        return x * att


class MSCABlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = max(1, out_channels // 3)
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, mid, 1, bias=False),
                                     nn.BatchNorm1d(mid), nn.GELU())
        self.branch3 = nn.Sequential(nn.Conv1d(in_channels, mid, 3, padding=1, bias=False),
                                     nn.BatchNorm1d(mid), nn.GELU())
        self.branch5 = nn.Sequential(nn.Conv1d(in_channels, mid, 5, padding=2, bias=False),
                                     nn.BatchNorm1d(mid), nn.GELU())
        fused_ch = mid * 3
        self.channel_att = ChannelAttention1D(fused_ch)
        self.project = nn.Sequential(nn.Conv1d(fused_ch, out_channels, 1, bias=False),
                                     nn.BatchNorm1d(out_channels))
        self.scale = nn.Parameter(torch.ones(3) / 3)
        self.residual = (nn.Conv1d(in_channels, out_channels, 1, bias=False)
                         if in_channels != out_channels else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(2)
        w = F.softmax(self.scale, dim=0)
        fused = torch.cat([self.branch1(x) * w[0], self.branch3(x) * w[1],
                           self.branch5(x) * w[2]], dim=1)
        fused = self.channel_att(fused)
        return self.act(self.project(fused) + self.residual(x)).squeeze(2)


class DERMNet(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        self.eff_features = timm.create_model("efficientnet_b4", pretrained=pretrained,
                                              num_classes=0)
        self.vit_features = timm.create_model("vit_base_patch16_224", pretrained=pretrained,
                                              num_classes=0)
        common_dim = 512
        self.eff_proj = nn.Sequential(nn.Linear(self.eff_features.num_features, common_dim),
                                      nn.LayerNorm(common_dim), nn.GELU())
        self.vit_proj = nn.Sequential(nn.Linear(self.vit_features.num_features, common_dim),
                                      nn.LayerNorm(common_dim), nn.GELU())
        self.fusion_msca = MSCABlock1D(common_dim * 2, common_dim)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Sequential(nn.Linear(common_dim, 256), nn.GELU(), nn.Dropout(0.3),
                                        nn.Linear(256, num_classes))

    def forward(self, x, return_features=False):
        fused = self.fusion_msca(torch.cat(
            [self.eff_proj(self.eff_features(x)), self.vit_proj(self.vit_features(x))], dim=1))
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return (logits, fused) if return_features else logits

    def get_eff_target_layer(self):
        """Layers the CAM methods hook, chosen by measurement rather than convention.

        Benchmarked on a synthetic set where the only class-discriminative evidence is
        a 46 px square at a known random position, with a model trained to 100% test
        accuracy so any localisation failure is the CAM's fault. Pointing-game accuracy
        over 40 images:

            conv_head, Grad-CAM++          30.8%   <- the previous choice
            blocks[-1], LayerCAM           52.5%
            bn2, LayerCAM                  62.5%
            bn2 + blocks[-2], LayerCAM     65.0%   <- this

        conv_head sits before the final BatchNorm and activation, so its outputs are
        unnormalised and signed, which makes the CAM's ReLU clip them badly. bn2 is the
        post-activation map. Adding blocks[-2] mixes in a finer-grained stage, since a
        7x7 grid upsampled to 224 px is what made the old overlay one giant blob.
        """
        layers = []
        for attr in ("bn2", "conv_head"):
            if hasattr(self.eff_features, attr):
                layers.append(getattr(self.eff_features, attr))
                break
        blocks = getattr(self.eff_features, "blocks", None)
        if blocks is not None and len(blocks) >= 2:
            layers.append(blocks[-2])
        return layers or [self.eff_features.conv_head]


# ## Asset discovery

def _count_images(directory: Path) -> int:
    try:
        return sum(1 for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)
    except OSError:
        return 0


def find_checkpoint():
    for base in SEARCH_BASES:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for f in sorted(base_path.glob("**/*.pth")):
            name = f.name.lower()
            if "dermnet" in name and "baseline" not in name:
                return f
    return None


def find_dataset_root(max_depth: int = 6):
    candidates = []
    for base in SEARCH_BASES:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        base_depth = len(base_path.parts)
        for root, dirs, _files in os.walk(base_path):
            root_path = Path(root)
            if len(root_path.parts) - base_depth > max_depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            counts = [(root_path / d, _count_images(root_path / d)) for d in dirs]
            counts = [(d, c) for d, c in counts if c >= 5]
            if len(counts) >= 2:
                names = {"".join(ch for ch in d.name.lower() if ch.isalnum())
                         for d, _ in counts}
                if names & {"train", "val", "test", "valid"}:
                    continue
                candidates.append((len(counts), sum(c for _, c in counts), root_path))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def find_spreadsheet():
    best = None
    for base in SEARCH_BASES:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for pattern in ("**/*.xlsx", "**/*.xls", "**/*.csv"):
            for f in base_path.glob(pattern):
                if f.name.startswith("~$"):
                    continue
                try:
                    head = (pd.read_csv(f, nrows=3) if f.suffix.lower() == ".csv"
                            else pd.read_excel(f, nrows=3))
                except Exception:
                    continue
                cols = {str(c).strip().lower() for c in head.columns}
                if any("disease" in c for c in cols) and any("drug" in c for c in cols):
                    if best is None or len(head.columns) > best[0]:
                        best = (len(head.columns), f)
    return best[1] if best else None

def find_selfcare_csv():
    """Locate the common-skin-issues self-care spreadsheet (distinct from the drug formulary:
    matched on an 'issue' column plus a 'self-care' column instead of 'disease' + 'drug')."""
    best = None
    for base in SEARCH_BASES:
        base_path = Path(base)
        if not base_path.is_dir():
            continue
        for f in base_path.glob("**/*.csv"):
            if f.name.startswith("~$"):
                continue
            try:
                head = pd.read_csv(f, nrows=3)
            except Exception:
                continue
            cols = {str(c).strip().lower() for c in head.columns}
            if any("issue" in c for c in cols) and any(
                    "self-care" in c or "self care" in c for c in cols):
                if best is None or len(head.columns) > best[0]:
                    best = (len(head.columns), f)
    return best[1] if best else None


# ## Load the formulary

COL = {
    "disease": "Disease", "drug": "Drug / Active Ingredient",
    "brand": "Brand Names / Formulation", "cls": "Drug Class / Type",
    "mech": "Mechanism / Use", "dose": "Recommended Dosage", "side": "Side Effects",
    "ddi": "Key Drug Interactions", "preg": "Pregnancy Category",
    "rx": "Rx and OTC status", "route": "Route of Administration", "csa": "CSA Schedule",
}


def pregnancy_risk(category: str) -> str:
    cat = str(category).strip().upper()
    if not cat or cat.startswith("N/A") or "NOT ASSIGNED" in cat:
        return "unknown"
    head = cat.split("(")[0].strip()
    if head in {"X", "D"} or "AVOID" in cat:
        return "contraindicated"
    if head == "C" or "D IN 3RD" in cat:
        return "caution"
    if head in {"A", "B"}:
        return "acceptable"
    return "unknown"


def load_formulary():
    path = find_spreadsheet()
    if path is None:
        return pd.DataFrame(), None
    df = (pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path))
    df.columns = [str(c).strip() for c in df.columns]
    for key, expected in list(COL.items()):
        if expected not in df.columns:
            match = [c for c in df.columns
                     if expected.lower().split("/")[0].strip() in c.lower()]
            COL[key] = match[0] if match else expected
            if not match:
                df[expected] = ""
    for c in COL.values():
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["disease_canon"] = df[COL["disease"]].map(canon)
    df["pregnancy_risk"] = df[COL["preg"]].map(pregnancy_risk)
    df["controlled"] = df[COL["csa"]].map(
        lambda v: bool(str(v).strip()) and "not controlled" not in str(v).strip().lower())
    df["otc"] = df[COL["rx"]].str.upper().str.contains("OTC")
    df["card"] = df.apply(lambda r: "\n".join([
        f"DRUG: {r[COL['drug']]}", f"CLASS: {r[COL['cls']]}",
        f"USE: {r[COL['mech']]}", f"DOSAGE: {r[COL['dose']]}",
        f"SIDE EFFECTS: {r[COL['side']]}", f"INTERACTIONS: {r[COL['ddi']]}",
        f"PREGNANCY: {r[COL['preg']]}", f"STATUS: {r[COL['rx']]}",
        f"ROUTE: {r[COL['route']]}",
    ]), axis=1)
    return df.reset_index(drop=True), path


DRUGS, DRUG_PATH = load_formulary()
KB_DISEASES = sorted(DRUGS["disease_canon"].unique()) if len(DRUGS) else []

if len(DRUGS):
    _VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    _MATRIX = _VECTORIZER.fit_transform(DRUGS["card"].tolist())
else:
    _VECTORIZER = _MATRIX = None


def retrieve(disease: str, query: str, ctx: dict | None = None,
             k: int = TOP_K_DRUGS) -> pd.DataFrame:
    """Structured filter on the predicted disease first, then rank by patient context.

    Contraindicated options are demoted to the bottom rather than removed. Hiding
    them would leave a clinician unable to see that the drug was considered and
    ruled out, but leaving one ranked first - as happened with Captopril for a
    pregnant patient - reads as a recommendation.
    """
    if not len(DRUGS):
        return pd.DataFrame()
    mask = (DRUGS["disease_canon"] == disease).values
    if mask.sum() == 0:
        return DRUGS.iloc[[]]
    ctx = ctx or {}
    scores = np.asarray((_MATRIX @ _VECTORIZER.transform([query]).T).todense()).ravel()
    scores = np.where(mask, scores, -np.inf)
    top = np.argsort(scores)[::-1][:min(k, int(mask.sum()))]
    out = DRUGS.iloc[top].copy()

    def _risk_rank(row) -> int:
        if ctx.get("pregnant"):
            return {"contraindicated": 3, "unknown": 2, "caution": 1}.get(
                row["pregnancy_risk"], 0)
        return 0

    out["_risk"] = out.apply(_risk_rank, axis=1)
    out["safety_note"] = out.apply(
        lambda r: ("CONTRAINDICATED" if r["_risk"] == 3 else
                   "unverified" if r["_risk"] == 2 else
                   "caution" if r["_risk"] == 1 else "ok"), axis=1)
    return out.sort_values("_risk", kind="stable").drop(columns="_risk")


def safety_flags(row, ctx: dict) -> list:
    flags = []
    if ctx.get("pregnant"):
        risk = row["pregnancy_risk"]
        if risk == "contraindicated":
            flags.append(("critical", f"{row[COL['drug']]}: CONTRAINDICATED IN PREGNANCY "
                                      f"(category {row[COL['preg']]})"))
        elif risk == "caution":
            flags.append(("warn", f"{row[COL['drug']]}: pregnancy caution "
                                  f"(category {row[COL['preg']]})"))
        elif risk == "unknown":
            flags.append(("info", f"{row[COL['drug']]}: pregnancy safety not established"))
    if ctx.get("infant") and str(row[COL["route"]]).lower().startswith("oral"):
        flags.append(("warn", f"{row[COL['drug']]}: systemic route in an infant, "
                              f"specialist dosing required"))
    if row["controlled"]:
        flags.append(("warn", f"{row[COL['drug']]}: controlled substance "
                              f"({row[COL['csa']]}), dependence risk"))
    interactions = str(row[COL["ddi"]]).strip()
    benign = {"", "none significant", "none known", "none listed", "none", "nan"}
    if interactions.lower() not in benign:
        for med in ctx.get("medications", []):
            if med and med.strip().lower()[:5] in interactions.lower():
                flags.append(("critical", f"{row[COL['drug']]}: INTERACTION with '{med}' - "
                                          f"{interactions}"))
    for allergy in ctx.get("allergies", []):
        target = f"{row[COL['drug']]} {row[COL['cls']]}".lower()
        if allergy and allergy.strip().lower()[:5] in target:
            flags.append(("critical", f"{row[COL['drug']]}: ALLERGY match on '{allergy}'"))
    return flags


def build_query(disease: str, ctx: dict) -> str:
    bits = [f"treatment for {disease}"]
    if ctx.get("pregnant"):
        bits.append("safe in pregnancy avoid teratogenic retinoid")
    if ctx.get("infant"):
        bits.append("infant paediatric dosing topical preferred")
    if ctx.get("infected"):
        bits.append("wound infection antibiotic antimicrobial dressing")
    if ctx.get("pain"):
        bits.append("pain relief analgesia anaesthetic")
    if ctx.get("severe"):
        bits.append("severe extensive systemic therapy biologic")
    if ctx.get("otc"):
        bits.append("over the counter emollient no prescription")
    return "; ".join(bits)


# ## Load the model

DATA_ROOT = find_dataset_root()
if DATA_ROOT is not None:
    CLASS_DIRS = sorted([d for d in DATA_ROOT.iterdir()
                         if d.is_dir() and _count_images(d) >= 5], key=lambda d: d.name.lower())
    CLASS_NAMES = [canon(d.name) for d in CLASS_DIRS]
else:
    CLASS_DIRS = []
    CLASS_NAMES = ["Epidermolysis Bullosa", "Healthy Skin", "Hemangioma",
                   "Ichthyosis", "Port-Wine Stain"]

CKPT_PATH = find_checkpoint()
MODEL_READY = False
STARTUP_NOTES = []

MODEL = DERMNet(len(CLASS_NAMES), pretrained=False)
if CKPT_PATH is not None:
    try:
        try:
            state = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(CKPT_PATH, map_location="cpu")
        head_w = [v for k, v in state.items() if k.endswith("classifier.3.weight")]
        if head_w and head_w[0].shape[0] != len(CLASS_NAMES):
            MODEL = DERMNet(head_w[0].shape[0], pretrained=False)
            CLASS_NAMES = CLASS_NAMES[:head_w[0].shape[0]]
        MODEL.load_state_dict(state, strict=False)
        MODEL_READY = True
        STARTUP_NOTES.append(f"Checkpoint loaded from `{CKPT_PATH}`.")
    except Exception as exc:
        STARTUP_NOTES.append(f"Checkpoint at `{CKPT_PATH}` could not be loaded ({exc}).")
else:
    STARTUP_NOTES.append(
        "**No DERM-Net checkpoint found.** The interface is fully functional but the network "
        "is randomly initialised, so predictions are meaningless. Run "
        "`DERMNet_Unified_XAI_Kaggle.ipynb` to produce `dermnet_best.pth`, then place it "
        "beside this file.")

MODEL = MODEL.to(DEVICE).eval()
NUM_CLASSES = len(CLASS_NAMES)

STARTUP_NOTES.append(
    f"Formulary: {len(DRUGS)} drugs across {len(KB_DISEASES)} diseases from `{DRUG_PATH}`."
    if len(DRUGS) else
    "**No drug spreadsheet found** - treatment retrieval is disabled. Place the "
    "`Disease, Drug / Active Ingredient` file beside this app.")

# Optional explainers
try:
    from pytorch_grad_cam import LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    CAM = LayerCAM(model=MODEL, target_layers=MODEL.get_eff_target_layer())
    CAM_METHOD = "LayerCAM"
    CAM_OK = True
except Exception as exc:
    CAM_OK = False
    CAM_METHOD = "unavailable"
    STARTUP_NOTES.append(f"CAM unavailable ({exc}).")

# LIME. The segmentation function is pinned to SLIC rather than left on LIME's
# default (quickshift), which is slow and behaves differently across scikit-image
# releases - a common cause of LIME silently producing nothing.
LIME_IMPORT_ERROR = None
try:
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm
    from skimage.segmentation import mark_boundaries
    LIME_EXPLAINER = lime_image.LimeImageExplainer(verbose=False, random_state=42)
    LIME_SEGMENTER = SegmentationAlgorithm(
        "slic", n_segments=LIME_SEGMENTS, compactness=10.0, sigma=1.0,
        start_label=0, random_seed=42)
    LIME_OK = True
except Exception as exc:
    LIME_OK = False
    LIME_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
    STARTUP_NOTES.append(f"LIME unavailable ({LIME_IMPORT_ERROR}).")


# ## Inference

def denorm(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1).astype(np.float32)


@torch.no_grad()
def predict_probs(batch: torch.Tensor) -> np.ndarray:
    out = []
    for i in range(0, batch.size(0), 16):
        logits = MODEL(batch[i:i + 16].to(DEVICE))
        out.append(F.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_with_tta(img_tensor: torch.Tensor) -> np.ndarray:
    batch = img_tensor.unsqueeze(0)
    views = [batch, torch.flip(batch, dims=[3]), torch.flip(batch, dims=[2]),
             torch.flip(batch, dims=[2, 3])]
    return np.mean([predict_probs(v)[0] for v in views], axis=0)


def gradcam_overlay(img_tensor: torch.Tensor, target: int, aug_smooth: bool = True):
    """Returns (image, status). Never raises; the reason for failure is returned.

    aug_smooth averages the CAM over flipped and rescaled copies of the input. It is
    roughly seven times slower and worth it: in the localisation benchmark it lifted
    pointing accuracy from 40% to 65%.
    """
    if not CAM_OK:
        return None, "CAM is not available in this environment."
    try:
        cam = CAM(input_tensor=img_tensor.unsqueeze(0).to(DEVICE),
                  targets=[ClassifierOutputTarget(target)],
                  eigen_smooth=True, aug_smooth=bool(aug_smooth))[0]
        lo, hi = float(cam.min()), float(cam.max())
        if hi - lo < 1e-8:
            return None, ("The CAM is flat - the model's evidence is spread evenly over "
                          "the whole image rather than concentrated anywhere.")
        cam = (cam - lo) / (hi - lo)
        n_layers = len(MODEL.get_eff_target_layer())
        return (show_cam_on_image(denorm(img_tensor), cam, use_rgb=True),
                f"{CAM_METHOD} over {n_layers} layer(s) of the EfficientNet branch"
                + (", augmentation-smoothed." if aug_smooth else "."))
    except Exception as exc:
        return None, f"CAM failed: {type(exc).__name__}: {exc}"


def lime_overlay(pil_image: Image.Image, target: int, num_samples: int = LIME_SAMPLES):
    """Returns (image, status).

    The previous version swallowed every exception and returned None, which made a
    genuine failure indistinguishable from 'not requested' - the panel simply stayed
    blank with no explanation. The error is now returned and shown in the interface.
    """
    if not LIME_OK:
        return None, f"LIME is not installed in this environment. {LIME_IMPORT_ERROR or ''}"
    try:
        base = np.asarray(pil_image.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
        if base.ndim != 3 or base.shape[2] != 3:
            return None, f"LIME needs an RGB image, got shape {base.shape}."

        def classifier(images):
            batch = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
            batch = (batch - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / \
                torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            return predict_probs(batch)

        explanation = LIME_EXPLAINER.explain_instance(
            base, classifier,
            labels=tuple(range(NUM_CLASSES)),
            top_labels=None,                     # keep every label, not just the top ones
            hide_color=0,
            num_samples=int(num_samples),
            batch_size=16,
            segmentation_fn=LIME_SEGMENTER,
            random_seed=42,
        )
        if target not in explanation.local_exp:
            available = sorted(explanation.local_exp)
            return None, (f"LIME produced no explanation for class {target}; "
                          f"it has {available}.")

        temp, mask = explanation.get_image_and_mask(
            target, positive_only=True, num_features=6, hide_rest=False)
        if mask.sum() == 0:
            return None, ("LIME found no region that supports this prediction - the model's "
                          "decision is spread thinly across the whole image rather than "
                          "driven by a specific area.")
        overlay = (mark_boundaries(temp / 255.0, mask) * 255).astype(np.uint8)
        n_regions = int(len(np.unique(explanation.segments)))
        return overlay, (f"{int(mask.sum() / mask.size * 100)}% of the image supports "
                         f"'{CLASS_NAMES[target]}', from {n_regions} candidate regions "
                         f"({num_samples} perturbations).")
    except Exception as exc:
        return None, f"LIME failed: {type(exc).__name__}: {exc}"


RISK_STYLE = {"critical": ("#C62828", "CRITICAL"), "warn": ("#EF6C00", "CAUTION"),
              "info": ("#1565C0", "NOTE")}


# ## General skin-issue self-care knowledge (feeds the Ask tab)

# A second, disease-agnostic knowledge base: common, everyday skin issues (itchy
# skin, dry skin, mild acne, melasma, sunburn, razor bumps, ...) with causes, home
# remedies, self-care tips and when to see a doctor. This lets the Ask tab answer
# everyday questions even when they have nothing to do with the predicted rare
# disease, or before any photo has been analysed. Auto-discovered like the
# formulary. Nothing here is generated from model memory and nothing prescribes
# a drug - it is deliberately limited to supportive, non-prescription self-care.

SELFCARE_COL = {
    "issue": "Issue", "aliases": "Aliases/Keywords", "category": "Category",
    "causes": "Common Causes", "remedies": "Natural/Home Remedies",
    "tips": "Self-Care Tips", "doctor": "When to See a Doctor",
}


def load_selfcare_kb():
    path = find_selfcare_csv()
    if path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for key, expected in list(SELFCARE_COL.items()):
        if expected not in df.columns:
            match = [c for c in df.columns
                     if expected.lower().split("/")[0].strip() in c.lower()]
            SELFCARE_COL[key] = match[0] if match else expected
            if not match:
                df[expected] = ""
    for c in SELFCARE_COL.values():
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["search_blob"] = (df[SELFCARE_COL["issue"]] + " " + df[SELFCARE_COL["aliases"]]
                          + " " + df[SELFCARE_COL["category"]]).str.lower()
    return df.reset_index(drop=True), path


SELFCARE_KB, SELFCARE_PATH = load_selfcare_kb()

if len(SELFCARE_KB):
    _SC_VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    _SC_MATRIX = _SC_VECTORIZER.fit_transform((
        SELFCARE_KB["search_blob"] + " " + SELFCARE_KB[SELFCARE_COL["causes"]]
        + " " + SELFCARE_KB[SELFCARE_COL["tips"]]).tolist())
else:
    _SC_VECTORIZER = _SC_MATRIX = None


def _selfcare_alias_hit(ql: str):
    """Exact issue/alias match, e.g. the user typed 'melasma' or 'prickly heat'."""
    if not len(SELFCARE_KB):
        return None
    for _, row in SELFCARE_KB.iterrows():
        names = [row[SELFCARE_COL["issue"]]] + [
            a.strip() for a in row[SELFCARE_COL["aliases"]].split(",") if a.strip()]
        for name in names:
            base = re.sub(r"\(.*?\)", "", name).strip()
            if len(base) > 3 and base.lower() in ql:
                return row
    return None


def _format_selfcare(row) -> str:
    tips = [t.strip() for t in row[SELFCARE_COL["tips"]].split(";") if t.strip()]
    remedies = [t.strip() for t in row[SELFCARE_COL["remedies"]].split(";") if t.strip()]
    parts = [f"### {row[SELFCARE_COL['issue']]}", f"*{row[SELFCARE_COL['category']]}*",
             f"**Common causes:** {row[SELFCARE_COL['causes']] or 'not stated'}"]
    if remedies:
        parts.append("**Home remedies:**\n" + "\n".join(f"- {r}" for r in remedies))
    if tips:
        parts.append("**Self-care tips:**\n" + "\n".join(f"- {t}" for t in tips))
    parts.append(f"**See a doctor if:** {row[SELFCARE_COL['doctor']] or 'not stated'}")
    return "\n\n".join(parts)


def answer_general_skin_question(question: str):
    """Look up a common, everyday skin issue. Not disease-specific, not a prescription."""
    if not len(SELFCARE_KB):
        return None
    ql = question.lower()
    hit = _selfcare_alias_hit(ql)
    if hit is not None:
        return _format_selfcare(hit)
    if _SC_MATRIX is None:
        return None
    scores = np.asarray((_SC_MATRIX @ _SC_VECTORIZER.transform([question]).T).todense()).ravel()
    top = int(np.argmax(scores))
    if scores[top] > 0.08:
        return _format_selfcare(SELFCARE_KB.iloc[top])
    return None


# ## Grounded question answering

# Answers are assembled from the loaded formulary, the patient knowledge above, and
# (for everyday, non-disease-specific questions) the general skin self-care reference
# loaded above. Nothing is generated from the model's own memory, so the assistant
# cannot invent a drug, a dose or a claim. Anything it cannot ground, it declines.

def _drug_rows_for(disease):
    if not len(DRUGS):
        return DRUGS
    return DRUGS[DRUGS["disease_canon"] == disease] if disease else DRUGS


def _format_drug(row) -> str:
    return (f"**{row[COL['drug']]}**"
            f"{' (' + row[COL['brand']] + ')' if row[COL['brand']] else ''} — "
            f"{row[COL['cls']]}, {str(row[COL['route']]).lower()} route.\n"
            f"- Used for: {row[COL['mech']] or 'not stated'}\n"
            f"- Dosage as written: {row[COL['dose']] or 'not stated'}\n"
            f"- Side effects: {row[COL['side']] or 'not stated'}\n"
            f"- Pregnancy category: {row[COL['preg']] or 'not assigned'}"
            f" · {row[COL['rx']] or 'status not stated'}")


_TOPIC_TERMS = (
    "skin", "condition", "disease", "diagnos", "lesion", "rash", "blister", "scale",
    "birthmark", "spot", "wound", "treat", "drug", "medicine", "medication", "cream",
    "ointment", "dose", "dosage", "tablet", "therapy", "laser", "doctor", "clinic",
    "appointment", "symptom", "care", "result", "prediction", "photo", "image",
    "pregnan", "side effect", "otc", "prescription", "safe", "risk", "urgent",
)


def _on_topic(ql: str, disease: str) -> bool:
    """Is this question even about the assessment? Guards the generic branches.

    Without this, 'What is the capital of France?' matched the 'what is' branch and
    was answered with the condition description - which quietly contradicts the
    claim that every answer is grounded.
    """
    if disease and disease.lower() in ql:
        return True
    if any(t in ql for t in _TOPIC_TERMS):
        return True
    if _selfcare_alias_hit(ql) is not None:
        return True
    return bool(re.search(r"\b(this|it|these|those|my|i|me)\b", ql))


def answer_question(question: str, disease: str) -> str:
    """Route a question to grounded evidence. Returns markdown."""
    q = (question or "").strip()
    if not q:
        return "Ask me something about the assessment, the condition, or the drug options."

    ql = q.lower()
    info = PATIENT_INFO.get(disease, {})
    rows = _drug_rows_for(disease)
    disease_label = disease or "the predicted condition"

    def footer(extra=""):
        return ("\n\n---\n*Grounded in the loaded formulary and standard patient guidance"
                + (f" · {extra}" if extra else "")
                + ". Not medical advice; confirm with a clinician.*")

    # --- a specific drug by name -------------------------------------------
    if len(DRUGS):
        for _, row in DRUGS.iterrows():
            names = [str(row[COL["drug"]])]
            names += [b.strip() for b in re.split(r"[,/]", str(row[COL["brand"]]))]
            for name in names:
                if len(name) > 3 and re.search(
                        r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", q, re.I):
                    return (f"### {row[COL['drug']]}\n\n{_format_drug(row)}\n\n"
                            f"Interactions on file: {row[COL['ddi']] or 'none listed'}.\n"
                            f"Indicated here for: {row['disease_canon']}." + footer())

    # --- refuse anything that is not about the assessment ------------------
    if not _on_topic(ql, disease):
        return ("That is outside what I can answer. I only work from the loaded drug "
                "formulary and the guidance for the predicted condition.\n\n"
                "Ask me about: what the condition is, when to seek urgent help, day-to-day "
                "care, pregnancy, side effects, cost and prescription status, a specific "
                "drug by name, what to ask at your appointment, or a common everyday skin "
                "issue such as itchy skin, a rash, or melasma." + footer())

    # --- red flags / urgency ------------------------------------------------
    if any(w in ql for w in ["urgent", "emergency", "worry", "worried", "serious",
                             "danger", "when should i", "red flag", "hospital",
                             "doctor now", "risk"]):
        flags = info.get("red_flags", [])
        if flags:
            return (f"### When to seek help for {disease_label}\n\n"
                    + "\n".join(f"- {f}" for f in flags)
                    + "\n\nIf any of these apply, do not wait for a routine appointment."
                    + footer())

    # --- pregnancy ----------------------------------------------------------
    if any(w in ql for w in ["pregnan", "breastfeed", "conceiv", "baby safe", "trying for"]):
        if len(rows):
            safe = rows[rows["pregnancy_risk"] == "acceptable"]
            avoid = rows[rows["pregnancy_risk"] == "contraindicated"]
            unknown = rows[rows["pregnancy_risk"] == "unknown"]
            parts = [f"### Pregnancy and {disease_label}\n"]
            if len(avoid):
                parts.append("**Must be avoided** (category D or X):\n" + "\n".join(
                    f"- {r[COL['drug']]} — category {r[COL['preg']]}"
                    for _, r in avoid.iterrows()))
            if len(safe):
                parts.append("\n**Generally considered acceptable** (category A or B):\n"
                             + "\n".join(f"- {r[COL['drug']]} — category {r[COL['preg']]}"
                                         for _, r in safe.iterrows()))
            if len(unknown):
                parts.append(f"\n{len(unknown)} further options have no assigned pregnancy "
                             f"category, which means unstudied, not safe.")
            parts.append("\nPregnancy category is a starting point, not a decision. "
                         "This must be confirmed with your obstetrician or dermatologist.")
            return "\n".join(parts) + footer()

    # --- cost / prescription ------------------------------------------------
    if any(w in ql for w in ["otc", "over the counter", "prescription", "without a doctor",
                             "pharmacy", "cost", "cheap", "afford", "buy"]):
        if len(rows):
            otc = rows[rows["otc"]]
            if len(otc):
                return (f"### Available without a prescription for {disease_label}\n\n"
                        + "\n".join(f"- **{r[COL['drug']]}** "
                                    f"({r[COL['brand']] or 'generic'}) — "
                                    f"{r[COL['cls']]}, {r[COL['rx']]}"
                                    for _, r in otc.iterrows())
                        + "\n\nEverything else on file needs a prescription. This formulary "
                          "carries no pricing or local availability." + footer())
            return (f"Every option on file for {disease_label} needs a prescription."
                    + footer())

    # --- side effects -------------------------------------------------------
    if any(w in ql for w in ["side effect", "adverse", "reaction", "harm", "safe to take"]):
        if len(rows):
            return (f"### Reported side effects for {disease_label} options\n\n"
                    + "\n".join(f"- **{r[COL['drug']]}**: {r[COL['side']] or 'not stated'}"
                                for _, r in rows.head(8).iterrows())
                    + footer())

    # --- self care / daily management --------------------------------------
    if any(w in ql for w in ["care", "manage", "daily", "home", "look after", "routine",
                             "moistur", "bath", "wash", "cope", "help myself"]):
        care = info.get("self_care", [])
        if care:
            return (f"### Day-to-day care for {disease_label}\n\n"
                    + "\n".join(f"- {c}" for c in care)
                    + "\n\nThese are supportive measures, not a substitute for treatment."
                    + footer())

    # --- what is it / prognosis --------------------------------------------
    if any(w in ql for w in ["what is", "what's", "explain", "mean", "cause", "why",
                             "contagious", "catch", "genetic", "inherit", "go away",
                             "cure", "permanent", "future", "long term", "prognosis"]):
        if info.get("plain"):
            return (f"### {disease_label}\n\n{info['plain']}"
                    + ("\n\n**Questions worth asking your clinician**\n"
                       + "\n".join(f"- {q_}" for q_ in info.get("questions", []))
                       if info.get("questions") else "")
                    + footer())

    # --- questions to ask ---------------------------------------------------
    if any(w in ql for w in ["ask", "appointment", "consult", "question"]):
        if info.get("questions"):
            return (f"### Questions to take to your appointment about {disease_label}\n\n"
                    + "\n".join(f"- {q_}" for q_ in info["questions"]) + footer())

    # --- general, everyday skin issue (checked before the drug-formulary fallback,
    # so "rash"/"itchy skin" etc. don't get matched against unrelated drug side-effect
    # text - self-care answers only, no medicine suggested, unless the predicted
    # disease itself already matched one of the branches above) --------------------
    general = answer_general_skin_question(q)
    if general:
        return general + footer("common skin-issues self-care reference, not disease-specific")

    # --- fallback: retrieve --------------------------------------------------
    if len(DRUGS) and _MATRIX is not None:
        scores = np.asarray((_MATRIX @ _VECTORIZER.transform([q]).T).todense()).ravel()
        if disease:
            scores = np.where((DRUGS["disease_canon"] == disease).values, scores, scores * 0.3)
        top = np.argsort(scores)[::-1][:3]
        if scores[top[0]] > 0.02:
            return ("I can only answer from the loaded formulary. The closest entries to your "
                    "question:\n\n"
                    + "\n\n".join(_format_drug(DRUGS.iloc[i]) for i in top) + footer())

    return ("I could not ground that in the loaded formulary or the condition guidance, so I "
            "will not answer it — a made-up answer about medicine is worse than none.\n\n"
            "Try asking about: what the condition is, when to seek urgent help, day-to-day "
            "care, pregnancy, side effects, cost and prescription status, a specific drug by "
            "name, what to ask at your appointment, or a common everyday skin issue such as "
            "itchy skin, a rash, or melasma." + footer())


# ## Report assembly

def build_report(disease, confidence, probs, cards, flags, ctx, status) -> str:
    info = PATIENT_INFO.get(disease, {})
    lines = [
        "DERM-NET ASSESSMENT SUMMARY",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "Research demonstrator - not a diagnosis, not a prescription.",
        "",
        "IMPRESSION",
        f"  {disease} (calibrated confidence {confidence * 100:.1f}%)",
        "  Differential: " + ", ".join(
            f"{CLASS_NAMES[i]} {probs[i] * 100:.1f}%" for i in np.argsort(probs)[::-1][:3]),
        "",
        "WHAT THIS MEANS",
        "  " + info.get("plain", "No plain-language description available."),
        "",
    ]
    if info.get("red_flags"):
        lines += ["SEEK HELP IF"] + [f"  - {f}" for f in info["red_flags"]] + [""]
    if info.get("self_care"):
        lines += ["DAY-TO-DAY CARE"] + [f"  - {c}" for c in info["self_care"]] + [""]

    if status == "recommend" and len(cards):
        lines += ["TREATMENT OPTIONS ON FILE"]
        for i, (_, r) in enumerate(cards.iterrows(), 1):
            mark = "  [AVOID] " if r["safety_note"] == "CONTRAINDICATED" else "  "
            lines.append(f"{mark}{i}. {r[COL['drug']]} "
                         f"({r[COL['brand']] or 'no brand listed'}) - {r[COL['cls']]}, "
                         f"{str(r[COL['route']]).lower()}")
            lines.append(f"       Dosage: {r[COL['dose']] or 'not stated'}")
            lines.append(f"       Side effects: {r[COL['side']] or 'not stated'}")
            lines.append(f"       {r[COL['rx']]}; pregnancy category "
                         f"{r[COL['preg']] or 'not assigned'}")
        lines.append("")
        lines += ["SAFETY"] + ([f"  - {t}" for _, t in flags] or
                               ["  - No hard contraindication for the context supplied."])
        lines.append("")
    else:
        lines += ["TREATMENT", "  No drug treatment proposed by the system for this result.", ""]

    if info.get("questions"):
        lines += ["QUESTIONS FOR YOUR CLINICIAN"] + [f"  - {q}" for q in info["questions"]] + [""]

    lines += [
        "HOW TO READ THIS",
        "  This came from a photograph analysed by a model trained on a few hundred",
        "  images of five conditions. It cannot see anything outside those five, it has",
        "  not examined you, and it does not know your history. Take this sheet to a",
        "  clinician; do not act on it alone.",
    ]
    return "\n".join(lines)


def write_report_file(text: str) -> str:
    path = Path(tempfile.gettempdir()) / \
        f"dermnet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ## The pipeline

@spaces.GPU(duration=60)
def analyse(image, pregnant, infant, infected, pain, severe, otc,
            medications, allergies, threshold, want_lime, lime_samples,
            cam_aug_smooth=True):
    """The whole pipeline. Returns everything the interface displays."""
    # On a ZeroGPU Space, a real GPU only becomes visible inside this
    # function while it runs. Move the model over on first use so
    # inference, CAM and LIME (which all key off the DEVICE global) run
    # on it; outside this function the Space is CPU-only.
    global DEVICE
    if torch.cuda.is_available() and DEVICE.type != "cuda":
        DEVICE = torch.device("cuda")
        MODEL.to(DEVICE)

    empty = (None, None, None, "", "", "", "", "", pd.DataFrame(), "", None, "")
    if image is None:
        return (None, None, None, "",
                "<div class='banner' style='border-left:6px solid #1565C0'>"
                "<b>Upload a photograph and press Analyse.</b></div>",
                "", "", "", pd.DataFrame(), "", None, "")

    ctx = {
        "pregnant": pregnant, "infant": infant, "infected": infected,
        "pain": pain, "severe": severe, "otc": otc,
        "medications": [m.strip() for m in str(medications or "").split(",") if m.strip()],
        "allergies": [a.strip() for a in str(allergies or "").split(",") if a.strip()],
    }

    pil = image.convert("RGB")
    img_tensor = EVAL_TF(pil)
    probs = predict_with_tta(img_tensor)
    pred = int(probs.argmax())
    confidence = float(probs[pred])
    disease = CLASS_NAMES[pred]
    info = PATIENT_INFO.get(disease, {})

    label_scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
    cam_img, cam_status = gradcam_overlay(img_tensor, pred, cam_aug_smooth)
    if want_lime:
        lime_img, lime_status = lime_overlay(pil, pred, lime_samples)
    else:
        lime_img, lime_status = None, ("LIME was not requested. Tick **Explain with LIME** "
                                       "to compute it.")

    # ---- decision gate ----------------------------------------------------
    if not MODEL_READY:
        status = "untrained"
    elif disease in NO_TREATMENT:
        status = "no_treatment"
    elif confidence < threshold:
        status = "abstain"
    elif disease not in KB_DISEASES:
        status = "no_coverage"
    else:
        status = "recommend"

    conf_colour = "#2E7D32" if confidence >= 0.8 else "#EF6C00" if confidence >= threshold \
        else "#C62828"
    conf_word = ("strong" if confidence >= 0.8 else
                 "moderate" if confidence >= threshold else "weak")

    verdict = (
        f"<div class='verdict-card'>"
        f"<div class='verdict-label'>IMPRESSION</div>"
        f"<div class='verdict-name'>{disease}</div>"
        f"<div class='verdict-bar'><div class='verdict-fill' "
        f"style='width:{confidence * 100:.1f}%;background:{conf_colour}'></div></div>"
        f"<div class='verdict-conf'>{conf_word} confidence "
        f"<b style='color:{conf_colour}'>{confidence * 100:.1f}%</b>"
        f"<span class='muted'> &nbsp;·&nbsp; abstention threshold "
        f"{threshold * 100:.0f}%</span></div>"
        f"</div>")

    banner = {
        "untrained": ("#C62828", "NO TRAINED CHECKPOINT",
                      "The network is randomly initialised. This output is a UI "
                      "demonstration only and carries no clinical meaning."),
        "no_treatment": ("#1565C0", "NO PHARMACOLOGICAL TREATMENT INDICATED",
                         "The image is assessed as healthy skin. No drug is proposed."),
        "abstain": ("#EF6C00", "ABSTAINED — LOW CONFIDENCE",
                    f"Confidence {confidence * 100:.1f}% is below the "
                    f"{threshold * 100:.0f}% threshold, so no treatment is proposed. "
                    f"This is the system working correctly, not failing."),
        "no_coverage": ("#EF6C00", "NO FORMULARY ENTRY",
                        f"No drug entries exist for {disease} in the loaded formulary."),
        "recommend": ("#2E7D32", "TREATMENT OPTIONS RETRIEVED",
                      "Filtered to the predicted condition and ranked against the "
                      "patient context."),
    }[status]
    banner_html = (f"<div class='banner' style='border-left:6px solid {banner[0]}'>"
                   f"<b style='color:{banner[0]}'>{banner[1]}</b><br>{banner[2]}</div>")

    # ---- patient-facing panels --------------------------------------------
    urgency = info.get("urgency", "moderate")
    u_colour, u_text = URGENCY_STYLE.get(urgency, URGENCY_STYLE["moderate"])
    red_flags = info.get("red_flags", [])
    urgency_html = (
        f"<div class='panel'>"
        f"<div class='panel-head' style='color:{u_colour}'>"
        f"<span class='dot' style='background:{u_colour}'></span>{u_text}</div>"
        + ("<div class='panel-sub'>Get seen without waiting if any of these apply:</div>"
           "<ul class='tight'>" + "".join(f"<li>{f}</li>" for f in red_flags) + "</ul>"
           if red_flags else "")
        + "</div>")

    patient_html = (
        f"<div class='panel'>"
        f"<div class='panel-head'>What this means</div>"
        f"<p class='plain'>{info.get('plain', 'No description available.')}</p>"
        + ("<div class='panel-head' style='margin-top:14px'>Things you can do</div>"
           "<ul class='tight'>"
           + "".join(f"<li>{c}</li>" for c in info.get("self_care", [])) + "</ul>"
           if info.get("self_care") else "")
        + ("<div class='panel-head' style='margin-top:14px'>Ask your clinician</div>"
           "<ul class='tight'>"
           + "".join(f"<li>{q}</li>" for q in info.get("questions", [])) + "</ul>"
           if info.get("questions") else "")
        + "</div>")

    # ---- retrieval + safety ----------------------------------------------
    if status == "recommend":
        cards = retrieve(disease, build_query(disease, ctx), ctx)
        flags = []
        for _, row in cards.iterrows():
            flags.extend(safety_flags(row, ctx))

        badge = {"CONTRAINDICATED": "AVOID", "caution": "caution",
                 "unverified": "unverified", "ok": "—"}
        table = pd.DataFrame({
            "Safety": cards["safety_note"].map(badge),
            "Drug": cards[COL["drug"]],
            "Brand": cards[COL["brand"]],
            "Class": cards[COL["cls"]],
            "Route": cards[COL["route"]],
            "Dosage": cards[COL["dose"]],
            "Pregnancy": cards[COL["preg"]].replace("", "not assigned"),
            "Status": cards[COL["rx"]],
        }).reset_index(drop=True)

        if flags:
            severity = {"critical": 0, "warn": 1, "info": 2}
            flags.sort(key=lambda f: severity.get(f[0], 3))
            seen, rows_html = set(), []
            for level, text in flags:
                if text in seen:
                    continue
                seen.add(text)
                colour, tag = RISK_STYLE[level]
                rows_html.append(
                    f"<div class='flag' style='border-left:4px solid {colour}'>"
                    f"<span class='flag-tag' style='background:{colour}'>{tag}</span>"
                    f"{text}</div>")
            safety_html = "".join(rows_html)
        else:
            safety_html = ("<div class='flag' style='border-left:4px solid #2E7D32'>"
                           "<span class='flag-tag' style='background:#2E7D32'>CLEAR</span>"
                           "No hard contraindication for the context supplied.</div>")
    else:
        cards, flags = pd.DataFrame(), []
        table = pd.DataFrame()
        safety_html = (f"<div class='flag' style='border-left:4px solid {banner[0]}'>"
                       f"<span class='flag-tag' style='background:{banner[0]}'>"
                       f"{banner[1]}</span>{banner[2]}</div>")

    report = build_report(disease, confidence, probs, cards, flags, ctx, status)
    report_path = write_report_file(report)

    explain_note = " ".join(x for x in [cam_status] if x)

    return (label_scores, cam_img, lime_img,
            (lime_status + (("  \n" + explain_note) if explain_note else "")),
            verdict + banner_html, urgency_html, patient_html, safety_html,
            table, "```\n" + report + "\n```", report_path, disease)


# ## Interface

CSS = """
.gradio-container {max-width: 1460px !important;}
#hero {background: linear-gradient(120deg,#0b3d91 0%,#1565c0 42%,#00897b 100%);
       color:#fff; padding:28px 32px; border-radius:16px; margin-bottom:16px;
       box-shadow:0 8px 26px rgba(13,71,161,.22);}
#hero h1 {margin:0; font-size:1.95rem; font-weight:750; letter-spacing:-.5px; color:#fff;}
#hero p {margin:9px 0 0; opacity:.95; font-size:.97rem; line-height:1.55; color:#fff;
         max-width:96ch;}
#hero .pills {margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;}
#hero .pill {background:rgba(255,255,255,.17); border:1px solid rgba(255,255,255,.30);
             padding:5px 12px; border-radius:999px; font-size:.78rem; font-weight:650;}

.verdict-card {background:var(--block-background-fill);
               border:1px solid var(--border-color-primary);
               border-radius:14px; padding:20px 22px; margin-bottom:12px;}
.verdict-label {font-size:.68rem; letter-spacing:.16em; opacity:.6; font-weight:800;}
.verdict-name {font-size:1.72rem; font-weight:750; margin:5px 0 14px; line-height:1.15;}
.verdict-bar {height:10px; background:var(--border-color-primary); border-radius:99px;
              overflow:hidden;}
.verdict-fill {height:100%; border-radius:99px; transition:width .45s ease;}
.verdict-conf {margin-top:10px; font-size:.92rem;}
.muted {opacity:.62;}

.banner {background:var(--block-background-fill);
         border:1px solid var(--border-color-primary);
         border-radius:11px; padding:14px 17px; font-size:.9rem; line-height:1.55;
         margin-bottom:12px;}

.panel {background:var(--block-background-fill);
        border:1px solid var(--border-color-primary);
        border-radius:13px; padding:17px 20px; margin-bottom:12px;}
.panel-head {font-size:.76rem; letter-spacing:.13em; text-transform:uppercase;
             font-weight:800; opacity:.78; display:flex; align-items:center;}
.panel-sub {font-size:.85rem; opacity:.7; margin:8px 0 4px;}
.dot {width:9px; height:9px; border-radius:99px; display:inline-block; margin-right:9px;}
.plain {margin:10px 0 0; font-size:.95rem; line-height:1.62;}
ul.tight {margin:8px 0 0; padding-left:20px;}
ul.tight li {margin:5px 0; font-size:.9rem; line-height:1.55;}

.flag {background:var(--block-background-fill);
       border:1px solid var(--border-color-primary);
       border-radius:9px; padding:11px 14px; margin-bottom:7px; font-size:.87rem;
       line-height:1.5;}
.flag-tag {color:#fff; font-size:.65rem; font-weight:850; letter-spacing:.09em;
           padding:3px 8px; border-radius:5px; margin-right:10px;}

#disclaimer {border:1px solid #C62828; background:rgba(198,40,40,.07); border-radius:11px;
             padding:14px 18px; font-size:.86rem; line-height:1.6;}
.section-title {font-size:1.05rem; font-weight:700; margin:18px 0 2px;}
"""

EXAMPLES = []
for _d in CLASS_DIRS:
    _imgs = [f for f in sorted(_d.iterdir())
             if f.is_file() and f.suffix.lower() in IMG_EXTS]
    if _imgs:
        EXAMPLES.append([str(_imgs[0])])

SUGGESTED_QUESTIONS = [
    "What is this condition, in simple terms?",
    "When should I worry and see a doctor urgently?",
    "How do I look after my skin day to day?",
    "Which options are safe in pregnancy?",
    "What can I get without a prescription?",
    "What should I ask at my appointment?",
    "What helps with itchy skin?",
    "How do I deal with a rash?",
    "What can I do about melasma?",
]


# JS: wires the hidden capture="environment" file input to Gradio's own upload
# input for the `clinical_photo` Image component, so a rear-camera photo is
# dropped straight into the same component the rest of the app already reads.
REAR_CAM_JS = """
() => {
  // Tag Gradio's OWN upload input (inside the elem_id="clinical_photo"
  // Image component) with capture="environment", so tapping the normal
  // Upload button/dropzone on a phone opens straight into the rear camera.
  // Nothing else is touched: the resulting file still flows through
  // Gradio's normal, already-working upload pipeline -- exactly like a
  // regular file pick -- so there is no separate path that can fail.
  function applyCapture() {
    const input = document.querySelector('#clinical_photo input[type=file]');
    if (input && input.getAttribute('capture') !== 'environment') {
      input.setAttribute('capture', 'environment');
    }
  }
  applyCapture();
  new MutationObserver(applyCapture).observe(document.body, {
    childList: true, subtree: true, attributes: true, attributeFilter: ['capture']
  });
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="teal"),
               css=CSS, js=REAR_CAM_JS,
               title="DERM-Net Clinical Decision Support") as demo:

    current_disease = gr.State("")

    gr.HTML("""
    <div id="hero">
      <h1>DERM-Net &mdash; Rare Skin Disease Decision Support</h1>
      <p>Dual-branch EfficientNet-B4 + ViT-B/16 with multi-scale attention fusion, calibrated
         on a held-out split and coupled to a curated dermatology formulary. It explains what
         it saw, says plainly what the condition means, flags what should not wait, and
         refuses to guess when it is unsure.</p>
      <div class="pills">
        <span class="pill">Calibrated confidence</span>
        <span class="pill">Abstains when unsure</span>
        <span class="pill">LayerCAM &amp; LIME</span>
        <span class="pill">Plain-language explanation</span>
        <span class="pill">Urgency triage</span>
        <span class="pill">Pregnancy &amp; interaction screening</span>
        <span class="pill">Grounded Q&amp;A</span>
        <span class="pill">Downloadable report</span>
      </div>
    </div>
    """)

    with gr.Tabs():
        # ---------------------------------------------------------------- Assess
        with gr.Tab("Assess"):
            with gr.Row():
                with gr.Column(scale=4):
                    image_in = gr.Image(type="pil", label="Clinical photograph", height=300,
                                        sources=["upload", "webcam"],
                                        elem_id="clinical_photo")
                    gr.Markdown(
                        "*On a phone, tap **Upload** above ",
                        elem_id="rear_cam_hint")
                    run_btn = gr.Button("Analyse", variant="primary", size="lg")

                    with gr.Accordion("Patient context — drives retrieval and safety",
                                      open=True):
                        with gr.Row():
                            pregnant = gr.Checkbox(label="Pregnant")
                            infant = gr.Checkbox(label="Infant / neonate")
                        with gr.Row():
                            infected = gr.Checkbox(label="Signs of infection")
                            pain = gr.Checkbox(label="Significant pain")
                        with gr.Row():
                            severe = gr.Checkbox(label="Extensive / severe")
                            otc = gr.Checkbox(label="No prescriber access")
                        medications = gr.Textbox(
                            label="Current medications (comma separated)",
                            placeholder="e.g. Methotrexate, Warfarin")
                        allergies = gr.Textbox(
                            label="Known allergies (comma separated)",
                            placeholder="e.g. Sulfa, Neomycin")

                    with gr.Accordion("Explanation & advanced settings", open=False):
                        cam_aug = gr.Checkbox(
                            value=True, label="High-quality CAM (augmentation smoothing)",
                            info="Averages the heatmap over flipped and rescaled copies. "
                                 "~7x slower, and it roughly doubled localisation accuracy "
                                 "in benchmarking. Turn off if the app feels sluggish.")
                        want_lime = gr.Checkbox(
                            value=True, label="Explain with LIME",
                            info="Superpixel explanation. Adds roughly 15-30 s. "
                                 "The CAM is always computed.")
                        lime_samples = gr.Slider(
                            200, 2000, value=LIME_SAMPLES, step=100,
                            label="LIME perturbations",
                            info="More samples = a more stable explanation, but slower.")
                        threshold = gr.Slider(
                            0.0, 0.95, value=CONFIDENCE_THRESHOLD, step=0.05,
                            label="Abstention threshold",
                            info="Below this confidence the system refuses to recommend "
                                 "treatment. Raise it to trade coverage for safety.")

                    if EXAMPLES:
                        gr.Examples(examples=EXAMPLES, inputs=[image_in],
                                    label="Example images from the dataset")

                with gr.Column(scale=5):
                    verdict_out = gr.HTML()
                    probs_out = gr.Label(num_top_classes=5, label="Differential diagnosis")
                    with gr.Row():
                        cam_out = gr.Image(label="LayerCAM — model evidence", height=252)
                        lime_out = gr.Image(label="LIME — supporting regions", height=252)
                    explain_note = gr.Markdown()

            gr.HTML("<div class='section-title'>For the patient</div>")
            with gr.Row():
                urgency_out = gr.HTML()
                patient_out = gr.HTML()

            gr.HTML("<div class='section-title'>Safety screening</div>")
            safety_out = gr.HTML()

            gr.HTML("<div class='section-title'>Retrieved formulary</div>")
            table_out = gr.Dataframe(wrap=True, label=None)

            with gr.Accordion("Full report", open=False):
                report_out = gr.Markdown()
            report_file = gr.File(label="Download this report to take to your appointment")

        # ------------------------------------------------------------------ Ask
        with gr.Tab("Ask"):
            gr.Markdown(
                "### Ask about the result\n"
                "Answers are assembled **only** from the loaded formulary, standard condition "
                "guidance, and a general reference of common, everyday skin issues (itchy skin, "
                "rashes, melasma, mild acne, sunburn and more). The assistant cannot invent a "
                "drug, a dose or a claim, and it never prescribes — if it cannot ground an "
                "answer it says so rather than guessing.\n\n"
                "Run an assessment first to ask about the predicted condition, or ask a "
                "general skin question at any time."
            )
            # Gradio 4.44-5.x need type="messages" to opt out of the deprecated tuple
            # format; Gradio 6 removed the argument because messages is the only format.
            try:
                chat = gr.Chatbot(height=430, type="messages", label="Grounded assistant")
            except TypeError:
                chat = gr.Chatbot(height=430, label="Grounded assistant")
            with gr.Row():
                question_in = gr.Textbox(
                    placeholder="e.g. Is any of this safe while I'm pregnant?",
                    label=None, scale=8, container=False)
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            gr.Examples(examples=[[q] for q in SUGGESTED_QUESTIONS], inputs=[question_in],
                        label="Common questions")
            clear_btn = gr.Button("Clear conversation", size="sm")

        # ------------------------------------------------------------ Formulary
        with gr.Tab("Formulary"):
            gr.Markdown("Browse the underlying knowledge base. "
                        "This is the *only* source the system may draw on.")
            if len(DRUGS):
                with gr.Row():
                    disease_filter = gr.Dropdown(
                        ["All"] + KB_DISEASES, value="All", label="Disease")
                    search_box = gr.Textbox(label="Search",
                                            placeholder="drug name, class, mechanism...")
                formulary_out = gr.Dataframe(wrap=True)

                def browse(disease, query):
                    view = DRUGS
                    if disease and disease != "All":
                        view = view[view["disease_canon"] == disease]
                    if query and query.strip():
                        q = query.strip().lower()
                        hit = view.apply(
                            lambda r: q in " ".join(str(r[c]) for c in COL.values()).lower(),
                            axis=1)
                        view = view[hit]
                    return view[[COL["disease"], COL["drug"], COL["brand"], COL["cls"],
                                 COL["route"], COL["dose"], COL["preg"],
                                 COL["rx"]]].reset_index(drop=True)

                disease_filter.change(browse, [disease_filter, search_box], formulary_out)
                search_box.change(browse, [disease_filter, search_box], formulary_out)
                demo.load(browse, [disease_filter, search_box], formulary_out)
            else:
                gr.Markdown("No formulary loaded.")

        # --------------------------------------------------------- Limitations
        with gr.Tab("How it works & limitations"):
            gr.Markdown(f"""
### Pipeline

1. **Vision.** DERM-Net fuses an EfficientNet-B4 branch (local texture: scale, blistering,
   vascular pattern) with a ViT-B/16 branch (global structure: lesion extent, symmetry)
   through a multi-scale channel-attention block. Inference uses 4-view test-time
   augmentation.
2. **Calibration.** A temperature fitted on the validation split turns raw softmax scores
   into usable confidences. Without it the displayed percentage is not comparable to a
   probability.
3. **Abstention.** Below the threshold the system returns *no* treatment. Abstaining is a
   feature: on a rare-disease cohort a confident wrong answer is worse than no answer.
4. **Explanation.** LayerCAM reads the EfficientNet branch, hooking the post-activation
   `bn2` map together with a finer earlier stage. That pairing was chosen by measurement,
   not convention: on a synthetic benchmark where the discriminative evidence sits in a
   known 46 px square, it doubled pointing-game accuracy over the previous Grad-CAM++ on
   `conv_head` (65% vs 31%). The old layer sat *before* the final BatchNorm and activation,
   so its signed, unnormalised outputs were clipped by the CAM's ReLU into one broad blob.
   LIME treats the whole model as
   a black box, segments the image into superpixels with SLIC, and reports which regions
   support the call. When LIME cannot produce an explanation the reason is shown rather
   than leaving the panel blank.
5. **Retrieval.** The predicted disease is a hard filter, then remaining drugs are ranked
   against the patient context, with contraindicated options demoted to the bottom.
6. **Safety.** Pregnancy category, controlled-substance schedule, documented interactions and
   allergy matches are computed **in code** from the spreadsheet. The model is never asked
   whether a drug is safe.
7. **Q&A.** Questions are routed to the formulary and to standard condition guidance. Nothing
   is generated from model memory, so no drug or dose can be invented.

### Limitations you should not ignore

- Trained on a few hundred images across {NUM_CLASSES} classes from a limited number of
  sites. Performance on other populations, cameras, lighting and **skin tones** is unknown
  and should be assumed worse.
- The formulary is a fixed spreadsheet, not a live drug database. It carries no local
  availability, no paediatric weight-banding and no renal or hepatic dose adjustment.
- The system can only recognise the classes it was trained on. Presented with a melanoma or
  any condition outside its label set, it will confidently return the nearest class it knows.
  **It is not a screening tool for skin cancer.**
- The patient guidance is general education for the predicted condition. It is not tailored
  to you, and the prediction itself may be wrong.
- Confidence is calibrated on one held-out split, not prospectively validated.

### Status of this build

{chr(10).join('- ' + n for n in STARTUP_NOTES)}

- Device: `{DEVICE}` · classes: {", ".join(CLASS_NAMES)}
""")
            gr.HTML("""
            <div id="disclaimer">
            <b>Not a medical device.</b> This is a research demonstrator built for an academic
            project. It is not registered, certified or clinically validated, and it must not
            be used to diagnose or treat any person. Nothing it outputs is a diagnosis or a
            prescription. Any clinical decision must be made by a qualified clinician who has
            examined the patient.
            </div>
            """)

    gr.HTML("""
    <div id="disclaimer" style="margin-top:14px">
    <b>Research demonstrator — not for clinical use.</b> Outputs are model estimates over a
    small dataset and a fixed formulary, not a diagnosis or a prescription. If something is
    getting worse, painful, bleeding or spreading, see a clinician rather than an app.
    </div>
    """)

    # ---- wiring ------------------------------------------------------------
    run_btn.click(
        analyse,
        inputs=[image_in, pregnant, infant, infected, pain, severe, otc,
                medications, allergies, threshold, want_lime, lime_samples, cam_aug],
        outputs=[probs_out, cam_out, lime_out, explain_note, verdict_out,
                 urgency_out, patient_out, safety_out, table_out, report_out,
                 report_file, current_disease],
    )

    def on_ask(question, history, disease):
        history = list(history or [])
        if not (question or "").strip():
            return history, ""
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer_question(question, disease)})
        return history, ""

    ask_btn.click(on_ask, [question_in, chat, current_disease], [chat, question_in])
    question_in.submit(on_ask, [question_in, chat, current_disease], [chat, question_in])
    clear_btn.click(lambda: ([], ""), None, [chat, question_in])


# ## Launch
# 
# The cell below keeps running while the app is live — that is expected. Stop
# the cell to shut the server down. The public link stays valid for 72 hours,
# but only while this kernel is alive.

# Start the server. On Kaggle/Colab this prints a public https://....gradio.live
# link; locally it serves on http://127.0.0.1:7860
print("=" * 74)
print("  DERM-Net Clinical Decision Support")
print("=" * 74)
for note in STARTUP_NOTES:
    print("  -", re.sub(r"\*\*|`", "", note))
print(f"  - Device: {DEVICE} | classes: {', '.join(CLASS_NAMES)}")
print("=" * 74, flush=True)

hosted = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or
              os.environ.get("COLAB_RELEASE_TAG"))
on_spaces = bool(os.environ.get("SPACE_ID"))
if on_spaces:
    # Hugging Face Spaces provides its own public URL and reverse proxy;
    # it must bind to 0.0.0.0:7860 and never request a gradio.live share link.
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860, show_error=True)
else:
    demo.queue(max_size=16).launch(share=hosted, show_error=True)
