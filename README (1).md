# DERM-Net — Clinical Decision Support (Research Demonstrator)

An AI-assisted skin-condition triage tool. Upload a clinical photograph and
the app runs a full pipeline: DERM-Net (EfficientNet-B4 + ViT-B/16 fusion) →
calibrated confidence → abstention gate → LayerCAM / LIME explanation →
plain-language patient guidance → drug-formulary retrieval → deterministic
safety checks (pregnancy / controlled substance / drug & allergy
interactions) → downloadable clinician report → grounded question
answering restricted to the loaded knowledge base.

**Not a medical device.** This is a research/academic demonstrator. It is
not validated, not certified, and must never be used to diagnose or treat
a real patient. See the in-app "How it works & limitations" tab for the
full list of caveats.

## Live demo

Hosted on Hugging Face Spaces (ZeroGPU):
https://huggingface.co/spaces/israt125/derm-net

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Gradio application — model, retrieval, safety engine, and UI |
| `dermnet_best.pth` | Trained DERM-Net checkpoint (~430MB, Git LFS) |
| `Disease_Drug__Active_Ingredient-f4.xlsx` | Drug formulary the safety engine and retrieval draw from |
| `common_skin_issues_selfcare.csv` | General, disease-agnostic self-care knowledge base |
| `Data/` | Example images used to populate the "try a sample" gallery (optional) |
| `requirements.txt` | Python dependencies |

All three knowledge/model files are auto-discovered by `app.py` from the
working directory — no path editing is required if they stay alongside it.

## Running locally

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## Deploying your own copy

The app runs unchanged on:
- **Hugging Face Spaces** (Gradio SDK) — simplest option, supports the
  ~430MB checkpoint via Git LFS out of the box.
- Any host that can run a persistent Python process (Render, Railway, a VPS).

It is **not** suitable for classic serverless platforms (e.g. Vercel
serverless functions) because of the checkpoint size and inference/LIME
runtime, which exceed typical serverless size and timeout limits.

## Pipeline summary

1. **Vision** — EfficientNet-B4 (local texture) + ViT-B/16 (global
   structure) fused through a multi-scale channel-attention block; 4-view
   test-time augmentation.
2. **Calibration** — temperature scaling turns raw softmax into a usable
   confidence.
3. **Abstention** — below the confidence threshold, no treatment is
   suggested.
4. **Explanation** — LayerCAM (EfficientNet branch) and LIME
   (model-agnostic, SLIC superpixels).
5. **Retrieval** — predicted disease filters the formulary; results are
   ranked against patient context, with contraindicated options demoted
   rather than hidden.
6. **Safety** — pregnancy category, controlled-substance schedule, drug
   interactions, and allergy matches are computed in code from the
   spreadsheet, never inferred by the model.
7. **Q&A** — answers are grounded only in the loaded formulary and
   knowledge base; nothing is generated from model memory.

## License

Add the license that fits your project (see `LICENSE`).
