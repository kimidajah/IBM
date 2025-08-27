#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Granite via Replicate — Text Classification & Summarization Pipeline
--------------------------------------------------------------------
- Loads TSV with columns: "Review" (text) and optional "Liked" (ground-truth: 1/0)
- Uses IBM Granite model hosted on Replicate to:
  1) Classify sentiment for each review (Positive/Negative/Mixed)
  2) Summarize themes across the dataset
- Saves:
  - predictions.csv (with model label + confidence if available)
  - report.md (overall metrics & qualitative summary)
Usage:
  export REPLICATE_API_TOKEN=your_token_here
  python granite_replicate_pipeline.py --data /path/to/data.tsv --model ibm-granite/granite-3.2-8b-instruct
"""
import os
import re
import argparse
import time
from typing import List, Dict, Any, Tuple

import pandas as pd

# You need: pip install replicate
try:
    import replicate
except ImportError:
    raise SystemExit("Please install the Replicate client: pip install replicate")

DEFAULT_MODEL = "ibm-granite/granite-3.2-8b-instruct"

CLASS_LABELS = ["Positive", "Negative", "Mixed"]

CLASSIFICATION_SYSTEM_PROMPT = """\
You are a precise data annotator. Classify each customer review's sentiment as exactly one of:
- Positive
- Negative
- Mixed

Return a single JSON object with keys:
- "label" -> one of ["Positive", "Negative", "Mixed"]
- "rationale" -> a brief explanation in one sentence
"""

CLASSIFICATION_USER_PROMPT_TEMPLATE = """\
Review:
\"\"\"
{review_text}
\"\"\"

Return JSON only.
"""

SUMMARY_SYSTEM_PROMPT = """\
You are a senior data analyst. Read a sample of customer reviews and provide:
1) 5 key themes you observe
2) 3 actionable recommendations for a product team
3) A concise executive summary (<= 120 words)
Return a clean markdown report.
"""

SUMMARY_USER_PROMPT_TEMPLATE = """\
Here are sample reviews (up to {n} items):
{joined_texts}
"""

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Normalize expected columns
    if "Review" not in df.columns:
        # Try to find a text-like column
        text_col = df.select_dtypes(include=["object"]).columns.tolist()
        if not text_col:
            raise ValueError("No text column found. Expected a 'Review' column.")
        df = df.rename(columns={text_col[0]: "Review"})
    return df

def replicate_client() -> "replicate.Client":
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("Missing REPLICATE_API_TOKEN environment variable.")
    return replicate.Client(api_token=token)

def call_granite_json(client: "replicate.Client", model: str, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 256) -> str:
    """
    Calls an instruction-tuned Granite model on Replicate and returns the raw string output.
    Note: Different models on Replicate may have slightly different input schemas.
    This uses the common 'input' with 'prompt' or 'system'+'prompt' pattern.
    """
    # Try common input shape
    try:
        output = client.run(
            model,
            input={
                "system_prompt": system_prompt,
                "prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    except Exception:
        # Fallback: some models expect "system" and "input"
        output = client.run(
            model,
            input={
                "system": system_prompt,
                "input": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
    # Replicate may stream chunks/list; join to string
    if isinstance(output, (list, tuple)):
        return "".join(map(str, output))
    return str(output)

def parse_label_from_json(text: str) -> Tuple[str, str]:
    """
    Extracts {"label":"...","rationale":"..."} from a JSON-like string.
    Robust to minor formatting noise.
    """
    # Quick path: find label
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
    label = label_match.group(1).strip() if label_match else "UNKNOWN"
    rationale = rationale_match.group(1).strip() if rationale_match else ""
    return label, rationale

def normalize_label(label: str) -> str:
    l = label.strip().lower()
    if "pos" in l:
        return "Positive"
    if "neg" in l:
        return "Negative"
    if "mix" in l or "neutral" in l:
        return "Mixed"
    return "UNKNOWN"

def map_ground_truth(x: Any) -> str:
    # If dataset has "Liked" 1/0, map to Positive/Negative
    try:
        val = int(x)
        return "Positive" if val == 1 else "Negative"
    except Exception:
        # If it's already a string label
        s = str(x).strip().lower()
        if s in ("pos", "positive", "1", "true", "yes"):
            return "Positive"
        if s in ("neg", "negative", "0", "false", "no"):
            return "Negative"
        return "Mixed"

def batch(iterable, n=16):
    b = []
    for item in iterable:
        b.append(item)
        if len(b) == n:
            yield b
            b = []
    if b:
        yield b

def classify_dataframe(df: pd.DataFrame, client: "replicate.Client", model: str, rate_limit_s: float = 0.6) -> pd.DataFrame:
    texts = df["Review"].fillna("").astype(str).tolist()
    labels, rationales, raw_outputs = [], [], []

    for i, text in enumerate(texts, start=1):
        user_prompt = CLASSIFICATION_USER_PROMPT_TEMPLATE.format(review_text=text)
        raw = call_granite_json(client, model, CLASSIFICATION_SYSTEM_PROMPT, user_prompt)
        label, rationale = parse_label_from_json(raw)
        label = normalize_label(label)

        labels.append(label)
        rationales.append(rationale)
        raw_outputs.append(raw)

        # crude rate-limit
        time.sleep(rate_limit_s)

        if i % 25 == 0:
            print(f"[progress] Classified {i}/{len(texts)} rows.", flush=True)

    out = df.copy()
    out["granite_label"] = labels
    out["granite_rationale"] = rationales
    out["granite_raw"] = raw_outputs
    return out

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if "Liked" in df.columns:
        gt = df["Liked"].apply(map_ground_truth)
        pred = df["granite_label"]
        cm = pd.crosstab(gt, pred, dropna=False)
        acc = (gt == pred).mean()
        metrics["accuracy"] = float(acc)
        metrics["confusion_matrix"] = cm
    else:
        metrics["note"] = "No ground truth column (Liked) found; skipping accuracy."
    return metrics

def build_summary_prompt(samples: List[str]) -> str:
    joined = "\n\n".join(f"- {s}" for s in samples)
    return SUMMARY_USER_PROMPT_TEMPLATE.format(n=len(samples), joined_texts=joined)

def summarize_reviews(df: pd.DataFrame, client: "replicate.Client", model: str, sample_size: int = 80) -> str:
    samples = df["Review"].dropna().astype(str).sample(min(sample_size, len(df)), random_state=42).tolist()
    user_prompt = build_summary_prompt(samples)
    raw = call_granite_json(client, model, SUMMARY_SYSTEM_PROMPT, user_prompt, temperature=0.3, max_tokens=600)
    return str(raw)

def save_artifacts(pred_df: pd.DataFrame, metrics: Dict[str, Any], summary_md: str, out_dir: str = ".") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    pred_path = os.path.join(out_dir, "predictions.csv")
    report_path = os.path.join(out_dir, "report.md")

    # Save predictions
    pred_df.to_csv(pred_path, index=False)

    # Build markdown report
    lines = ["# Granite Classification Report", ""]
    if "accuracy" in metrics:
        lines.append(f"**Accuracy**: {metrics['accuracy']:.4f}")
        lines.append("")
    if "confusion_matrix" in metrics:
        lines.append("## Confusion Matrix (GT x Pred)")
        lines.append("")
        lines.append(metrics["confusion_matrix"].to_markdown())
        lines.append("")
    if "note" in metrics:
        lines.append(f"> {metrics['note']}")
        lines.append("")

    lines.append("## Thematic Summary")
    lines.append("")
    lines.append(summary_md.strip())
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return pred_path, report_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to TSV file (expects column 'Review' and optional 'Liked')")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Replicate model slug")
    parser.add_argument("--out_dir", default="outputs", help="Directory to write outputs")
    parser.add_argument("--rate_limit_s", type=float, default=0.6, help="Sleep seconds between API calls")
    parser.add_argument("--sample_size", type=int, default=80, help="How many reviews to include in the summary prompt")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

    client = replicate_client()

    pred_df = classify_dataframe(df, client, args.model, rate_limit_s=args.rate_limit_s)
    metrics = compute_metrics(pred_df)

    summary_md = summarize_reviews(df, client, args.model, sample_size=args.sample_size)

    pred_path, report_path = save_artifacts(pred_df, metrics, summary_md, out_dir=args.out_dir)

    print(f"\nSaved predictions to: {pred_path}")
    print(f"Saved report to: {report_path}")
    print("\nDone.")

if __name__ == "__main__":
    main()
