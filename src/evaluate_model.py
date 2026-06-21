"""
Evaluate the trained model on the Validation set using the same metrics
the competition uses (ROUGE-1 F1, ROUGE-L F1), to estimate expected
leaderboard performance BEFORE spending a submission.
"""

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import evaluate

MODEL_DIR = "models/mt5-small-baseline"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 256
BATCH_SIZE = 16
SEED = 42
N_EVAL_SAMPLES = 1000  # use a larger, more reliable sample than the 500 used during training

torch.manual_seed(SEED)

print("Loading validation data...")
val_df = pd.read_csv("data/Val.csv")
val_df = val_df.sample(n=min(N_EVAL_SAMPLES, len(val_df)), random_state=SEED).reset_index(drop=True)
print(f"Evaluating on {len(val_df)} samples")

def make_prefixed_input(row):
    return f"answer health question in {row['subset']}: {row['input']}"

val_df["model_input"] = val_df.apply(make_prefixed_input, axis=1)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()

predictions = []
inputs_list = val_df["model_input"].tolist()

print("Generating predictions on validation set...")
for i in tqdm(range(0, len(inputs_list), BATCH_SIZE)):
    batch_texts = inputs_list[i:i + BATCH_SIZE]
    encoded = tokenizer(
        batch_texts,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_length=MAX_TARGET_LEN,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
        )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    predictions.extend(decoded)

val_df["prediction"] = predictions

# ---------------------------------------------------------------------------
# Compute ROUGE-1 and ROUGE-L (same metrics as competition, 0.37 + 0.37 weight)
# ---------------------------------------------------------------------------
print("\nComputing ROUGE scores...")
rouge = evaluate.load("rouge")
result = rouge.compute(
    predictions=val_df["prediction"].tolist(),
    references=val_df["output"].tolist(),
    use_stemmer=False,
)

print("\n" + "=" * 60)
print("OVERALL RESULTS (on", len(val_df), "validation samples)")
print("=" * 60)
print(f"ROUGE-1 F1: {result['rouge1']:.4f}")
print(f"ROUGE-L F1: {result['rougeL']:.4f}")

# Estimate competition score (excluding LLM-judge which we can't compute locally)
# Weighted: 0.37 * rouge1 + 0.37 * rougeL (LLM-judge weight 0.26 unknown, often correlates with rouge)
estimated_partial = 0.37 * result['rouge1'] + 0.37 * result['rougeL']
print(f"\nPartial weighted score (ROUGE only, 0.74 of total weight): {estimated_partial:.4f}")
print("(Full competition score also includes LLM-as-Judge, weight 0.26 - not computable locally)")

# ---------------------------------------------------------------------------
# Breakdown by language subset - important to see if some languages lag behind
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BREAKDOWN BY LANGUAGE SUBSET")
print("=" * 60)
for subset in sorted(val_df["subset"].unique()):
    subset_df = val_df[val_df["subset"] == subset]
    if len(subset_df) < 3:
        continue
    subset_result = rouge.compute(
        predictions=subset_df["prediction"].tolist(),
        references=subset_df["output"].tolist(),
        use_stemmer=False,
    )
    print(f"{subset:12s} (n={len(subset_df):4d})  ROUGE-1: {subset_result['rouge1']:.4f}  ROUGE-L: {subset_result['rougeL']:.4f}")

print("=" * 60)

# Save a few examples for manual inspection
print("\nSample predictions vs references:")
for i in range(3):
    print(f"\n--- Example {i+1} ({val_df.iloc[i]['subset']}) ---")
    print("INPUT:", val_df.iloc[i]['input'][:150])
    print("REFERENCE:", val_df.iloc[i]['output'][:200])
    print("PREDICTION:", val_df.iloc[i]['prediction'][:200])