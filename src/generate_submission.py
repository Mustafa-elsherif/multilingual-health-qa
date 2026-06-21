"""
Generate predictions on Test.csv using the trained model,
and build the submission file in the exact required format:
ID, TargetRLF1, TargetR1F1, TargetLLM (identical values across the 3 columns).
"""

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

MODEL_DIR = "models/mt5-small-baseline"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 256
BATCH_SIZE = 16
SEED = 42

torch.manual_seed(SEED)

print("Loading test data...")
test_df = pd.read_csv("data/Test.csv")
print(f"Test shape: {test_df.shape}")

def make_prefixed_input(row):
    return f"answer health question in {row['subset']}: {row['input']}"

test_df["model_input"] = test_df.apply(make_prefixed_input, axis=1)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()

predictions = []

print("Generating predictions...")
inputs_list = test_df["model_input"].tolist()

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

test_df["prediction"] = predictions

# ---------------------------------------------------------------------------
# Build submission file - EXACT format required by competition:
# ID, TargetRLF1, TargetR1F1, TargetLLM (same value in all 3 columns)
# ---------------------------------------------------------------------------
print("Building submission file...")
submission = pd.DataFrame({
    "ID": test_df["ID"],
    "TargetRLF1": test_df["prediction"],
    "TargetR1F1": test_df["prediction"],
    "TargetLLM": test_df["prediction"],
})

OUTPUT_PATH = "submissions/submission_v2_mt5small_antirep.csv"
submission.to_csv(OUTPUT_PATH, index=False)

print(f"Submission saved to: {OUTPUT_PATH}")
print(f"Shape: {submission.shape}")
print("\nFirst 3 rows:")
print(submission.head(3))