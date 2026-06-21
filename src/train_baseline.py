"""
Baseline training script for Multilingual Health QA Challenge (Zindi/ITU)
Model: google/mt5-small
Task: input (question) -> output (health answer), across 8 language subsets
"""

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "google/mt5-small"
MAX_INPUT_LEN = 128     # covers ~95% of inputs (max seen was 520 chars ~ much fewer tokens)
MAX_TARGET_LEN = 256    # balances coverage vs training speed
OUTPUT_DIR = "models/mt5-small-baseline"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv("data/Train.csv")
val_df = pd.read_csv("data/Val.csv")

print(f"Train: {train_df.shape}, Val: {val_df.shape}")

# Prefix with task + language tag so the model knows what to generate
# (mT5 has no special language tokens, so we add a simple text prefix)
def make_prefixed_input(row):
    return f"answer health question in {row['subset']}: {row['input']}"

train_df["model_input"] = train_df.apply(make_prefixed_input, axis=1)
val_df["model_input"] = val_df.apply(make_prefixed_input, axis=1)

train_dataset = Dataset.from_pandas(train_df[["model_input", "output"]])
# Use a subsample for per-epoch eval (full eval set is too slow during training)
val_df_subset = val_df.sample(n=min(500, len(val_df)), random_state=SEED).reset_index(drop=True)
val_dataset = Dataset.from_pandas(val_df_subset[["model_input", "output"]])

# ---------------------------------------------------------------------------
# Tokenizer & model
# ---------------------------------------------------------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

def preprocess(examples):
    model_inputs = tokenizer(
        examples["model_input"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["output"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=["model_input", "output"])
val_tokenized = val_dataset.map(preprocess, batched=True, remove_columns=["model_input", "output"])

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ---------------------------------------------------------------------------
# Metrics (ROUGE-1, ROUGE-L -- matches 2 of the 3 competition metrics)
# ---------------------------------------------------------------------------
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=False)
    return {"rouge1": result["rouge1"], "rougeL": result["rougeL"]}

# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,   # effective batch size = 16
    weight_decay=0.01,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    bf16=True,                       # speeds up training, more stable than fp16 for mT5
    logging_steps=50,
    report_to="none",
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("Starting training...")
trainer.train()

print("Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Done! Model saved to:", OUTPUT_DIR)