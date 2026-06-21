"""
Quick benchmark: measure actual training step speed with realistic batch size
to estimate total training time before committing to a full run.
"""

import time
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

MODEL_NAME = "google/mt5-small"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("Loading a realistic sample (200 rows) from full Train.csv...")
train_df = pd.read_csv("data/Train.csv").sample(n=200, random_state=SEED).reset_index(drop=True)

def make_prefixed_input(row):
    return f"answer health question in {row['subset']}: {row['input']}"

train_df["model_input"] = train_df.apply(make_prefixed_input, axis=1)
train_dataset = Dataset.from_pandas(train_df[["model_input", "output"]])

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cuda")

def preprocess(examples):
    model_inputs = tokenizer(examples["model_input"], max_length=MAX_INPUT_LEN, truncation=True)
    labels = tokenizer(text_target=examples["output"], max_length=MAX_TARGET_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=["model_input", "output"])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Test a realistic batch size for training (no eval here, pure train speed)
BATCH_SIZE = 8

training_args = Seq2SeqTrainingArguments(
    output_dir="models/benchmark-tmp",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    bf16=True,
    logging_steps=5,
    report_to="none",
    save_strategy="no",
    seed=SEED,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print(f"\nBenchmarking with batch_size={BATCH_SIZE} on 200 samples...")
start = time.time()
trainer.train()
elapsed = time.time() - start

steps_per_sec = trainer.state.global_step / elapsed
samples_per_sec = 200 / elapsed

print("\n" + "=" * 60)
print(f"RESULTS:")
print(f"Total time for 200 samples, 1 epoch, batch={BATCH_SIZE}: {elapsed:.1f} sec")
print(f"Samples/sec: {samples_per_sec:.3f}")
print(f"Steps/sec: {steps_per_sec:.3f}")
print()

FULL_TRAIN_SIZE = 29815
estimated_per_epoch = FULL_TRAIN_SIZE / samples_per_sec
print(f"ESTIMATED time for 1 full epoch ({FULL_TRAIN_SIZE} samples): {estimated_per_epoch/60:.1f} minutes ({estimated_per_epoch/3600:.2f} hours)")
print(f"ESTIMATED time for 3 full epochs: {3*estimated_per_epoch/60:.1f} minutes ({3*estimated_per_epoch/3600:.2f} hours)")
print("=" * 60)