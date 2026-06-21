\# Multilingual Health Question Answering — Low-Resource African Languages



Solution for the \[Zindi/ITU Multilingual Health QA Challenge](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge), built in collaboration with HASH (Hub for AI in Maternal, Sexual and Reproductive Health).



\*\*Implemented by:\*\* Mustafa Elsherif

\*\*Email:\*\* \[mustafaelsherif99@gmail.com](mailto:mustafaelsherif99@gmail.com)

\*\*LinkedIn:\*\* <a href="https://www.linkedin.com/in/mustafa-elsherif-77b74729a">https://www.linkedin.com/in/mustafa-elsherif-77b74729a</a>



\*\*Public Leaderboard Score: 0.277961\*\*



\## Problem



Build a multilingual sequence-to-sequence model that answers maternal, sexual, and reproductive health (MSRH) questions in the same language they were asked, across 5 languages and 8 language-country configurations:



| Subset                             | Language | Country                        |

| ---------------------------------- | -------- | ------------------------------ |

| Eng\_Uga, Eng\_Gha, Eng\_Eth, Eng\_Ken | English  | Uganda, Ghana, Ethiopia, Kenya |

| Aka\_Gha                            | Akan     | Ghana                          |

| Amh\_Eth                            | Amharic  | Ethiopia                       |

| Lug\_Uga                            | Luganda  | Uganda                         |

| Swa\_Ken                            | Swahili  | Kenya                          |



\## Dataset



\* \*\*Train\*\*: 29,815 question-answer pairs

\* \*\*Validation\*\*: 6,686 question-answer pairs

\* \*\*Test\*\*: 2,618 questions (answers to be generated)



Data is not included in this repo (per competition guidelines, available directly from \[Zindi](https://zindi.africa/competitions/multilingual-health-question-answering-in-low-resource-african-languages-challenge/data)). Place `Train.csv`, `Val.csv`, `Test.csv`, `SampleSubmission.csv` into `data/` before running.



\## Approach



\*\*Model\*\*: \[`google/mt5-small`](https://huggingface.co/google/mt5-small) — a multilingual sequence-to-sequence transformer, fine-tuned end-to-end on all 8 language subsets jointly, with the target subset included as a text prefix (e.g. `"answer health question in Aka\_Gha: ..."`).



\*\*Why mT5-small\*\*: Given an 8GB GPU and a tight time budget, larger variants (mT5-base, mT5-large) were benchmarked and found to require 7-21+ hours per epoch with this hardware — infeasible within the competition deadline. mT5-small trains a full 3-epoch run in \~2.5-4 hours and produces stable, non-degenerate training (verified via loss/gradient-norm monitoring) — the dependable choice under the constraints.



\*\*Generation\*\*: Beam search (`num\_beams=4`) with `no\_repeat\_ngram\_size=3` and `repetition\_penalty=1.3` to prevent the repetition loops common in small seq2seq models — this alone improved local ROUGE-1 by \~22% over plain beam search.



\*\*Training config\*\*:



\* `max\_input\_length=128`, `max\_target\_length=256`

\* Effective batch size 16 (`batch\_size=2` × `gradient\_accumulation\_steps=8`)

\* `bf16` mixed precision (more stable than `fp16` for mT5 architectures)

\* 3 epochs, `learning\_rate=3e-4`

\* Seed fixed at 42 throughout for reproducibility



\## Repository Structure



```text

├── src/

│   ├── explore\_data.py          # Initial dataset exploration

│   ├── train\_baseline.py        # Main training script

│   ├── evaluate\_model.py        # Local ROUGE evaluation on Val.csv (per-language breakdown)

│   └── generate\_submission.py   # Inference on Test.csv + submission file builder

├── submissions/

│   └── submission\_v2\_mt5small\_antirep.csv   # Submitted file (Public Score: 0.277961)

├── models/                      # Trained model (gitignored — large files; reproducible via src/train\_baseline.py)

└── requirements.txt

```



\## Reproducing Results



```bash

python -m venv venv

venv\\Scripts\\activate        # Windows

pip install -r requirements.txt



python src/explore\_data.py            # optional: inspect the data

python src/train\_baseline.py          # trains and saves to models/mt5-small-baseline

python src/evaluate\_model.py          # local ROUGE-1 / ROUGE-L on validation set

python src/generate\_submission.py     # generates submissions/\*.csv

```



\## Local Validation Results (1000-sample held-out check)



| Metric     | Score  |

| ---------- | ------ |

| ROUGE-1 F1 | 0.2713 |

| ROUGE-L F1 | 0.1979 |



Per-language breakdown showed strong variance — high-resource subsets (English, Akan) scored 0.27-0.40 ROUGE-1, while Amharic (Ge'ez script, smallest subset at 1,845 training rows) lagged significantly (\~0.02), suggesting tokenizer/script coverage limitations for mT5 on Ge'ez script as the primary bottleneck rather than data volume alone.



\## Future Improvements



Given more time/compute, the following were identified as promising next steps:



\* Larger models (mT5-base/large) with corrected hyperparameters (lower LR, longer warmup) to address training instability observed at scale

\* Amharic-specific handling (script-aware tokenization or a dedicated model for Ge'ez script)

\* Longer `max\_target\_length` (some references run up to 2,956 characters) traded against compute budget



\## Compliance



\* Open-source tools only (Hugging Face Transformers, PyTorch, pandas)

\* No AutoML

\* Fixed seed (42) for reproducibility

\* Trained entirely on competition-provided data



