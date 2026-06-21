# 🌍 Multilingual Health Question Answering for Low-Resource African Languages

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://multilingual-health-app.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

A multilingual question-answering system developed for the **Zindi & ITU Multilingual Health Question Answering Challenge**, in collaboration with HASH (Hub for AI in Maternal, Sexual and Reproductive Health).

The goal is to generate accurate answers to maternal, sexual, and reproductive health (MSRH) questions in the same language as the input question across multiple low-resource African languages.

---

## Live Demo

Try the deployed Streamlit dashboard:

https://multilingual-health-app.streamlit.app/

### Features

* Dataset overview and language distribution
* Per-language model performance breakdown (ROUGE-1 / ROUGE-L)
* Full approach and methodology summary

> The dashboard displays results from the trained model; it does not run live inference in the cloud.

---

## Author

**Mustafa Elsherif**

Email: mustafaelsherif99@gmail.com

LinkedIn: https://www.linkedin.com/in/mustafa-elsherif-77b74729a

---

## Challenge Overview

This project addresses multilingual health question answering across five languages and eight language-country configurations.

| Language Configuration | Language | Country  |
| ----------------------- | -------- | -------- |
| Eng_Uga                 | English  | Uganda   |
| Eng_Gha                 | English  | Ghana    |
| Eng_Eth                 | English  | Ethiopia |
| Eng_Ken                 | English  | Kenya    |
| Aka_Gha                 | Akan     | Ghana    |
| Amh_Eth                 | Amharic  | Ethiopia |
| Lug_Uga                 | Luganda  | Uganda   |
| Swa_Ken                 | Swahili  | Kenya    |

The model receives a health-related question and generates a natural-language answer in the same language.

---

## Dataset

| Split      | Samples |
| ---------- | ------: |
| Train      |  29,815 |
| Validation |   6,686 |
| Test       |   2,618 |

Competition data is not included in this repository. Download the dataset from Zindi and place the following files inside the data/ directory:

```text
data/
├── Train.csv
├── Val.csv
├── Test.csv
└── SampleSubmission.csv
```

---

## Model Approach

### Base Model

google/mt5-small

A multilingual sequence-to-sequence Transformer model fine-tuned jointly on all language subsets.

### Input Prompt Format

```text
answer health question in Aka_Gha: <question>
```

### Why mT5-Small?

Training was performed on an 8GB GPU. Larger variants such as mT5-Base and mT5-Large were benchmarked but required substantially longer training times and exceeded practical resource constraints.

mT5-Small provided stable training behavior, reasonable memory usage, fast experimentation cycles, and competitive multilingual performance.

---

## Training Configuration

| Parameter                   | Value     |
| ---------------------------- | --------- |
| Model                         | mT5-Small |
| Epochs                        | 3         |
| Learning Rate                 | 3e-4      |
| Max Input Length               | 128       |
| Max Target Length              | 256       |
| Batch Size                     | 2         |
| Gradient Accumulation Steps    | 8         |
| Effective Batch Size           | 16        |
| Precision                      | bf16      |
| Random Seed                    | 42        |

---

## Decoding Strategy

| Parameter            | Value |
| --------------------- | ----- |
| num_beams              | 4     |
| no_repeat_ngram_size   | 3     |
| repetition_penalty     | 1.3   |

---

## Local Validation Results

| Metric     |  Score |
| ----------- | -----: |
| ROUGE-1 F1   | 0.2713 |
| ROUGE-L F1   | 0.1979 |

### Key Observations

* English and Akan subsets achieved the strongest performance.
* Amharic remained the most challenging language.
* Ge'ez script tokenization appears to be a major bottleneck.

---

## Repository Structure

```text
.
├── data/
├── src/
│   ├── explore_data.py
│   ├── train_baseline.py
│   ├── evaluate_model.py
│   └── generate_submission.py
├── models/
├── submissions/
├── .streamlit/
│   └── config.toml
├── dashboard.py
├── requirements.txt
├── requirements-dashboard.txt
└── README.md
```

### Scripts

| Script                  | Description                               |
| ------------------------ | ------------------------------------------ |
| explore_data.py           | Dataset exploration and analysis           |
| train_baseline.py         | Model training pipeline                    |
| evaluate_model.py          | Validation evaluation                      |
| generate_submission.py     | Test inference and submission generation   |
| dashboard.py                | Streamlit dashboard                        |

---

## Reproducing Results

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_baseline.py
python src/evaluate_model.py
python src/generate_submission.py
streamlit run dashboard.py
```

---

## Technologies Used

Python, PyTorch, Hugging Face Transformers, Pandas, NumPy, Streamlit, ROUGE Evaluation

---

## Reproducibility

* Fixed random seed (42)
* Open-source libraries only
* No AutoML systems used
* Trained exclusively on competition-provided data

---

## License

This repository is intended for research and educational purposes. Please comply with the original competition rules, dataset licenses, and usage policies provided by Zindi and ITU.
