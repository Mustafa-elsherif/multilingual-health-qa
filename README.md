# 🌍 Multilingual Health Question Answering for Low-Resource African Languages

A multilingual question-answering system developed for the Zindi & ITU **Multilingual Health Question Answering Challenge**, in collaboration with HASH (Hub for AI in Maternal, Sexual and Reproductive Health).

The goal is to generate accurate answers to maternal, sexual, and reproductive health (MSRH) questions in the same language as the input question across multiple low-resource African languages.

---

## 👨‍💻 Author

**Mustafa Elsherif**

📧 Email: [mustafaelsherif99@gmail.com](mailto:mustafaelsherif99@gmail.com)

💼 LinkedIn: https://www.linkedin.com/in/mustafa-elsherif-77b74729a

---

## 🎯 Challenge Overview

This project addresses multilingual health question answering across five languages and eight language-country configurations.

| Language Configuration | Language | Country  |
| ---------------------- | -------- | -------- |
| Eng_Uga                | English  | Uganda   |
| Eng_Gha                | English  | Ghana    |
| Eng_Eth                | English  | Ethiopia |
| Eng_Ken                | English  | Kenya    |
| Aka_Gha                | Akan     | Ghana    |
| Amh_Eth                | Amharic  | Ethiopia |
| Lug_Uga                | Luganda  | Uganda   |
| Swa_Ken                | Swahili  | Kenya    |

The model receives a health-related question and generates a natural-language answer in the same language.

---

## 📊 Dataset

| Split      | Samples |
| ---------- | ------: |
| Train      |  29,815 |
| Validation |   6,686 |
| Test       |   2,618 |

### Dataset Access

Competition data is not included in this repository.

Download the dataset from Zindi and place the following files inside the `data/` directory:

```text
data/
├── Train.csv
├── Val.csv
├── Test.csv
└── SampleSubmission.csv
```

---

## 🧠 Model Approach

### Base Model

**google/mt5-small**

A multilingual sequence-to-sequence Transformer model fine-tuned on all language subsets jointly.

### Input Format

Each sample is converted into a task-oriented prompt:

```text
answer health question in Aka_Gha: <question>
```

This allows a single model to learn language-specific generation while sharing knowledge across languages.

---

## ⚙️ Training Configuration

| Parameter             | Value     |
| --------------------- | --------- |
| Model                 | mT5-small |
| Epochs                | 3         |
| Learning Rate         | 3e-4      |
| Max Input Length      | 128       |
| Max Target Length     | 256       |
| Batch Size            | 2         |
| Gradient Accumulation | 8         |
| Effective Batch Size  | 16        |
| Precision             | bf16      |
| Seed                  | 42        |

---

## 🚀 Decoding Strategy

To reduce repetitive generations commonly observed in small seq2seq models:

| Parameter            | Value |
| -------------------- | ----- |
| num_beams            | 4     |
| no_repeat_ngram_size | 3     |
| repetition_penalty   | 1.3   |

This configuration produced noticeably more stable and coherent outputs compared to standard beam search.

---

## 📁 Repository Structure

```text
.
├── data/
│
├── src/
│   ├── explore_data.py
│   ├── train_baseline.py
│   ├── evaluate_model.py
│   └── generate_submission.py
│
├── models/
│   └── (generated after training)
│
├── submissions/
│   └── submission.csv
│
├── requirements.txt
└── README.md
```

### Scripts

| Script                 | Description                              |
| ---------------------- | ---------------------------------------- |
| explore_data.py        | Dataset exploration and statistics       |
| train_baseline.py      | Model training                           |
| evaluate_model.py      | Validation evaluation                    |
| generate_submission.py | Test inference and submission generation |

---

## 🔄 Reproducing Results

### 1. Create Environment

```bash
python -m venv venv
```

### 2. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Pipeline

```bash
python src/explore_data.py

python src/train_baseline.py

python src/evaluate_model.py

python src/generate_submission.py
```

---

## 📈 Local Validation Results

Validation was performed on a held-out subset of 1,000 samples.

| Metric     |  Score |
| ---------- | -----: |
| ROUGE-1 F1 | 0.2713 |
| ROUGE-L F1 | 0.1979 |

### Key Observations

* English and Akan subsets achieved the strongest performance.
* Amharic remained the most challenging language.
* Performance degradation appears strongly linked to Ge'ez script tokenization limitations.
* Cross-lingual training improved generalization for low-resource subsets.

---

## 🔍 Challenges

### Limited Compute Resources

Training was conducted on an 8GB GPU.

Larger variants such as mT5-Base and mT5-Large were explored but proved impractical under the available hardware and competition timeline.

### Low-Resource Languages

Some language subsets contained significantly fewer examples, making generalization difficult, especially for Amharic.

---

## 💡 Future Work

Potential improvements include:

* Fine-tuning larger mT5 variants
* Language-specific optimization for Amharic
* Improved tokenization for Ge'ez script
* Longer output sequence lengths
* Hyperparameter optimization and learning-rate scheduling
* Data augmentation for low-resource subsets

---

## ✅ Reproducibility

* Fixed random seed (`42`)
* Open-source libraries only
* No AutoML systems used
* Trained exclusively on competition-provided data

---

## 🛠️ Technologies

* Python
* PyTorch
* Hugging Face Transformers
* Pandas
* NumPy
* ROUGE Evaluation

---

## 📜 License

This repository is intended for research and educational purposes. Please follow the original competition terms and dataset usage policies provided by Zindi.
