# Next Word Prediction using LSTM

A deep learning-based **Next Word Prediction** model built with TensorFlow and Keras. Trained on *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, this model predicts the next sequence of words given an initial seed phrase.

---

## 📌 Project Overview

This project implements a Natural Language Processing (NLP) pipeline from scratch to train a stacked Long Short-Term Memory (LSTM) neural network. The workflow covers dataset ingestion via Kaggle, text cleaning, lemmatization, sequence generation, model training, and multi-word inference.

---

## 🛠️ Tech Stack & Libraries

* **Python 3.12**
* **TensorFlow / Keras** — Model architecture, tokenization, and sequence padding
* **NLTK** — WordNet Lemmatization and sentence tokenization
* **NumPy & Scikit-Learn** — Data reshaping and train-test splitting
* **KaggleHub** — Automated dataset downloading

---

## 📊 Dataset

* **Source:** [Next Word Prediction Dataset on Kaggle](https://kaggle.com) (`ronikdedhia/next-word-prediction`)
* **Corpus:** *The Adventures of Sherlock Holmes* (Project Gutenberg text)
* **Vocabulary Size:** 7,820 unique lemmatized words

---

## ⚙️ Data Preprocessing Pipeline

1. **Lowercasing & Regex Filtering:** Removed punctuation, numbers, and special characters using regex (`[a-z\s]`).
2. **Lemmatization:** Applied `WordNetLemmatizer` to reduce words to their base dictionary forms (reducing corpus character count from `557,883` to `547,918`).
3. **Tokenization:** Converted words into integer indices using Keras `Tokenizer`.
4. **Sliding Window Sequence Generation:** Created input-output pairs using a context window of **10 words** (`X`) to predict the **11th word** (`y`).

---

## 🧠 Model Architecture

The model uses a sequential architecture with two stacked LSTM layers followed by a dense softmax output layer:

| Layer | Type | Output Shape | Parameters |
| :--- | :--- | :--- | :--- |
| **Input** | InputLayer | `(None, 10, 1)` | 0 |
| **LSTM 1** | LSTM (`return_sequences=True`) | `(None, 10, 512)` | 1,052,672 |
| **LSTM 2** | LSTM | `(None, 256)` | 787,456 |
| **Output** | Dense (`softmax`) | `(None, 8646)` | 2,222,022 |

* **Loss Function:** `sparse_categorical_crossentropy`
* **Optimizer:** `adam`
* **Total Trainable Parameters:** `4,062,150`

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd your-repo-name
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow nltk numpy scikit-learn kagglehub
   ```

3. **Run the Jupyter Notebook / Google Colab:**
   Open the `.ipynb` file in Jupyter Notebook or Google Colab and execute all cells sequentially.

---

## 💡 Usage & Example Output

Use the `predict_5_real_words()` helper function to generate the next 5 words from any seed text:

```python
test_seed = "sherlock holmes by arthur"
print(predict_5_real_words(test_seed))
```

**Sample Output:**
```text
Input seed: sherlock holmes by arthur
Predicted: sherlock holmes by arthur and the man of the
```

---

## 🔮 Future Improvements

* Add an **Embedding layer** (`layers.Embedding`) before the LSTM layers to capture semantic word relationships rather than passing raw integer indices.
* Train for additional epochs (e.g., 20–50 epochs) with `EarlyStopping` and `ModelCheckpoint` callbacks to improve prediction coherence.
* Implement **Temperature Sampling** or **Top-K sampling** during inference to avoid repetitive word loops.
