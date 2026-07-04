# Architecture — Disaster Tweet Classification

## System overview

This is a classical NLP classification pipeline. Raw tweet text enters a preprocessing stage that normalizes and cleans it, then passes through a TF-IDF vectorizer that converts text to numerical sparse matrix form, and finally through a classifier that outputs binary predictions (1 = real disaster, 0 = not). The pipeline is fit on training data only and evaluated on a held-out validation set before generating Kaggle submission predictions on the test set.

## Pipeline diagram

```
Raw Tweet Text (train.csv / test.csv)
        |
        v
[ Text Cleaning ]
  - Lowercase
  - Remove URLs (re.sub)
  - Remove @mentions
  - Remove special characters / punctuation
  - Strip extra whitespace
        |
        v
[ Tokenization + Normalization ]
  - NLTK word_tokenize
  - Remove stopwords (NLTK English corpus)
  - Lemmatize tokens (NLTK WordNetLemmatizer)
  - Rejoin into cleaned string
        |
        v
[ TF-IDF Vectorization ]           ← fit ONLY on X_train
  - TfidfVectorizer
  - ngram_range=(1,2) [word-level]
  - max_features=50000
  - Output: sparse matrix (n_samples × vocab_size)
        |
        v
[ Classifier ]
  Option A: MultinomialNB         ← baseline
  Option B: LogisticRegression    ← improved
  Option C: LinearSVC             ← improved
        |
        v
[ Predictions ]
  - Validation: F1, precision, recall, confusion matrix
  - Test: submission.csv for Kaggle leaderboard
```

## Data flow — train vs test

```
train.csv  ──→  split 80/20  ──→  X_train, y_train  ──→  fit(vectorizer + model)
                              ──→  X_val,   y_val    ──→  evaluate F1

test.csv   ──→  transform only (never fit)  ──→  predict  ──→  submission.csv
```

**Critical invariant:** `vectorizer.fit()` is called exactly once, on `X_train`. `vectorizer.transform()` is called on `X_val` and `X_test`. Fitting on the full dataset before splitting is data leakage — this architecture prevents it.

## sklearn Pipeline structure

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
    ('clf',   LogisticRegression(C=1.0, max_iter=1000))
])

pipeline.fit(X_train, y_train)      # vectorizer fit happens here, on train only
pipeline.predict(X_val)             # vectorizer transforms val here
pipeline.predict(X_test)            # vectorizer transforms test here
```

Using `Pipeline` enforces the no-leakage constraint structurally — when you call `pipeline.fit(X_train)`, only `X_train` is seen during `.fit()`. No manual management needed.

## Notebook structure

```
01_eda.ipynb
  └── Load data → inspect columns → class distribution → length EDA
      → keyword frequency → missing values

02_preprocessing.ipynb
  └── Write clean_text() function → test on samples
      → apply to full dataset → compare before/after

03_modeling_evaluation.ipynb
  └── Build Pipeline → train/val split → fit baseline
      → fit improved models → comparison table
      → confusion matrix → error analysis
      → generate submission.csv
```

## Feature engineering notes

The dataset has three potentially useful columns beyond `text`:

- `keyword` — 221 unique disaster-related keywords, ~0.8% missing. May be useful as an additional feature.
- `location` — user-entered free text, ~33% missing, noisy. Likely not worth the complexity for MVP.
- `text` — primary feature. All models start from text only.

Keyword as a feature will be tested in an experiment after the baseline is established.
