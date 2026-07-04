# Tech Stack — Disaster Tweet Classification

## Core ML / NLP

| Library      | Version | Why chosen                                                                                                         |
| ------------ | ------- | ------------------------------------------------------------------------------------------------------------------ |
| scikit-learn | ≥1.4    | TF-IDF, Pipeline, all classifiers, cross-validation, metrics — single library handles the full classical NLP stack |
| NLTK         | ≥3.8    | Stopword corpus + WordNetLemmatizer for text normalization                                                         |
| pandas       | ≥2.0    | DataFrame operations, EDA, loading CSVs                                                                            |
| numpy        | ≥1.26   | Array operations, sparse matrix handling                                                                           |

## Text Preprocessing

| Library                | Why chosen                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------ |
| `re` (stdlib)          | Regex for URL removal, mention stripping, punctuation cleanup — no external dependency needed          |
| NLTK stopwords         | Standard English stopword list; faster to use than building custom                                     |
| NLTK WordNetLemmatizer | Reduces words to base form — "fires" → "fire", "killed" → "kill" — better than stemming for short text |

## Visualization

| Library    | Why chosen                                                                        |
| ---------- | --------------------------------------------------------------------------------- |
| matplotlib | Base plotting — confusion matrix, distribution charts                             |
| seaborn    | Cleaner statistical plots with less code — class distributions, length histograms |
| wordcloud  | Visual EDA only — not used for modeling decisions, shown as supporting visual     |

## Environment

| Tool             | Why chosen                                                                      |
| ---------------- | ------------------------------------------------------------------------------- |
| Jupyter Notebook | Standard for ML EDA + iterative experimentation; Kaggle runs notebooks natively |
| Python 3.11      | Current stable; NLTK and sklearn both fully supported                           |

## Alternatives considered

- **CountVectorizer over TF-IDF:** Rejected as primary — TF-IDF suppresses common words that appear across all tweets (e.g. "the", "a") and amplifies discriminative terms. Will still be tested in an experiment for comparison.
- **spaCy over NLTK:** Rejected for this project — spaCy is more powerful but adds unnecessary complexity for lemmatization + stopwords. NLTK is sufficient and lighter.
- **XGBoost on sparse features:** Will be tested as an improvement experiment. Not baseline because XGBoost on sparse TF-IDF matrices is slower to tune than LinearSVC and rarely outperforms it on text classification.
- **HuggingFace Transformers (DistilBERT):** Out of scope for MVP — requires GPU for reasonable training time, and classical models establish the baseline understanding first.

## Known tradeoffs

- NLTK lemmatizer requires a `WordNet` download on first run (`nltk.download('wordnet')`) — documented in notebook setup cell
- TF-IDF produces sparse matrices — memory efficient but requires classifiers that handle sparse input (LinearSVC, LR, Naive Bayes all do; tree-based models need `.toarray()` which is expensive)
- No transformer fine-tuning means a ceiling around F1 ~0.83 for this dataset — acceptable for the learning goals of this project
