# Folder Structure — Disaster Tweet Classification

```
disaster-tweet-classification/
│
├── data/
│   ├── train.csv               # Kaggle training data — 7,613 tweets with labels
│   ├── test.csv                # Kaggle test data — 3,263 tweets, no labels
│   └── sample_submission.csv   # Kaggle submission format reference
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis — class dist, lengths, keywords
│   ├── 02_preprocessing.ipynb  # clean_text() function, before/after comparison
│   ├── 03_modeling_evaluation.ipynb  # Baseline models, comparison, error analysis, submission
│   └── 04_model_improvement.ipynb  # Improved models
│
├── docs/
│   ├── project_context.md      # Paste at top of every AI chat — 300 words max
│   ├── PRD.md                  # What we're building and why
│   ├── tech_stack.md           # Tech choices + reasoning
│   ├── architecture.md         # Pipeline design + data flow
│   ├── folder_structure.md     # This file
│   ├── tasks.md                # Atomic task list by day
│   ├── experiment_log.md       # Every training run logged
│   ├── model_card.md           # Final model documentation
│   ├── data_doc.md             # Dataset documentation
│   ├── eval.md                 # Evaluation methodology
│   ├── learnings.md            # Personal knowledge log (update every session)
│   └── debug_log.md            # Bug tracker
│
├── outputs/
│   ├── submission.csv          # Final Kaggle submission file
│   └── figures/                # Saved plots for README
│       ├── class_distribution.png
│       ├── tweet_length_dist.png
│       ├── top_keywords.png
│       ├── confusion_matrix.png
│       └── model_comparison.png
│
├── requirements.txt            # All dependencies with versions
└── README.md                   # Public-facing project documentation
```

## Why this structure

`data/` — Raw Kaggle files live here. Never modified after download. The notebooks read from here and produce cleaned outputs in memory — no derived data files stored on disk (keeps the repo clean, forces reproducibility through the notebook pipeline).

`notebooks/` — Split into three numbered notebooks with a single responsibility each. `01` is read-only analysis. `02` is the preprocessing function. `03` is all modeling. This separation means you can re-run `03` with different models without re-running EDA every time.

`docs/` — All docs in one flat folder. No subdirectories — the filenames are descriptive enough. Easy to paste `project_context.md` content without navigating.

`outputs/` — Separates generated artifacts (submission file, saved figures) from source code. `figures/` stores plots used in the README so they render on GitHub without referencing a running notebook.

## Naming conventions

- Notebooks: numbered prefix + descriptive snake_case (`01_eda.ipynb`)
- Python variables: snake_case (`clean_text`, `tfidf_vectorizer`, `X_train`)
- Functions: verb_noun format (`clean_text()`, `plot_class_distribution()`, `train_model()`)
- Constants: ALL_CAPS (`RANDOM_STATE = 42`, `MAX_FEATURES = 50000`)
- No magic numbers in code — all tunable values defined as named constants at the top of each notebook

## What NOT to commit

Add to `.gitignore`:

```
data/           # Kaggle data — too large, re-downloadable
outputs/submission.csv  # Optional — fine to commit if small
__pycache__/
.ipynb_checkpoints/
*.pyc
.env
```
