# Disaster Tweet Classification

This project predicts whether a tweet is about a real disaster or not.

I built it as a Kaggle NLP project to practice the full workflow properly: EDA, preprocessing, baseline modeling, model comparison, error analysis, tuning, and final submission. The goal was not just to get a score, but to understand why the model works, where it breaks, and how to improve it in a structured way.

## Problem statement

When a real emergency happens, social media fills up fast. Some tweets are genuine updates about disasters. A lot of them are jokes, metaphors, opinions, or unrelated chatter. The task here is to classify tweets into:

- `1` = disaster tweet
- `0` = not a disaster tweet

This is a binary text classification problem from Kaggle's **NLP with Disaster Tweets** competition.

## Dataset

- Competition: [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- Source: Kaggle
- Training rows: 7,613
- Main feature used in this version: cleaned `text`
- Target column: `target`

The dataset also includes optional fields like `keyword` and `location`, but the final version of this project uses the cleaned tweet text as the core input feature.

## Project workflow

### Day 1 - Exploratory data analysis

The first step was understanding the data before touching the model. This included class balance, missing values, duplicate tweets, keyword patterns, and tweet length distributions.

The classes are not perfectly balanced, which is why F1 score matters more than accuracy for this competition.

### Day 2 - Text preprocessing

The raw tweets were cleaned using a simple but effective preprocessing pipeline:

- Lowercasing
- URL removal
- Mention removal
- Special character cleanup
- Tokenization
- Stopword removal
- Lemmatization

The cleaned text was then used for TF-IDF feature extraction.

### Day 3 and Day 4 - Baselines and model comparison

I started with a Multinomial Naive Bayes baseline and then compared multiple classical NLP models:

- MultinomialNB + unigram TF-IDF
- Logistic Regression + unigram TF-IDF
- Logistic Regression + bigram TF-IDF
- LinearSVC + bigram TF-IDF
- Logistic Regression + character TF-IDF

## Results

| Model                                | Validation F1 |  Precision |     Recall |   Accuracy |
| ------------------------------------ | ------------: | ---------: | ---------: | ---------: |
| MultinomialNB + unigram TF-IDF       |        0.7732 |     0.8540 |     0.7064 |     0.8221 |
| Logistic Regression + unigram TF-IDF |        0.7769 |     0.8453 |     0.7187 |     0.8227 |
| Logistic Regression + bigram TF-IDF  |        0.7649 |     0.8447 |     0.6988 |     0.8155 |
| LinearSVC + bigram TF-IDF            |        0.7750 |     0.7923 |     0.7584 |     0.8109 |
| **Final tuned LinearSVC**            |    **0.7836** | **0.8445** | **0.7309** | **0.8267** |

**Kaggle public leaderboard score:** **0.79619**

### Final candidate comparison

The final selection came down to tuned Logistic Regression vs tuned LinearSVC. LinearSVC finished slightly ahead on validation F1, so it became the final submission model.

## Day 5 - Error analysis

This was the most useful part of the project.

Looking at wrong predictions made it clear that false positives and false negatives had different patterns:

- False positives often had strong disaster words like `fire`, `burning`, `storm`, or `nuclear`, but the tweet was not actually describing a real disaster.
- False negatives were often short, vague, conversational, or dependent on context.

That pattern explained why some models looked decent on paper but still made predictable mistakes.

## Day 6 - Tuning and final model

The final model was:

- **LinearSVC + TF-IDF (1,2)**

I tuned the strongest candidates, compared them on validation F1, and then retrained the final model on the full training set before generating the Kaggle submission.

The final confusion matrix shows a solid balance, but it also makes one thing obvious: the model still misses a meaningful number of real disaster tweets. That is the main gap left in the classical NLP approach.

## What this project taught me

A better score did not come from randomly trying more models. It came from being systematic:

- build a baseline first
- compare models properly
- inspect the errors
- tune only the strongest candidates
- validate the final choice with both metrics and examples

That workflow matters more than any single model choice.

## Repository structure

```bash
disaster-tweet-classification/
│
├── data/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling_evaluation.ipynb
│   ├── 04_model_improvement.ipynb
│   ├── 05_error_analysis.ipynb
│   └── 06_tuning_submission.ipynb
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── submission.csv
│
├── docs/
│   ├── eval.md
│   ├── model_card.md
│   ├── experiment_log.md
│   └── data_doc.md
│
└── README.md
```

## Tech stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- NLTK

## How to run

1. Clone the repository.
2. Download the competition data from Kaggle.
3. Place the raw files in the `data/` folder.
4. Run the notebooks in order from `01_eda.ipynb` to `06_tuning_submission.ipynb`.

## Future improvements

If this project is extended further, the next steps would be:

- using `keyword` as an additional feature
- combining word and character TF-IDF
- trying a soft ensemble
- testing DistilBERT or another transformer model

## Links

- GitHub repo: [Heart Disease Detection](https://github.com/KaustubhMukdam/heart-disease-predictor)
- GitHub repo: [Teleco Customer Churn](https://github.com/KaustubhMukdam/customer-churn-prediction)
- GitHub repo: [Retail Sales Forecasting](https://github.com/KaustubhMukdam/retail-sales-forecasting)

## Author

**Kaustubh Mukdam**
