# PRD — Disaster Tweet Classification

## Problem statement

People use Twitter to announce emergencies in real time. But the same language used to describe real disasters is constantly used figuratively — "this traffic is a disaster", "the concert was fire", "I'm dying of boredom". An automated classifier that can distinguish real disaster tweets from figurative ones would allow emergency response teams, news aggregators, and crisis monitoring systems to filter signal from noise at scale.

This is Kaggle competition: [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview).

**Who has this problem:** Emergency response organizations, crisis monitoring platforms, news aggregation services, social media analytics teams.

**Why it matters:** In a real disaster, Twitter volume spikes. Manual filtering is impossible. An ~80%+ F1 classifier can meaningfully reduce noise for human reviewers even if it isn't perfect.

## Target users

For portfolio purposes: ML engineering recruiters evaluating NLP fundamentals.
For the problem itself: Crisis monitoring teams who need to triage high-volume tweet streams during emergencies.

## Core features (MVP)

- [x] Exploratory Data Analysis — class distribution, tweet length, keyword frequency, location coverage
- [ ] Text preprocessing pipeline — URL removal, mention removal, punctuation handling, lowercasing, stopword removal, lemmatization
- [ ] TF-IDF vectorization pipeline (fit on train only — no leakage)
- [ ] Baseline model — Multinomial Naive Bayes with unigram TF-IDF
- [ ] Improved models — Logistic Regression, LinearSVC with tuned n-gram ranges
- [ ] Model comparison table — F1, precision, recall, accuracy across all models
- [ ] Error analysis — manual review of false positives and false negatives
- [ ] Kaggle submission file (`submission.csv`)
- [ ] Polished README

## Nice-to-have (post-MVP)

- [ ] Character-level n-gram TF-IDF as additional feature set
- [ ] `keyword` column as an additional engineered feature
- [ ] Ensemble: soft-vote between LR + LinearSVC
- [ ] Sentence embeddings (sentence-transformers) as alternative to TF-IDF
- [ ] DistilBERT fine-tuning — stretch goal only if time permits after MVP is clean

## Non-goals

- This project will NOT fine-tune a large transformer as its primary deliverable
- This project will NOT build an API or web interface
- This project will NOT use external tweet data beyond the Kaggle dataset
- This project will NOT use GPU or cloud compute

## Success metrics

- Primary: F1 score ≥ 0.79 on Kaggle public leaderboard
- Secondary: At least 3 models compared with a clear analysis of tradeoffs
- Tertiary: Error analysis section exists with manually reviewed examples — not just confusion matrix numbers
- Process: TF-IDF fitted only on training split (no leakage — verifiable in notebook)

## Constraints

- Time: 5–7 days
- Cost: Free tier only (Kaggle free GPU if needed, but CPU is sufficient for classical NLP)
- Tech: Python + scikit-learn stack only for MVP
