# Predicting Kickstarter Project Success

A reproduction and extension of **"Crowdfunding: Predicting Kickstarter Project Success"** (Khosla, Reinecke and Wittenbrink, Stanford). Given only what is known when a campaign launches (goal, category, location, duration, launch date and the one-sentence blurb), the code predicts whether the project will reach its funding goal.

The script reproduces the paper's pipeline, then goes beyond it: leak-free train/test splits, stronger features, hyperparameter tuning, CatBoost, SHAP and a 50-seed variance study.

## Results at a glance

All numbers are test accuracy on about 66,000 held-out projects (about 220,000 in total, 59.9% successful), from one run on the April 2019 to April 2021 Web Robots snapshots. Your numbers will differ slightly with a different snapshot.

| Experiment | Accuracy | AUC | Paper |
|---|---|---|---|
| Sample 1: metadata only, gradient boosting | 0.812 | 0.906 | 0.81 |
| Sample 2: + Naive Bayes blurb score | 0.821 | 0.914 | 0.83 |
| Sample 2, group split (relaunches kept apart) | 0.820 | 0.912 | n/a |
| Sample 2, time split (test on the latest 30%) | 0.818 | 0.883 | n/a |
| Sample 2 + Optuna tuning (30 trials) | 0.825 | 0.919 | n/a |
| Sample 2 + smoothed encoding, log goal, text shapes | 0.830 | 0.920 | n/a |
| Sample 6: + MiniLM sentence-embedding feature | 0.833 | 0.922 | n/a |
| Sample 2, CatBoost with native categoricals | 0.830 | 0.923 | n/a |

50-seed means (gradient boosting, paper settings): Sample 1 0.8110, Sample 2 0.8214, Sample 3 0.8216, Sample 4 0.8215, Sample 5 0.8258, with a seed-to-seed standard deviation of about 0.0014.

What the experiments show:
- The paper reproduces to within about a point. Sample 2 lands one point below its 0.83.
- Of the optional text features, only the Naive Bayes score clearly helps (about +1.0 point on all 50 seeds). Sentiment and LDA add nothing; Word2Vec adds about +0.4.
- Keeping near-duplicate relaunches apart costs only about 0.1 point.
- A time-ordered split keeps accuracy but drops AUC from 0.914 to 0.883 and recall on failed projects from 0.82 to 0.55, because the success rate rose from about 55% to 71% between the training and test periods.

## Repository contents

| File | Purpose |
|---|---|
| `kickstarter_pipeline.py` | The whole pipeline: loading, features, models, splits, tuning, plots, seed study |
| `requirements.txt` | Python packages |
| `data/` | Kickstarter CSVs (downloaded from webrobots.io) |
| `outputs/` | Created by the script: result tables, figures and feature caches |

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Python 3.10 or 3.11 is recommended; TensorFlow (used only for the neural net) lags behind the newest Python. If TensorFlow is missing, the neural net is skipped.
- For a lighter start, install only `pandas numpy scikit-learn lightgbm matplotlib`. The other packages are needed only for the options that use them:

| Package | Needed for |
|---|---|
| `nltk` (downloads the VADER lexicon on first use) | Sample 3 (sentiment) |
| `gensim` | Sample 5 (Word2Vec; about a 1.6 GB download, about 4 GB RAM) |
| `sentence-transformers` | Sample 6 (MiniLM embeddings) |
| `catboost` | `--models cat` |
| `optuna` | `--tune` |
| `shap` | `--shap` |
| `tensorflow` | `--models nn` |

Everything runs on CPU; no GPU is needed. Plan for 8 GB of RAM or more.

## 2. Get the data

1. Download the monthly snapshots from <https://webrobots.io/kickstarter-datasets/>, from **April 2019 to April 2021** (about 25 files).
2. Unzip them into `./data`. Nested folders are fine; the script searches recursively for `Kickstarter*.csv`.
3. Files outside `--min_date 2019-04 --max_date 2021-04` are skipped automatically, based on the date in the file or folder name.

## 3. Check that it works

```bash
python kickstarter_pipeline.py --data_dir ./data --sample 1 --models gb,logit,lasso
```

The first printed lines should report about **221,000 projects** (the paper has 221,248) and a success rate of about **0.6**. A much higher count means files outside the date window were included; a much lower one means files are missing.

## 4. Run the experiments

Every command starts with `python kickstarter_pipeline.py --data_dir ./data`. Results go to `./outputs`.

### 4.1 Reproduce the paper

| What | Add these flags | Compare with |
|---|---|---|
| Sample 1, metadata only | `--sample 1 --plots` | Paper Table 3 |
| Sample 2, + Naive Bayes | `--sample 2 --plots` | Paper Table 4 and Figure 1 |
| Sample 3, + sentiment | `--sample 3` | VADER replaces CoreNLP |
| Sample 4, + LDA topics | `--sample 4` | |
| Sample 5, + Word2Vec | `--sample 5` | |
| 50-seed box plots | `--seed_study 50 --samples 1,2,3,4,5` | Paper Figures 2 and 3 |

By default each run fits eight models: gradient boosting (LightGBM at the paper's settings), neural net, random forest, logistic regression, SVM, ridge, OLS and lasso. Lasso is deliberately **not standardised**, as the paper warns; it should land near 0.69 to 0.71.

### 4.2 Extensions

```bash
# Time split: train on the past, test on the latest 30% of launches
python kickstarter_pipeline.py --data_dir ./data --sample 2 --split time --plots

# Group split: relaunches (same name and blurb) never straddle train and test
python kickstarter_pipeline.py --data_dir ./data --sample 2 --split group

# Better encodings and cheap text features
python kickstarter_pipeline.py --data_dir ./data --sample 2 --smooth 20 --log_goal --extra_text

# Sentence-embedding text feature (the first run encodes every blurb; slow on CPU, then cached)
python kickstarter_pipeline.py --data_dir ./data --sample 6 --smooth 20 --log_goal --extra_text

# CatBoost with native categoricals, next to LightGBM
python kickstarter_pipeline.py --data_dir ./data --sample 2 --models gb,cat

# Hyperparameter tuning (30 Optuna trials) with a tuned decision threshold
python kickstarter_pipeline.py --data_dir ./data --sample 2 --models gb --tune 30 --tune_threshold --plots

# Feature importance and SHAP summary (add --tune 30 to explain the tuned model instead)
python kickstarter_pipeline.py --data_dir ./data --sample 2 --models gb --plots --shap
```

### 4.3 The 50-seed study
Fitting LightGBM at the paper's settings on 150,000 rows many times is the slow part. Three options make it manageable:

- **Caching.** Sentiment, LDA, Word2Vec and embedding features are computed once and saved as `outputs/cache_*.npy`.
- **`--resume`** continues from the saved CSV and skips seed/sample pairs already finished. Results are saved after every fit, so you can stop with Ctrl+C at any time.
- **`--fast`** uses 150 smaller trees instead of 500 large ones. This is fine for comparing samples with each other but does not match the paper's exact numbers. Fast results are saved under a separate `_fast` file name.

```bash
# Quick look: 10 seeds, fast model
python kickstarter_pipeline.py --data_dir ./data --seed_study 10 --samples 1,2,3 --fast

# Full study at the paper's settings, resumable
python kickstarter_pipeline.py --data_dir ./data --seed_study 50 --samples 1,2,3,4,5 --resume

# Redraw the plot from whatever has finished (no data loading; use the same flags as the run)
python kickstarter_pipeline.py --out_dir ./outputs --plot_only --samples 1,2,3,4,5
```

When resuming or plotting, pass the same `--split`, `--smooth`, `--log_goal`, `--extra_text` and `--fast` flags as the original run, because they decide which CSV is used.

## 5. Outputs

| File | Contents |
|---|---|
| `results_sample{N}_{split}[_flags].csv` | One row per model: accuracy, F1, precision and recall per class, ROC-AUC, PR-AUC, LogLoss |
| `confusion_*.png` | Confusion-matrix heatmaps (`--plots`) |
| `importance_*.csv`, `importance_*.png` | LightGBM split-gain feature importance (`--plots`) |
| `shap_*.png` | SHAP summary plot (`--shap`) |
| `seed_study_*.csv`, `seed_study_*.png` | Per-seed accuracy and F1 and the box plots |
| `cache_*.npy` | Cached per-project text features |

**Output files can be overwritten.** The result file name contains the sample, split and feature flags, but not the model list or the tuning flags. Running `--models gb,cat` after a default run, or `--tune 30` after an untuned one, overwrites the earlier CSV. Copy or rename a result file before running a variant that shares its name, or use a different `--out_dir` for each experiment.

## 6. How it works

1. **Load and clean.** Reads only the 12 needed columns, drops rows sharing name, blurb, launch date and deadline (the snapshots are cumulative), and keeps completed projects (successful or failed).
2. **Metadata features.** For each categorical field (currency, country, category, parent category, location, location type), an out-of-fold 5-fold success rate and mean goal, plus the project's goal minus that mean. Levels with fewer than 1,000 projects fall back to the global mean (`--smooth M` replaces this with a smoothed estimate). Category and parent category are also one-hot encoded.
3. **Text feature.** A Multinomial Naive Bayes classifier on tf-idf unigrams turns the blurb into one out-of-fold success probability. Samples 3 to 5 add sentiment, LDA topics and Word2Vec averages; Sample 6 adds a MiniLM embedding.
4. **Models.** Paper hyperparameters, balanced class weights and a 0.5 threshold. Out-of-fold encodings keep a project's own label out of its features.
5. **Splits.** `random` is the paper's stratified 70/30 split; `group` and `time` are stricter.

| Sample | Features |
|---|---|
| 1 | Metadata |
| 2 | Metadata + Naive Bayes |
| 3 | + VADER sentiment |
| 4 | + 20-topic LDA |
| 5 | + Word2Vec averages |
| 6 | Sample 2 + MiniLM embedding feature |

## 7. Command-line reference

| Flag | Meaning | Default |
|---|---|---|
| `--data_dir`, `--out_dir` | Input and output folders | `./data`, `./outputs` |
| `--min_date`, `--max_date` | Snapshot window (YYYY-MM) | `2019-04`, `2021-04` |
| `--sample 1-6` | Feature set (table above) | `2` |
| `--split random\|group\|time` | Train/test split | `random` |
| `--models` | Comma list from `gb,rf,logit,svm,ridge,ols,lasso,nn,cat` | all except `cat` |
| `--smooth M` | Smoothed target encoding | off |
| `--log_goal` | Use log(1 + goal) | off |
| `--extra_text` | Cheap name/blurb features (digits, `!`, `?`, capitals, word count) | off |
| `--tune N` | Optuna trials for LightGBM | 0 (paper settings) |
| `--tune_threshold` | Pick the threshold on out-of-fold predictions | off |
| `--plots`, `--shap` | Confusion matrix, importance and SHAP figures | off |
| `--seed_study N` | Run N random splits, saving box plots | 0 |
| `--samples` | Samples to include in the seed study | `1,2,3,4` |
| `--resume`, `--fast`, `--plot_only` | Seed-study helpers (section 4.3) | off |
| `--nn_batch` | Neural net batch size (paper: 3, very slow) | 256 |
| `--seed` | Random seed | 42 |
| `--dedup paper\|id` | Duplicate-removal rule | `paper` |

Run `python kickstarter_pipeline.py --help` for the full list.

## 8. Known differences from the paper

- **Data snapshot.** Web Robots snapshots differ slightly from the paper's, so the project count and metrics shift a little.
- **Sentiment.** VADER replaces Stanford CoreNLP.
- **Neural net.** Batch size 256 instead of 3, for speed.
- **LDA** is fitted on all blurbs (no labels involved) so its output can be cached.

## Reference

Khosla, Reinecke and Wittenbrink. *Crowdfunding: Predicting Kickstarter Project Success.* Stanford University.
