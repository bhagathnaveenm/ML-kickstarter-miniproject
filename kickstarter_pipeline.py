"""
Kickstarter project-success prediction -- improved reproduction (v2)
Based on "Crowdfunding: Predicting Kickstarter Project Success" (Khosla, Reinecke, Wittenbrink).

What is new compared with the 1:1 script
  * Lasso no longer standardised (the paper says it collapses when standardised)
  * usecols / low_memory loading, recursive file discovery, date-window filter, id-duplicate report
  * --split random | time | group      (random = paper; time/group = realistic, leak-free)
  * --smooth M       smoothed (m-estimate) target encoding instead of the hard 1000 cutoff
  * --log_goal       log1p(goal) for goal features
  * --extra_text     cheap blurb/name features (digits, !, ?, caps ratio, word count)
  * --sample 6       Metadata + NB + sentence-embedding (SBERT) OOF logistic feature
  * --models ... cat CatBoost with native categoricals (ablation vs. hand-built encodings)
  * --tune N         Optuna tuning for LightGBM
  * --tune_threshold choose decision threshold on out-of-fold predictions
  * extra metrics    ROC-AUC, PR-AUC, LogLoss
  * --plots [--shap] confusion-matrix heatmap, LightGBM importance, SHAP summary
  * --seed_study N   50-seed box plots (paper Figures 2 and 3)

Examples
  python kickstarter_pipeline_v2.py --data_dir ./data --sample 1
  python kickstarter_pipeline_v2.py --data_dir ./data --sample 2 --plots
  python kickstarter_pipeline_v2.py --data_dir ./data --sample 2 --split time --smooth 20 --log_goal --extra_text
  python kickstarter_pipeline_v2.py --data_dir ./data --seed_study 50 --samples 1,2,3,4
"""
import argparse
import functools
import glob
import json
import os
import re
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import LatentDirichletAllocation  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GroupShuffleSplit, KFold, cross_val_predict,  # noqa: E402
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

warnings.filterwarnings("ignore")

SEED = 42
MIN_COUNT = 1000
CAT_COLS = ["currency", "country", "category_name", "parent_category",
            "location", "location_type"]
USECOLS = ["id", "name", "blurb", "goal", "static_usd_rate", "state", "launched_at",
           "deadline", "category", "location", "currency", "country"]
SAMPLE_NAMES = {1: "Metadata", 2: "Metadata + NB", 3: "Metadata + NB + Sentiment",
                4: "Metadata + NB + LDA", 5: "Metadata + NB + W2V",
                6: "Metadata + NB + SBERT"}
MODEL_KEYS = {"gb": "G. Boosting", "rf": "R. Forest", "logit": "Logistic", "svm": "SVM",
              "ridge": "Ridge", "ols": "OLS", "lasso": "Lasso", "nn": "Neural Net",
              "cat": "CatBoost"}
# NOTE: Lasso deliberately NOT scaled (paper: it "suffered tremendously" when standardised)
SCALERS = {"logit": StandardScaler, "ridge": StandardScaler, "ols": StandardScaler,
           "svm": MinMaxScaler}


# --------------------------------------------------------------------------
# 1. Data loading
# --------------------------------------------------------------------------
def _js(s):
    try:
        return json.loads(s)
    except Exception:
        return {}


def _find_files(data_dir, min_date, max_date):
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "Kickstarter*.csv"), recursive=True))
    keep = []
    for p in paths:
        m = re.search(r"(\d{4}-\d{2})-\d{2}", p)
        if m and not (min_date <= m.group(1) <= max_date):
            continue
        keep.append(p)
    return keep


def load_data(args):
    files = _find_files(args.data_dir, args.min_date, args.max_date)
    assert files, f"No Kickstarter*.csv found under {args.data_dir} for {args.min_date}..{args.max_date}"
    print(f"Reading {len(files)} csv files")
    parts = [pd.read_csv(f, usecols=lambda c: c in USECOLS, low_memory=False) for f in files]
    df = pd.concat(parts, ignore_index=True)
    del parts

    df = df[df["state"].isin(["successful", "failed"])]
    df = df.dropna(subset=["goal", "static_usd_rate", "launched_at", "deadline"])
    if "id" in df.columns:
        print(f"  rows sharing a project id: {df['id'].duplicated().sum():,}")
    if args.dedup == "id" and "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["name", "blurb", "launched_at", "deadline"])
    df = df.reset_index(drop=True)

    cat = df["category"].map(_js)
    loc = df["location"].map(_js)
    df["category_name"] = cat.map(lambda d: d.get("name", "NA"))
    df["parent_category"] = cat.map(lambda d: d.get("parent_name", d.get("name", "NA")))
    df["location"] = loc.map(lambda d: d.get("name", "NA"))
    df["location_type"] = loc.map(lambda d: d.get("type", "NA"))
    df["currency"] = df["currency"].fillna("NA").astype(str)
    df["country"] = df["country"].fillna("NA").astype(str)

    df["blurb"] = df["blurb"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]
    df["y"] = (df["state"] == "successful").astype(int)

    launched = pd.to_datetime(df["launched_at"], unit="s")
    deadline = pd.to_datetime(df["deadline"], unit="s")
    df["launched_ts"] = df["launched_at"].astype(np.int64)
    df["duration_days"] = (deadline - launched).dt.total_seconds() / 86400
    df["launch_year"] = launched.dt.year
    df["launch_month"] = launched.dt.month
    df["launch_dow"] = launched.dt.dayofweek
    df["blurb_len"] = df["blurb"].str.len()
    df["name_len"] = df["name"].str.len()
    # relaunches share name+blurb: used by the group-aware split
    df["group"] = pd.factorize(df["name"] + "||" + df["blurb"])[0]
    df["_row"] = np.arange(len(df))
    df = df.drop(columns=["category", "state"])
    print(f"{len(df):,} projects (paper: 221,248); success rate {df['y'].mean():.3f}")
    return df


def make_split(df, mode, seed):
    if mode == "random":
        tr, te = train_test_split(df, test_size=0.30, random_state=seed, stratify=df["y"])
    elif mode == "time":
        d = df.sort_values("launched_ts", kind="stable")
        cut = int(0.7 * len(d))
        tr, te = d.iloc[:cut], d.iloc[cut:]
    elif mode == "group":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        a, b = next(gss.split(df, groups=df["group"]))
        tr, te = df.iloc[a], df.iloc[b]
    else:
        raise ValueError(mode)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    print(f"split={mode}: train {len(tr):,} (succ {tr['y'].mean():.3f}) | "
          f"test {len(te):,} (succ {te['y'].mean():.3f})")
    return tr, te


# --------------------------------------------------------------------------
# 2. Categorical / metadata encodings (all out-of-fold)
# --------------------------------------------------------------------------
def oof_group_mean(tr_key, tr_val, te_key, n_splits=5, min_count=MIN_COUNT,
                   smooth=None, seed=SEED):
    """OOF mean of tr_val per level. Rare levels -> global mean (hard cutoff),
    or, if smooth=M, m-estimate shrinkage: (n*mean + M*global) / (n + M)."""
    tr_key, tr_val, te_key = (pd.Series(s).reset_index(drop=True) for s in (tr_key, tr_val, te_key))

    def fit(k, v):
        g = pd.DataFrame({"k": k, "v": v}).groupby("k")["v"].agg(["mean", "count"])
        gm = v.mean()
        if smooth:
            enc = (g["count"] * g["mean"] + smooth * gm) / (g["count"] + smooth)
        else:
            enc = g.loc[g["count"] >= min_count, "mean"]
        return enc, gm

    oof = np.zeros(len(tr_key))
    for a, b in KFold(n_splits, shuffle=True, random_state=seed).split(tr_key):
        m, gm = fit(tr_key.iloc[a], tr_val.iloc[a])
        oof[b] = tr_key.iloc[b].map(m).fillna(gm).values
    m, gm = fit(tr_key, tr_val)
    return oof, te_key.map(m).fillna(gm).values


def build_metadata(train, test, args, seed=SEED):
    g_tr = pd.Series(np.log1p(train["goal_usd"]) if args.log_goal else train["goal_usd"]).reset_index(drop=True)
    g_te = pd.Series(np.log1p(test["goal_usd"]) if args.log_goal else test["goal_usd"]).reset_index(drop=True)
    num = ["blurb_len", "name_len", "duration_days", "launch_year", "launch_month", "launch_dow"]
    Xtr = train[num].reset_index(drop=True).copy()
    Xte = test[num].reset_index(drop=True).copy()
    Xtr["goal"], Xte["goal"] = g_tr, g_te

    for c in CAT_COLS:
        a, b = oof_group_mean(train[c], train["y"], test[c], smooth=args.smooth, seed=seed)
        Xtr[f"{c}_te"], Xte[f"{c}_te"] = a, b
        a, b = oof_group_mean(train[c], g_tr, test[c], smooth=args.smooth, seed=seed)
        Xtr[f"{c}_goal_te"], Xte[f"{c}_goal_te"] = a, b
        Xtr[f"{c}_goal_diff"] = g_tr.values - a
        Xte[f"{c}_goal_diff"] = g_te.values - b

    oh_cols = ["category_name", "parent_category"]
    oh = pd.get_dummies(pd.concat([train[oh_cols], test[oh_cols]], ignore_index=True), dtype=int)
    Xtr = pd.concat([Xtr, oh.iloc[:len(train)].reset_index(drop=True)], axis=1)
    Xte = pd.concat([Xte, oh.iloc[len(train):].reset_index(drop=True)], axis=1)
    return Xtr, Xte


def text_extras(df):
    out = pd.DataFrame(index=range(len(df)))
    for col in ("name", "blurb"):
        s = df[col].reset_index(drop=True)
        out[f"{col}_digits"] = s.str.count(r"\d")
        out[f"{col}_excl"] = s.str.count("!")
        out[f"{col}_quest"] = s.str.count(r"\?")
        out[f"{col}_upper_ratio"] = s.str.count(r"[A-Z]") / s.str.len().clip(lower=1)
        out[f"{col}_words"] = s.str.split().str.len().fillna(0)
    return out


# --------------------------------------------------------------------------
# 3. Text encodings
# --------------------------------------------------------------------------
def nb_feature(train, test):
    tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
    Atr = tfidf.fit_transform(train["blurb"])
    Ate = tfidf.transform(test["blurb"])
    nb = MultinomialNB()
    oof = cross_val_predict(nb, Atr, train["y"], cv=5, method="predict_proba")[:, 1]
    nb.fit(Atr, train["y"])
    return oof, nb.predict_proba(Ate)[:, 1]


class RowCache:
    """Per-row features that depend on neither the labels nor the split (VADER sentiment, LDA
    topics, Word2Vec averages, sentence embeddings). Computed once for the whole dataset,
    saved to disk, and reused by every seed and every later run.
    NOTE: LDA is fitted on all blurbs (no labels involved) instead of train only; this is what
    makes it cacheable and has no label leakage."""

    def __init__(self, df, out_dir):
        self.df, self.dir, self.mem = df, out_dir, {}
        h = int(pd.util.hash_pandas_object(df["blurb"], index=False).sum()) % 10 ** 10
        self.key = f"{len(df)}_{h}"

    def get(self, name):
        if name in self.mem:
            return self.mem[name]
        path = os.path.join(self.dir, f"cache_{name}_{self.key}.npy")
        if os.path.exists(path):
            print(f"loading cached {name} features")
            arr = np.load(path)
        else:
            print(f"computing {name} features for {len(self.df):,} rows (cached afterwards)")
            arr = getattr(self, f"_{name}")(self.df["blurb"].tolist())
            np.save(path, arr)
        self.mem[name] = arr
        return arr

    def _sentiment(self, texts):
        """VADER compound score averaged over sentences (paper used CoreNLP)."""
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

        def score(text):
            sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()] or [text]
            return np.mean([sia.polarity_scores(s)["compound"] for s in sents])

        return np.array([score(t) for t in texts], dtype=np.float32)

    def _lda(self, texts, n_topics=20):
        cv = CountVectorizer(stop_words="english", min_df=10, max_df=0.35,
                             token_pattern=r"(?u)\b[a-zA-Z]{2,}\b")
        C = cv.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, learning_method="online",
                                        random_state=SEED, n_jobs=-1)
        return lda.fit_transform(C).astype(np.float32)

    def _w2v(self, texts):
        kv = _load_w2v()
        tok = re.compile(r"[a-zA-Z]+")

        def embed(text):
            vecs = [kv[w] for w in tok.findall(text) if w in kv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(300)

        return np.vstack([embed(t) for t in texts]).astype(np.float32)

    def _emb(self, texts, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        return m.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True,
                        normalize_embeddings=True).astype(np.float32)


@functools.lru_cache(maxsize=1)
def _load_w2v():
    import gensim.downloader as api
    return api.load("word2vec-google-news-300")  # ~1.6 GB download, ~4 GB RAM


def emb_feature(train, test, emb):
    """Logistic regression on sentence embeddings -> OOF P(success); same stacking idea as NB."""
    Etr, Ete = emb[train["_row"].values], emb[test["_row"].values]
    lr = LogisticRegression(C=1.0, max_iter=1000)
    oof = cross_val_predict(lr, Etr, train["y"].values, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
    lr.fit(Etr, train["y"].values)
    return oof, lr.predict_proba(Ete)[:, 1]


def clean_cols(Xtr, Xte):
    """LightGBM rejects feature names with special characters (category names have them)."""
    seen, new = {}, []
    for c in Xtr.columns:
        n = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        if n in seen:
            seen[n] += 1
            n = f"{n}_{seen[n]}"
        else:
            seen[n] = 0
        new.append(n)
    Xtr.columns = new
    Xte.columns = new
    return Xtr, Xte


def _add_cols(X, arr, prefix):
    return pd.concat([X, pd.DataFrame(arr, columns=[f"{prefix}_{i}" for i in range(arr.shape[1])])], axis=1)


def build_features_multi(train, test, samples, args, cache, seed=SEED):
    """Feature matrices for several samples at once. The expensive split-dependent parts
    (metadata encodings, text extras, Naive Bayes) are computed ONCE and shared."""
    Btr, Bte = build_metadata(train, test, args, seed)
    if args.extra_text:
        Btr = pd.concat([Btr, text_extras(train)], axis=1)
        Bte = pd.concat([Bte, text_extras(test)], axis=1)
    if any(s >= 2 for s in samples):
        nb_tr, nb_te = nb_feature(train, test)
    rt, re_ = train["_row"].values, test["_row"].values
    out = {}
    for s in samples:
        Xtr, Xte = Btr.copy(), Bte.copy()
        if s >= 2:
            Xtr["nb"], Xte["nb"] = nb_tr, nb_te
        if s == 3:
            arr = cache.get("sentiment")
            Xtr["sentiment"], Xte["sentiment"] = arr[rt], arr[re_]
        elif s == 4:
            arr = cache.get("lda")
            Xtr, Xte = _add_cols(Xtr, arr[rt], "lda"), _add_cols(Xte, arr[re_], "lda")
        elif s == 5:
            arr = cache.get("w2v")
            Xtr, Xte = _add_cols(Xtr, arr[rt], "w2v"), _add_cols(Xte, arr[re_], "w2v")
        elif s == 6:
            Xtr["emb_lr"], Xte["emb_lr"] = emb_feature(train, test, cache.get("emb"))
        out[s] = clean_cols(Xtr, Xte)
    return out


def build_features(train, test, sample, args, cache, seed=SEED):
    return build_features_multi(train, test, [sample], args, cache, seed)[sample]


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def lgbm_params(override=None):
    p = dict(n_estimators=500, learning_rate=0.05, max_depth=55, num_leaves=400,
             subsample=0.75, subsample_freq=1, colsample_bytree=0.2, max_bin=500,
             class_weight="balanced", n_jobs=-1, random_state=SEED, verbose=-1)
    p.update(override or {})
    return p


def tune_lgbm(Xtr, ytr, n_trials):
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    idx_a, idx_b = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=SEED, stratify=ytr)

    def objective(t):
        p = dict(n_estimators=t.suggest_int("n_estimators", 300, 1200),
                 learning_rate=t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                 num_leaves=t.suggest_int("num_leaves", 31, 512, log=True),
                 max_depth=t.suggest_int("max_depth", 6, 60),
                 min_child_samples=t.suggest_int("min_child_samples", 10, 200),
                 subsample=t.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=t.suggest_float("colsample_bytree", 0.1, 0.8),
                 reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
                 max_bin=255)
        m = lgb.LGBMClassifier(**lgbm_params(p))
        m.fit(Xtr.iloc[idx_a], ytr[idx_a])
        return roc_auc_score(ytr[idx_b], m.predict_proba(Xtr.iloc[idx_b])[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"Optuna best AUC {study.best_value:.4f} with {study.best_params}")
    return study.best_params


def make_sklearn_models():
    return {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=55, min_samples_leaf=10,
                                     max_features=15, criterion="gini", class_weight="balanced",
                                     n_jobs=-1, random_state=SEED),
        "logit": LogisticRegression(penalty="l2", C=0.1, class_weight="balanced", max_iter=2000),
        "svm": LinearSVC(loss="squared_hinge", penalty="l2", C=1, class_weight="balanced",
                         max_iter=5000),
        "ridge": Ridge(alpha=0.75),
        "ols": LinearRegression(),
        "lasso": Lasso(alpha=0.75),
    }


def build_nn(n_features):
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(25, activation="selu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_catboost(train, test, Xtr, Xte, ytr, sw):
    """CatBoost with native categoricals: skips the hand-built target/goal encodings."""
    from catboost import CatBoostClassifier
    keep = [c for c in Xtr.columns
            if not (c.endswith(("_te", "_goal_te", "_goal_diff"))
                    or c.startswith(("category_name_", "parent_category_")))]
    Ctr = pd.concat([Xtr[keep], train[CAT_COLS].reset_index(drop=True)], axis=1)
    Cte = pd.concat([Xte[keep], test[CAT_COLS].reset_index(drop=True)], axis=1)
    m = CatBoostClassifier(iterations=800, learning_rate=0.08, depth=8,
                           auto_class_weights="Balanced", random_seed=SEED, verbose=0,
                           cat_features=CAT_COLS, thread_count=-1)
    m.fit(Ctr, ytr)
    return m.predict_proba(Cte)[:, 1]


# --------------------------------------------------------------------------
# 5. Evaluation / plots
# --------------------------------------------------------------------------
def report(name, y, pred, proba=None, score=None):
    s = proba if proba is not None else score
    r = {"Model": name,
         "A": accuracy_score(y, pred), "F1": f1_score(y, pred),
         "P1": precision_score(y, pred, pos_label=1), "P0": precision_score(y, pred, pos_label=0),
         "R1": recall_score(y, pred, pos_label=1), "R0": recall_score(y, pred, pos_label=0)}
    if s is not None:
        r["AUC"] = roc_auc_score(y, s)
        r["PR_AUC"] = average_precision_score(y, s)
    if proba is not None:  # LogLoss is on the class-balanced scale (models are re-weighted)
        r["LogLoss"] = log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))
    return r


def best_threshold(model_params, Xtr, ytr):
    import lightgbm as lgb
    oof = cross_val_predict(lgb.LGBMClassifier(**model_params), Xtr, ytr, cv=3,
                            method="predict_proba")[:, 1]
    grid = np.arange(0.30, 0.71, 0.01)
    scores = [balanced_accuracy_score(ytr, (oof >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def plot_confusion(y, pred, path):
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted outcome"); ax.set_ylabel("True outcome")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_importance(model, cols, path, top=25):
    imp = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    imp.to_csv(path.replace(".png", ".csv"))
    fig, ax = plt.subplots(figsize=(7, 7))
    imp.head(top)[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Total gain"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_shap(model, X, path, n=5000):
    try:
        import shap
    except ImportError:
        print("shap not installed - skipping SHAP plot")
        return
    samp = X.sample(min(n, len(X)), random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(samp)
    if isinstance(sv, list):
        sv = sv[1]
    if getattr(sv, "ndim", 2) == 3:
        sv = sv[..., 1]
    shap.summary_plot(sv, samp, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# --------------------------------------------------------------------------
# 6. Single run
# --------------------------------------------------------------------------
def run_models(train, test, Xtr, Xte, args):
    ytr, yte = train["y"].values, test["y"].values
    Xtr_v, Xte_v = Xtr.values.astype(float), Xte.values.astype(float)
    sw = compute_sample_weight("balanced", ytr)
    sk = make_sklearn_models()
    rows, gb_model, gb_pred = [], None, None
    lgb_over = tune_lgbm(Xtr, ytr, args.tune) if (args.tune and "gb" in args.models) else None

    for key in args.models:
        name, t0, proba, score = MODEL_KEYS[key], time.time(), None, None
        if key == "gb":
            import lightgbm as lgb
            gb_model = lgb.LGBMClassifier(**lgbm_params(lgb_over)).fit(Xtr, ytr)
            proba = gb_model.predict_proba(Xte)[:, 1]
            pred = gb_pred = (proba >= 0.5).astype(int)
            rows.append(report(name, yte, pred, proba=proba))
            if args.tune_threshold:
                thr = best_threshold(lgbm_params(lgb_over), Xtr, ytr)
                rows.append(report(f"{name} (thr={thr:.2f})", yte, (proba >= thr).astype(int), proba=proba))
            print(f"done: {name} ({time.time() - t0:.0f}s)")
            continue
        if key in sk:
            A, B = Xtr_v, Xte_v
            if key in SCALERS:
                sc = SCALERS[key]().fit(Xtr_v)
                A, B = sc.transform(Xtr_v), sc.transform(Xte_v)
            model = sk[key]
            if key in ("ridge", "ols", "lasso"):
                model.fit(A, ytr, sample_weight=sw)
                score = model.predict(B)
                pred = (score >= 0.5).astype(int)
            else:
                model.fit(A, ytr)
                pred = model.predict(B)
                if key in ("rf", "logit"):
                    proba = model.predict_proba(B)[:, 1]
                else:
                    score = model.decision_function(B)
        elif key == "nn":
            try:
                sc = StandardScaler().fit(Xtr_v)
                nn = build_nn(Xtr.shape[1])
                cw = {0: sw[ytr == 0][0], 1: sw[ytr == 1][0]}
                nn.fit(sc.transform(Xtr_v), ytr, epochs=30, batch_size=args.nn_batch,
                       class_weight=cw, verbose=0)
                proba = nn.predict(sc.transform(Xte_v), verbose=0).ravel()
                pred = (proba >= 0.5).astype(int)
            except ImportError:
                print("tensorflow not installed - skipping neural net")
                continue
        elif key == "cat":
            try:
                proba = run_catboost(train, test, Xtr, Xte, ytr, sw)
                pred = (proba >= 0.5).astype(int)
            except ImportError:
                print("catboost not installed - skipping")
                continue
        rows.append(report(name, yte, pred, proba=proba, score=score))
        print(f"done: {name} ({time.time() - t0:.0f}s)")
    return rows, gb_model, gb_pred


# --------------------------------------------------------------------------
# 7. Seed study (paper Figures 2 and 3)
# --------------------------------------------------------------------------
FAST_LGBM = dict(n_estimators=150, learning_rate=0.1, num_leaves=63, max_depth=12,
                 colsample_bytree=0.4, max_bin=255)  # --fast: much cheaper than the paper's settings


def run_tag(args):
    return (args.split + ("_smooth" if args.smooth else "") + ("_loggoal" if args.log_goal else "")
            + ("_extra" if args.extra_text else "") + ("_fast" if args.fast else ""))


def plot_seed_study(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = sorted(res["sample"].unique())
    n = res["seed"].nunique()
    for ax, metric in zip(axes, ["accuracy", "f1"]):
        ax.boxplot([res.loc[res["sample"] == l, metric] for l in labels])
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([SAMPLE_NAMES[l] for l in labels], rotation=25, ha="right")
        ax.set_title(f"Gradient boosting {metric} ({n} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def seed_study(df, args, cache):
    import lightgbm as lgb
    tag = run_tag(args)
    csv_path = os.path.join(args.out_dir, f"seed_study_{tag}.csv")
    png_path = csv_path.replace(".csv", ".png")
    rows, done = [], set()
    if os.path.exists(csv_path):
        if args.resume:
            rows = pd.read_csv(csv_path).to_dict("records")
            done = {(int(r["seed"]), int(r["sample"])) for r in rows}
            print(f"resuming: {len(done)} (seed, sample) runs already in {csv_path}")
        else:
            print(f"WARNING: {csv_path} exists and will be overwritten (use --resume to continue it)")
    over = FAST_LGBM if args.fast else {}
    if args.fast:
        print("--fast: reduced LightGBM settings (fine for comparing samples, not for paper numbers)")
    t0, finished = time.time(), 0
    for seed in range(args.n_seeds):
        todo = [s for s in args.samples if (seed, s) not in done]
        if not todo:
            continue
        train, test = make_split(df, args.split, seed)
        feats = build_features_multi(train, test, todo, args, cache, seed=seed)
        for s in todo:
            Xtr, Xte = feats[s]
            m = lgb.LGBMClassifier(**lgbm_params({**over, "random_state": seed}))
            m.fit(Xtr, train["y"].values)
            p = m.predict(Xte)
            rows.append({"seed": seed, "sample": s,
                         "accuracy": accuracy_score(test["y"], p), "f1": f1_score(test["y"], p)})
            pd.DataFrame(rows).to_csv(csv_path, index=False)  # saved after every single fit
            print(f"seed {seed} sample {s}: acc {rows[-1]['accuracy']:.4f} f1 {rows[-1]['f1']:.4f}")
        finished += 1
        plot_seed_study(pd.DataFrame(rows), png_path)  # PNG is always up to date
        left = sum(1 for sd in range(args.n_seeds) if any((sd, s) not in done and sd > seed for s in args.samples))
        per = (time.time() - t0) / finished
        print(f"seed {seed} done | {per / 60:.1f} min/seed | ~{per * left / 60:.0f} min left")
    res = pd.DataFrame(rows)
    print(res.groupby("sample")[["accuracy", "f1"]].agg(["mean", "std"]).round(4))
    print(f"results: {csv_path}\nplot: {png_path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_dir", default="./outputs")
    ap.add_argument("--min_date", default="2019-04", help="first snapshot month (YYYY-MM)")
    ap.add_argument("--max_date", default="2021-04", help="last snapshot month (YYYY-MM)")
    ap.add_argument("--dedup", choices=["paper", "id"], default="paper")
    ap.add_argument("--sample", type=int, default=2, choices=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--split", choices=["random", "time", "group"], default="random")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smooth", type=float, default=None, help="m-estimate smoothing; omit for paper cutoff")
    ap.add_argument("--log_goal", action="store_true")
    ap.add_argument("--extra_text", action="store_true")
    ap.add_argument("--models", default="gb,rf,logit,svm,ridge,ols,lasso,nn",
                    help="comma list from: " + ",".join(MODEL_KEYS))
    ap.add_argument("--nn_batch", type=int, default=256, help="paper: 3 (very slow)")
    ap.add_argument("--tune", type=int, default=0, help="Optuna trials for LightGBM (0 = paper params)")
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--seed_study", type=int, default=0, metavar="N_SEEDS")
    ap.add_argument("--samples", default="1,2,3,4", help="samples for --seed_study")
    ap.add_argument("--resume", action="store_true", help="--seed_study: continue from the saved csv")
    ap.add_argument("--fast", action="store_true",
                    help="--seed_study: smaller/faster LightGBM (relative comparisons only)")
    ap.add_argument("--plot_only", action="store_true",
                    help="redraw the seed-study plot from the saved csv (pass the same split/"
                         "smooth/log_goal/extra_text/fast flags); loads no data")
    args = ap.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.samples = [int(s) for s in args.samples.split(",")]
    args.n_seeds = args.seed_study
    bad = [m for m in args.models if m not in MODEL_KEYS]
    assert not bad, f"unknown model keys {bad}"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.plot_only:
        path = os.path.join(args.out_dir, f"seed_study_{run_tag(args)}.csv")
        plot_seed_study(pd.read_csv(path), path.replace(".csv", ".png"))
        print("plot written:", path.replace(".csv", ".png"))
        return

    df = load_data(args)
    cache = RowCache(df, args.out_dir)

    if args.seed_study:
        seed_study(df, args, cache)
        return
    if args.fast:
        print("note: --fast only applies to --seed_study; ignoring it for this run")
        args.fast = False

    train, test = make_split(df, args.split, args.seed)
    Xtr, Xte = build_features(train, test, args.sample, args, cache, seed=args.seed)
    print(f"Sample {args.sample} ({SAMPLE_NAMES[args.sample]}): {Xtr.shape[1]} features")

    rows, gb_model, gb_pred = run_models(train, test, Xtr, Xte, args)
    out = pd.DataFrame(rows).sort_values("A", ascending=False).round(3)
    print("\n", out.to_string(index=False))
    tag = f"sample{args.sample}_{run_tag(args)}"
    out.to_csv(os.path.join(args.out_dir, f"results_{tag}.csv"), index=False)

    if args.plots and gb_model is not None:
        plot_confusion(test["y"].values, gb_pred, os.path.join(args.out_dir, f"confusion_{tag}.png"))
        plot_importance(gb_model, list(Xtr.columns), os.path.join(args.out_dir, f"importance_{tag}.png"))
        if args.shap:
            plot_shap(gb_model, Xte, os.path.join(args.out_dir, f"shap_{tag}.png"))
        print(f"plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
