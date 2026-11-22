"""Train + compare models. Works locally or in Google Colab.

Dataset: CSV with columns `url,label` (label: phishing/bad/1 vs legitimate/good/0).
Good sources: PhishTank or OpenPhish (phishing) + Tranco top sites (legitimate),
or the Kaggle "Phishing Site URLs" dataset.

Usage:  python train_model.py --data data/urls.csv
        python train_model.py --demo      # tiny synthetic data, smoke test only
"""
import argparse, random, string
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_NAMES, extract_features


def demo_data(n=1500, seed=0):
    rnd = random.Random(seed)
    rw = lambda k: "".join(rnd.choices(string.ascii_lowercase, k=k))
    good = [f"https://www.{rw(rnd.randint(5, 10))}.com/{rw(6)}" for _ in range(n)]
    bad = []
    for _ in range(n):
        kind = rnd.random()
        if kind < .3:
            bad.append(f"http://{'.'.join(str(rnd.randint(1, 254)) for _ in range(4))}/login/verify")
        elif kind < .6:
            bad.append(f"http://secure-login.{rw(6)}-account.{rw(5)}.xyz/verify?id={rnd.randint(100000, 999999999)}")
        else:
            bad.append(f"http://bit.ly/{rw(7)}" if rnd.random() < .5 else f"http://{rw(8)}.tk/update/password")
    return pd.DataFrame({"url": good + bad, "label": [0] * n + [1] * n})


def load(path):
    df = pd.read_csv(path).dropna(subset=["url", "label"]).drop_duplicates("url")
    m = {"phishing": 1, "bad": 1, "1": 1, "malicious": 1, "legitimate": 0, "good": 0, "0": 0, "benign": 0}
    df["label"] = df["label"].astype(str).str.lower().map(m)
    return df.dropna(subset=["label"]).astype({"label": int})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    df = demo_data() if a.demo or not a.data else load(a.data)
    if a.demo or not a.data:
        print("!! Using SYNTHETIC demo data: accuracy numbers are meaningless.\n")

    X = pd.DataFrame([extract_features(u) for u in df["url"]])[FEATURE_NAMES]
    y = df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                                class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True)),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6, eval_metric="logloss")
    except ImportError:
        pass

    scores = {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        scores[name] = mdl.score(Xte, yte)
        print(f"{name:20s} accuracy = {scores[name]:.4f}")

    rf = models["Random Forest"]          # shipped model; swap if another wins on your data
    print("\nRandom Forest report\n", classification_report(yte, rf.predict(Xte),
          target_names=["legitimate", "phishing"]))
    print("Confusion matrix\n", confusion_matrix(yte, rf.predict(Xte)))
    imp = sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda t: -t[1])[:8]
    print("\nTop features:", *[f"\n  {n}: {v:.3f}" for n, v in imp])

    joblib.dump({"model": rf, "features": FEATURE_NAMES}, "models/phishing_rf.joblib")
    print("\nSaved models/phishing_rf.joblib")


if __name__ == "__main__":
    main()
