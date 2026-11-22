import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from urllib.parse import urlparse

from features import FEATURE_NAMES, explain, extract_features

BASE = Path(__file__).parent
app = Flask(__name__)
MODEL_PATH = BASE / "models" / "phishing_rf.joblib"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
saved_features = bundle.get("features", FEATURE_NAMES)
if list(saved_features) != FEATURE_NAMES:
    raise RuntimeError("Model feature order does not match features.py. Do not run predictions until this is resolved.")
if 1 not in list(model.classes_):
    raise RuntimeError("Loaded model does not contain phishing class label 1.")
PHISHING_INDEX = list(model.classes_).index(1)
DB = BASE / "history.db"

# Exact-host homepage exceptions for a small student-project demo list.
# Subdomains and URLs with paths/query strings are intentionally excluded.
TRUSTED_HOME_HOSTS = {
    "google.com", "www.google.com",
    "wikipedia.org", "www.wikipedia.org",
    "microsoft.com", "www.microsoft.com",
    "apple.com", "www.apple.com",
    "amazon.com", "www.amazon.com",
    "youtube.com", "www.youtube.com",
    "github.com", "www.github.com",
    "mozilla.org", "www.mozilla.org",
    "python.org", "www.python.org",
    "openai.com", "www.openai.com",
}


def is_trusted_homepage(url: str) -> bool:
    """Match only an exact hostname at its root homepage."""
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = (parsed.hostname or "").lower()
        return (
            host in TRUSTED_HOME_HOSTS
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
            and parsed.port in (None, 80, 443)
        )
    except ValueError:
        return False


def db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY, url TEXT, score INTEGER, verdict TEXT,
        ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    return con


def analyze(url: str) -> dict:
    feats = extract_features(url)

    # Work around known false positives on a small set of exact root homepages.
    # This does not whitelist subdomains (e.g. docs.google.com) or other paths.
    if is_trusted_homepage(url):
        return {
            "url": url,
            "score": 0,
            "verdict": "Looks safe (known homepage)",
            "level": "safe",
            "warnings": [
                "Exact homepage hostname matches the app's small built-in trusted list.",
                "This list does not verify page content and is not a guarantee of safety."
            ],
            "notice": "Educational demonstration only. The app does not verify live domain reputation or page content."
        }

    row = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    score = round(float(model.predict_proba(row)[0][PHISHING_INDEX]) * 100)
    if score >= 70:
        verdict, level = "Likely phishing", "danger"
    elif score >= 40:
        verdict, level = "Suspicious", "warn"
    else:
        verdict, level = "Lower detected risk", "safe"
    return {"url": url, "score": score, "verdict": verdict, "level": level,
            "warnings": explain(feats),
            "notice": "This is a machine-learning estimate based on URL structure, not a verified reputation check. A low score does not guarantee safety."}


def save(result):
    with db() as con:
        con.execute("INSERT INTO history(url,score,verdict) VALUES(?,?,?)",
                    (result["url"], result["score"], result["verdict"]))


def recent(n=8):
    with db() as con:
        return con.execute("SELECT url,score,verdict FROM history ORDER BY id DESC LIMIT ?", (n,)).fetchall()


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if not url or len(url) > 2048:
            error = "Enter a URL (up to 2048 characters)."
        else:
            try:
                result = analyze(url)
                save(result)
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("URL analysis failed")
                error = "Could not analyse this URL. Check the input and try again."
    return render_template("index.html", result=result, error=error, history=recent())


@app.route("/api/check", methods=["POST"])
def api_check():
    payload = request.get_json(silent=True) or {}
    url = payload.get("url", "")
    if not isinstance(url, str) or not url.strip():
        return jsonify(error="Missing or invalid 'url'"), 400
    if len(url) > 2048:
        return jsonify(error="URL is too long (maximum 2048 characters)."), 400
    try:
        result = analyze(url.strip())
        save(result)
        return jsonify(result)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/history", methods=["GET"])
def api_history():
    with db() as con:
        rows = con.execute("SELECT url,score,verdict,ts FROM history ORDER BY id DESC LIMIT 20").fetchall()
    return jsonify([{"url": r[0], "score": r[1], "verdict": r[2], "timestamp": r[3]} for r in rows])


if __name__ == "__main__":
    app.run(debug=True)
